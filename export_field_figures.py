#!/usr/bin/env python3
"""Export field-variable figures for the FEniCSx high-performance cases.

The script intentionally avoids reading XDMF through PyVista/VTK. It recomputes
small visualization-sized solutions in memory and exports Matplotlib PNG files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import ufl
from basix.ufl import element
from mpi4py import MPI
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from petsc4py import PETSc

from dolfinx import default_real_type, fem
from dolfinx.fem import Constant, Expression, Function, dirichletbc, extract_function_spaces
from dolfinx.fem import form, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem, apply_lifting, assemble_matrix, assemble_vector
from dolfinx.fem.petsc import create_vector
from dolfinx.mesh import CellType, GhostMode, create_box, create_rectangle, locate_entities_boundary

from case1_elasticity_amg import build_nullspace, interpolation_points


DTYPE = PETSc.ScalarType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PNG field visualizations.")
    parser.add_argument(
        "--output-dir",
        default="fenics_hpc_cases/docs/figures",
        help="Directory where PNG figures are written.",
    )
    parser.add_argument("--case1-nx", type=int, default=8)
    parser.add_argument("--case1-ny", type=int, default=5)
    parser.add_argument("--case1-nz", type=int, default=5)
    parser.add_argument("--case2-nx", type=int, default=32)
    parser.add_argument("--case2-ny", type=int, default=32)
    parser.add_argument(
        "--warp-scale",
        type=float,
        default=75.0,
        help="Displacement amplification factor for case 1 deformation plots.",
    )
    return parser.parse_args()


def cell_vertices(mesh):
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    conn = mesh.topology.connectivity(tdim, 0)
    offsets = conn.offsets
    return [conn.array[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def solve_elasticity(nx: int, ny: int, nz: int):
    comm = MPI.COMM_WORLD
    msh = create_box(
        comm,
        [np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])],
        (nx, ny, nz),
        CellType.tetrahedron,
        ghost_mode=GhostMode.shared_facet,
    )

    V = functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    E = 1.0e9
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def strain(w):
        return ufl.sym(ufl.grad(w))

    def stress(w):
        return 2.0 * mu * strain(w) + lambda_ * ufl.tr(strain(w)) * ufl.Identity(len(w))

    x = ufl.SpatialCoordinate(msh)
    body_force = ufl.as_vector((10.0 * 300.0**2 * x[0], 10.0 * 300.0**2 * x[1], 0.0))
    a = form(ufl.inner(stress(u), strain(v)) * ufl.dx)
    L = form(ufl.inner(body_force, v) * ufl.dx)

    fdim = msh.topology.dim - 1
    fixed_facets = locate_entities_boundary(
        msh,
        fdim,
        marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], 1.0),
    )
    bc = dirichletbc(
        np.zeros(3, dtype=DTYPE),
        locate_dofs_topological(V, entity_dim=fdim, entities=fixed_facets),
        V=V,
    )

    A = assemble_matrix(a, bcs=[bc])
    A.assemble()
    A.setNearNullSpace(build_nullspace(V))
    A.setOption(PETSc.Mat.Option.SPD, True)

    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bc.set(b.array_w)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.setTolerances(rtol=1.0e-8, max_it=500)
    solver.getPC().setType("gamg")
    solver.setFromOptions()

    uh = Function(V)
    uh.name = "displacement"
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    if solver.getConvergedReason() <= 0:
        raise RuntimeError("Elasticity solver did not converge")

    sigma_dev = stress(uh) - (1.0 / 3.0) * ufl.tr(stress(uh)) * ufl.Identity(len(uh))
    sigma_vm = ufl.sqrt((3.0 / 2.0) * ufl.inner(sigma_dev, sigma_dev))
    W = functionspace(msh, ("Discontinuous Lagrange", 0))
    sigma_vm_h = Function(W)
    sigma_vm_h.name = "von_mises_stress"
    sigma_vm_h.interpolate(Expression(sigma_vm, interpolation_points(W.element)))
    return msh, V, uh, sigma_vm_h


def solve_stokes(nx: int, ny: int):
    comm = MPI.COMM_WORLD
    msh = create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        (nx, ny),
        CellType.triangle,
    )
    gdim = msh.geometry.dim

    def noslip_boundary(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)

    def lid(x):
        return np.isclose(x[1], 1.0)

    def lid_velocity_expression(x):
        return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))

    P2 = element("Lagrange", msh.basix_cell(), degree=2, shape=(gdim,), dtype=default_real_type)
    P1 = element("Lagrange", msh.basix_cell(), degree=1, dtype=default_real_type)
    V = functionspace(msh, P2)
    Q = functionspace(msh, P1)

    fdim = msh.topology.dim - 1
    facets = locate_entities_boundary(msh, fdim, noslip_boundary)
    bc0 = dirichletbc(np.zeros(gdim, dtype=PETSc.ScalarType), locate_dofs_topological(V, fdim, facets), V)

    lid_velocity = Function(V)
    lid_velocity.interpolate(lid_velocity_expression)
    facets = locate_entities_boundary(msh, fdim, lid)
    bc1 = dirichletbc(lid_velocity, locate_dofs_topological(V, fdim, facets))
    bcs = [bc0, bc1]

    u = ufl.TrialFunction(V)
    p = ufl.TrialFunction(Q)
    v = ufl.TestFunction(V)
    q = ufl.TestFunction(Q)
    body_force = Constant(msh, (PETSc.ScalarType(0.0), PETSc.ScalarType(0.0)))

    a_ufl = [
        [ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, ufl.inner(p, ufl.div(v)) * ufl.dx],
        [ufl.inner(ufl.div(u), q) * ufl.dx, None],
    ]
    L_ufl = [ufl.inner(body_force, v) * ufl.dx, ufl.ZeroBaseForm((q,))]
    a = form(a_ufl)
    L = form(L_ufl)
    a_p11 = form(ufl.inner(p, q) * ufl.dx)
    a_p = [[a[0][0], None], [None, a_p11]]

    problem = LinearProblem(
        a_ufl,
        L_ufl,
        kind="nest",
        bcs=bcs,
        P=a_p,
        petsc_options_prefix="stokes_fieldsplit_figures_",
        petsc_options={
            "ksp_type": "minres",
            "ksp_rtol": 1.0e-9,
            "ksp_max_it": 500,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "additive",
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "gamg",
            "fieldsplit_1_ksp_type": "preonly",
            "fieldsplit_1_pc_type": "jacobi",
        },
    )

    null_vec = create_vector(extract_function_spaces(L), "nest")
    null_vecs = null_vec.getNestSubVecs()
    null_vecs[0].set(0.0)
    null_vecs[1].set(1.0)
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    problem.A.setNullSpace(nsp)
    problem.A.getNestSubMatrix(0, 0).setOption(PETSc.Mat.Option.SPD, True)
    problem.P_mat.getNestSubMatrix(0, 0).setOption(PETSc.Mat.Option.SPD, True)
    problem.P_mat.getNestSubMatrix(1, 1).setOption(PETSc.Mat.Option.SPD, True)

    u_h, p_h = problem.solve()
    if problem.solver.getConvergedReason() <= 0:
        raise RuntimeError("Stokes solver did not converge")
    u_h.x.scatter_forward()
    p_h.x.scatter_forward()
    return msh, V, Q, u_h, p_h


def plot_case1(output_dir: Path, nx: int, ny: int, nz: int, warp_scale: float) -> list[Path]:
    msh, V, uh, sigma_vm_h = solve_elasticity(nx, ny, nz)
    coords = V.tabulate_dof_coordinates()[:, :3]
    disp = np.real(uh.x.array).reshape((-1, 3))
    disp_mag = np.linalg.norm(disp, axis=1)
    warped = coords + warp_scale * disp

    fig = plt.figure(figsize=(7.2, 5.4), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        warped[:, 0],
        warped[:, 1],
        warped[:, 2],
        c=disp_mag,
        s=15,
        cmap="viridis",
        depthshade=False,
    )
    ax.set_title(f"Case 1 displacement magnitude, warped x{warp_scale:g}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-58)
    fig.colorbar(sc, ax=ax, shrink=0.72, label="|u|")
    fig.tight_layout()
    disp_path = output_dir / "case1_displacement_warp.png"
    fig.savefig(disp_path, bbox_inches="tight")
    plt.close(fig)

    verts = cell_vertices(msh)
    x = msh.geometry.x[:, :3]
    centers = np.array([x[v].mean(axis=0) for v in verts])
    sigma = np.real(sigma_vm_h.x.array[: len(centers)])
    center_disp = np.array([disp[v].mean(axis=0) for v in verts])
    centers_warped = centers + warp_scale * center_disp

    fig = plt.figure(figsize=(7.2, 5.4), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        centers_warped[:, 0],
        centers_warped[:, 1],
        centers_warped[:, 2],
        c=sigma,
        s=9,
        cmap="magma",
        depthshade=False,
    )
    ax.set_title(f"Case 1 Von Mises stress, warped x{warp_scale:g}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-58)
    fig.colorbar(sc, ax=ax, shrink=0.72, label="Von Mises stress")
    fig.tight_layout()
    vm_path = output_dir / "case1_von_mises.png"
    fig.savefig(vm_path, bbox_inches="tight")
    plt.close(fig)
    return [disp_path, vm_path]


def plot_case2(output_dir: Path, nx: int, ny: int) -> list[Path]:
    _msh, V, Q, u_h, p_h = solve_stokes(nx, ny)

    u_coords = V.tabulate_dof_coordinates()[:, :2]
    u_vals = np.real(u_h.x.array).reshape((-1, 2))
    speed = np.linalg.norm(u_vals, axis=1)
    tri_u = mtri.Triangulation(u_coords[:, 0], u_coords[:, 1])

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=180)
    contour = ax.tricontourf(tri_u, speed, levels=24, cmap="viridis")
    stride = max(1, len(u_coords) // 320)
    ax.quiver(
        u_coords[::stride, 0],
        u_coords[::stride, 1],
        u_vals[::stride, 0],
        u_vals[::stride, 1],
        color="white",
        alpha=0.75,
        scale=28,
        width=0.003,
    )
    ax.set_aspect("equal")
    ax.set_title("Case 2 velocity magnitude and direction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(contour, ax=ax, label="|u|")
    fig.tight_layout()
    velocity_path = output_dir / "case2_velocity.png"
    fig.savefig(velocity_path, bbox_inches="tight")
    plt.close(fig)

    p_coords = Q.tabulate_dof_coordinates()[:, :2]
    pressure = np.real(p_h.x.array)
    tri_p = mtri.Triangulation(p_coords[:, 0], p_coords[:, 1])
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=180)
    contour = ax.tricontourf(tri_p, pressure, levels=24, cmap="coolwarm")
    ax.tricontour(tri_p, pressure, levels=12, colors="black", linewidths=0.35, alpha=0.45)
    ax.set_aspect("equal")
    ax.set_title("Case 2 pressure field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(contour, ax=ax, label="pressure")
    fig.tight_layout()
    pressure_path = output_dir / "case2_pressure.png"
    fig.savefig(pressure_path, bbox_inches="tight")
    plt.close(fig)
    return [velocity_path, pressure_path]


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("Figure export is serial-only. Run without mpirun.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.extend(plot_case1(output_dir, args.case1_nx, args.case1_ny, args.case1_nz, args.warp_scale))
    paths.extend(plot_case2(output_dir, args.case2_nx, args.case2_ny))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
