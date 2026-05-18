#!/usr/bin/env python3
"""3D linear elasticity solved with PETSc CG + GAMG in DOLFINx."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import la
from dolfinx.fem import Expression, Function, FunctionSpace, dirichletbc, form, functionspace
from dolfinx.fem import locate_dofs_topological
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, GhostMode, create_box, locate_entities_boundary


DTYPE = PETSc.ScalarType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a 3D linear elasticity problem with PETSc CG + GAMG."
    )
    parser.add_argument("--small", action="store_true", help="Use a fast smoke-test mesh.")
    parser.add_argument("--nx", type=int, default=16, help="Cells in the x direction.")
    parser.add_argument("--ny", type=int, default=16, help="Cells in the y direction.")
    parser.add_argument("--nz", type=int, default=16, help="Cells in the z direction.")
    parser.add_argument("--length", type=float, default=2.0, help="Beam/domain length.")
    parser.add_argument("--width", type=float, default=1.0, help="Beam/domain width.")
    parser.add_argument("--height", type=float, default=1.0, help="Beam/domain height.")
    parser.add_argument("--rtol", type=float, default=1.0e-8, help="KSP relative tolerance.")
    parser.add_argument("--max-it", type=int, default=500, help="KSP maximum iterations.")
    parser.add_argument("--monitor", action="store_true", help="Print KSP residual history.")
    parser.add_argument("--view-solver", action="store_true", help="Print PETSc solver setup.")
    parser.add_argument(
        "--output-dir",
        default="fenics_hpc_cases/results/case1_elasticity_amg",
        help="Directory for metrics and XDMF files.",
    )
    parser.add_argument(
        "--no-write-xdmf",
        dest="write_xdmf",
        action="store_false",
        help="Skip XDMF field output.",
    )
    parser.set_defaults(write_xdmf=True)
    args = parser.parse_args()

    if args.small:
        args.nx, args.ny, args.nz = 4, 4, 4
    return args


def interpolation_points(element):
    points = element.interpolation_points
    return points() if callable(points) else points


def build_nullspace(V: FunctionSpace) -> PETSc.NullSpace:
    """Build rigid-body near-nullspace for 3D elasticity."""
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=DTYPE) for _ in range(6)]
    arrays = [vec.array for vec in basis]

    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    for i in range(3):
        arrays[i][dofs[i]] = 1.0

    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]

    arrays[3][dofs[0]] = -x1
    arrays[3][dofs[1]] = x0
    arrays[4][dofs[0]] = x2
    arrays[4][dofs[2]] = -x0
    arrays[5][dofs[2]] = x1
    arrays[5][dofs[1]] = -x2

    la.orthonormalize(basis)
    basis_petsc = [
        PETSc.Vec().createWithArray(vec.array[: bs * length0], bsize=3, comm=V.mesh.comm)
        for vec in basis
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    outdir = Path(args.output_dir)
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    mesh_start = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    mesh_end = np.array([args.length, args.width, args.height], dtype=np.float64)
    msh = create_box(
        comm,
        [mesh_start, mesh_end],
        (args.nx, args.ny, args.nz),
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

    rho = 10.0
    omega = 300.0
    x = ufl.SpatialCoordinate(msh)
    body_force = ufl.as_vector((rho * omega**2 * x[0], rho * omega**2 * x[1], 0.0))

    a = form(ufl.inner(stress(u), strain(v)) * ufl.dx)
    L = form(ufl.inner(body_force, v) * ufl.dx)

    fdim = msh.topology.dim - 1
    fixed_facets = locate_entities_boundary(
        msh,
        fdim,
        marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], args.width),
    )
    bc = dirichletbc(
        np.zeros(3, dtype=DTYPE),
        locate_dofs_topological(V, entity_dim=fdim, entities=fixed_facets),
        V=V,
    )

    comm.Barrier()
    t0 = MPI.Wtime()

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
    solver.setTolerances(rtol=args.rtol, max_it=args.max_it)
    solver.getPC().setType("gamg")
    solver.setFromOptions()

    residuals: list[float] = []
    if args.monitor:
        def monitor(_ksp, its, rnorm):
            residuals.append(float(rnorm))
            if comm.rank == 0:
                print(f"Iteration: {its}, residual: {rnorm:.6e}")

        solver.setMonitor(monitor)

    uh = Function(V)
    uh.name = "displacement"
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    reason = solver.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"PETSc KSP did not converge, reason={reason}")

    if args.view_solver:
        solver.view()

    sigma_dev = stress(uh) - (1.0 / 3.0) * ufl.tr(stress(uh)) * ufl.Identity(len(uh))
    sigma_vm = ufl.sqrt((3.0 / 2.0) * ufl.inner(sigma_dev, sigma_dev))
    W = functionspace(msh, ("Discontinuous Lagrange", 0))
    sigma_vm_h = Function(W)
    sigma_vm_h.name = "von_mises_stress"
    sigma_vm_h.interpolate(Expression(sigma_vm, interpolation_points(W.element)))

    comm.Barrier()
    elapsed = MPI.Wtime() - t0

    if args.write_xdmf:
        with XDMFFile(comm, str(outdir / "displacement.xdmf"), "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(uh)
        with XDMFFile(comm, str(outdir / "von_mises_stress.xdmf"), "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(sigma_vm_h)

    cell_map = msh.topology.index_map(msh.topology.dim)
    global_cells = cell_map.size_global if cell_map is not None else -1
    global_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    solution_norm = la.norm(uh.x)

    metrics = {
        "case": "3d_linear_elasticity_amg",
        "mpi_size": comm.size,
        "mesh": [args.nx, args.ny, args.nz],
        "global_cells": int(global_cells),
        "global_dofs": int(global_dofs),
        "ksp_type": solver.getType(),
        "pc_type": solver.getPC().getType(),
        "iterations": int(solver.getIterationNumber()),
        "converged_reason": int(reason),
        "solution_norm": float(solution_norm),
        "elapsed_seconds": float(elapsed),
        "write_xdmf": bool(args.write_xdmf),
        "output_dir": str(outdir),
    }
    if residuals:
        metrics["last_residual"] = residuals[-1]

    if comm.rank == 0:
        print("METRICS " + json.dumps(metrics, ensure_ascii=False, sort_keys=True))
        with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
