#!/usr/bin/env python3
"""Taylor-Hood Stokes solver using PETSc MatNest and fieldsplit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ufl
from basix.ufl import element
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import default_real_type, fem, la
from dolfinx.fem import Constant, Function, bcs_by_block, dirichletbc, extract_function_spaces
from dolfinx.fem import form, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem, create_vector
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a lid-driven cavity Stokes problem with PETSc fieldsplit."
    )
    parser.add_argument("--small", action="store_true", help="Use a fast smoke-test mesh.")
    parser.add_argument("--nx", type=int, default=32, help="Cells in the x direction.")
    parser.add_argument("--ny", type=int, default=32, help="Cells in the y direction.")
    parser.add_argument("--rtol", type=float, default=1.0e-9, help="KSP relative tolerance.")
    parser.add_argument("--max-it", type=int, default=500, help="KSP maximum iterations.")
    parser.add_argument("--monitor", action="store_true", help="Print KSP residual history.")
    parser.add_argument("--view-solver", action="store_true", help="Print PETSc solver setup.")
    parser.add_argument(
        "--output-dir",
        default="fenics_hpc_cases/results/case2_stokes_fieldsplit",
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
        args.nx, args.ny = 8, 8
    return args


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    outdir = Path(args.output_dir)
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    msh = create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        (args.nx, args.ny),
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
    noslip = np.zeros(gdim, dtype=PETSc.ScalarType)
    facets = locate_entities_boundary(msh, fdim, noslip_boundary)
    bc0 = dirichletbc(noslip, locate_dofs_topological(V, fdim, facets), V)

    lid_velocity = Function(V)
    lid_velocity.interpolate(lid_velocity_expression)
    facets = locate_entities_boundary(msh, fdim, lid)
    bc1 = dirichletbc(lid_velocity, locate_dofs_topological(V, fdim, facets))
    bcs = [bc0, bc1]

    u = ufl.TrialFunction(V)
    p = ufl.TrialFunction(Q)
    v = ufl.TestFunction(V)
    q = ufl.TestFunction(Q)
    body_force = Constant(
        msh,
        (PETSc.ScalarType(0.0), PETSc.ScalarType(0.0)),
    )

    a_ufl = [
        [ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, ufl.inner(p, ufl.div(v)) * ufl.dx],
        [ufl.inner(ufl.div(u), q) * ufl.dx, None],
    ]
    L_ufl = [ufl.inner(body_force, v) * ufl.dx, ufl.ZeroBaseForm((q,))]
    a = form(a_ufl)
    L = form(L_ufl)

    a_p11 = form(ufl.inner(p, q) * ufl.dx)
    a_p = [[a[0][0], None], [None, a_p11]]

    comm.Barrier()
    t0 = MPI.Wtime()

    problem = LinearProblem(
        a_ufl,
        L_ufl,
        kind="nest",
        bcs=bcs,
        P=a_p,
        petsc_options_prefix="stokes_fieldsplit_",
        petsc_options={
            "ksp_type": "minres",
            "ksp_rtol": args.rtol,
            "ksp_max_it": args.max_it,
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

    A00 = problem.A.getNestSubMatrix(0, 0)
    A00.setOption(PETSc.Mat.Option.SPD, True)
    P00 = problem.P_mat.getNestSubMatrix(0, 0)
    P11 = problem.P_mat.getNestSubMatrix(1, 1)
    P00.setOption(PETSc.Mat.Option.SPD, True)
    P11.setOption(PETSc.Mat.Option.SPD, True)

    residuals: list[float] = []
    if args.monitor:
        def monitor(_ksp, its, rnorm):
            residuals.append(float(rnorm))
            if comm.rank == 0:
                print(f"Iteration: {its}, residual: {rnorm:.6e}")

        problem.solver.setMonitor(monitor)

    u_h, p_h = problem.solve()
    reason = problem.solver.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"PETSc KSP did not converge, reason={reason}")
    nullspace_ok = bool(nsp.test(problem.A))

    if args.view_solver:
        problem.solver.view()

    u_h.name = "velocity_p2"
    p_h.name = "pressure"
    u_h.x.scatter_forward()
    p_h.x.scatter_forward()

    div_l2_local = fem.assemble_scalar(fem.form(ufl.inner(ufl.div(u_h), ufl.div(u_h)) * ufl.dx))
    div_l2 = float(np.sqrt(comm.allreduce(div_l2_local, op=MPI.SUM)))

    if args.write_xdmf:
        P1_vec = element("Lagrange", msh.basix_cell(), degree=1, shape=(gdim,), dtype=default_real_type)
        V_out = functionspace(msh, P1_vec)
        u_out = Function(V_out)
        u_out.name = "velocity"
        u_out.interpolate(u_h)
        with XDMFFile(comm, str(outdir / "velocity.xdmf"), "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(u_out)
        with XDMFFile(comm, str(outdir / "pressure.xdmf"), "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(p_h)

    comm.Barrier()
    elapsed = MPI.Wtime() - t0

    cell_map = msh.topology.index_map(msh.topology.dim)
    global_cells = cell_map.size_global if cell_map is not None else -1
    velocity_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    pressure_dofs = Q.dofmap.index_map.size_global * Q.dofmap.index_map_bs

    metrics = {
        "case": "taylor_hood_stokes_fieldsplit",
        "mpi_size": comm.size,
        "mesh": [args.nx, args.ny],
        "global_cells": int(global_cells),
        "velocity_dofs": int(velocity_dofs),
        "pressure_dofs": int(pressure_dofs),
        "total_dofs": int(velocity_dofs + pressure_dofs),
        "ksp_type": problem.solver.getType(),
        "pc_type": problem.solver.getPC().getType(),
        "iterations": int(problem.solver.getIterationNumber()),
        "converged_reason": int(reason),
        "nullspace_test": nullspace_ok,
        "velocity_norm": float(la.norm(u_h.x)),
        "pressure_norm": float(la.norm(p_h.x)),
        "divergence_l2": div_l2,
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
