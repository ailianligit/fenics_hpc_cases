#!/usr/bin/env python3
"""Preprocessing diagnostic for the Taylor-Hood Stokes FieldSplit case."""

from __future__ import annotations

import argparse
import json

import numpy as np
import ufl
from basix.ufl import element
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import default_real_type
from dolfinx.fem import Constant, Function, dirichletbc, form, functionspace, locate_dofs_topological
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preprocessing diagnostics for case 2.")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD

    comm.Barrier()
    t0 = MPI.Wtime()

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
    noslip_facets = locate_entities_boundary(msh, fdim, noslip_boundary)
    lid_facets = locate_entities_boundary(msh, fdim, lid)

    noslip = np.zeros(gdim, dtype=PETSc.ScalarType)
    noslip_dofs = locate_dofs_topological(V, fdim, noslip_facets)
    bc0 = dirichletbc(noslip, noslip_dofs, V)

    lid_velocity = Function(V)
    lid_velocity.interpolate(lid_velocity_expression)
    lid_dofs = locate_dofs_topological(V, fdim, lid_facets)
    bc1 = dirichletbc(lid_velocity, lid_dofs)
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

    # Keep variables live so this diagnostic exercises the full preprocessing path.
    _ = (bcs, a, L, a_p)

    comm.Barrier()
    elapsed = MPI.Wtime() - t0
    elapsed_max = comm.allreduce(elapsed, op=MPI.MAX)

    cell_map = msh.topology.index_map(msh.topology.dim)
    vertex_map = msh.topology.index_map(0)
    v_bs = V.dofmap.index_map_bs
    q_bs = Q.dofmap.index_map_bs

    expected_vertices = (args.nx + 1) * (args.ny + 1)
    expected_cells = 2 * args.nx * args.ny
    expected_velocity_scalar_dofs = 2 * (2 * args.nx + 1) * (2 * args.ny + 1)
    expected_pressure_dofs = expected_vertices
    expected_noslip_vertices = (args.ny + 1) + (args.ny + 1) + (args.nx + 1) - 2
    expected_lid_vertices = args.nx + 1
    expected_noslip_p2_block_dofs = (2 * args.ny + 1) + (2 * args.ny + 1) + (2 * args.nx + 1) - 2
    expected_lid_p2_block_dofs = 2 * args.nx + 1

    rank_data = {
        "rank": comm.rank,
        "local_cells": int(cell_map.size_local),
        "local_vertices": int(vertex_map.size_local),
        "local_velocity_dofs": int(V.dofmap.index_map.size_local * v_bs),
        "local_pressure_dofs": int(Q.dofmap.index_map.size_local * q_bs),
        "local_noslip_facets": int(len(noslip_facets)),
        "local_lid_facets": int(len(lid_facets)),
        "local_noslip_block_dofs": int(len(noslip_dofs)),
        "local_lid_block_dofs": int(len(lid_dofs)),
        "local_noslip_scalar_components": int(len(noslip_dofs) * v_bs),
        "local_lid_scalar_components": int(len(lid_dofs) * v_bs),
        "preprocess_elapsed_seconds": float(elapsed),
    }

    summary = {
        "mpi_size": comm.size,
        "mesh": [args.nx, args.ny],
        "domain": [1.0, 1.0],
        "cell_type": "triangle",
        "geometric_dim": int(gdim),
        "velocity_element": "vector Lagrange P2",
        "pressure_element": "scalar Lagrange P1",
        "velocity_block_size": int(v_bs),
        "pressure_block_size": int(q_bs),
        "global_cells": int(cell_map.size_global),
        "global_vertices": int(vertex_map.size_global),
        "velocity_dofs": int(V.dofmap.index_map.size_global * v_bs),
        "pressure_dofs": int(Q.dofmap.index_map.size_global * q_bs),
        "total_dofs": int(V.dofmap.index_map.size_global * v_bs + Q.dofmap.index_map.size_global * q_bs),
        "sum_noslip_facets": int(comm.allreduce(len(noslip_facets), op=MPI.SUM)),
        "sum_lid_facets": int(comm.allreduce(len(lid_facets), op=MPI.SUM)),
        "sum_noslip_block_dofs": int(comm.allreduce(len(noslip_dofs), op=MPI.SUM)),
        "sum_lid_block_dofs": int(comm.allreduce(len(lid_dofs), op=MPI.SUM)),
        "sum_noslip_scalar_components": int(comm.allreduce(len(noslip_dofs) * v_bs, op=MPI.SUM)),
        "sum_lid_scalar_components": int(comm.allreduce(len(lid_dofs) * v_bs, op=MPI.SUM)),
        "expected_cells": int(expected_cells),
        "expected_vertices": int(expected_vertices),
        "expected_velocity_scalar_dofs": int(expected_velocity_scalar_dofs),
        "expected_pressure_dofs": int(expected_pressure_dofs),
        "expected_noslip_boundary_vertices": int(expected_noslip_vertices),
        "expected_lid_boundary_vertices": int(expected_lid_vertices),
        "expected_noslip_p2_block_dofs": int(expected_noslip_p2_block_dofs),
        "expected_lid_p2_block_dofs": int(expected_lid_p2_block_dofs),
        "expected_noslip_p2_scalar_components": int(expected_noslip_p2_block_dofs * v_bs),
        "expected_lid_p2_scalar_components": int(expected_lid_p2_block_dofs * v_bs),
        "forms_created": True,
        "nest_blocks": "2x2",
        "preconditioner_blocks": "velocity stiffness + pressure mass",
        "noslip_value": [0.0, 0.0],
        "lid_value": [1.0, 0.0],
        "body_force": [0.0, 0.0],
        "preprocess_elapsed_seconds_max_rank": float(elapsed_max),
    }

    ranks = comm.gather(rank_data, root=0)
    if comm.rank == 0:
        print("PREPROCESS " + json.dumps({"summary": summary, "ranks": ranks}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
