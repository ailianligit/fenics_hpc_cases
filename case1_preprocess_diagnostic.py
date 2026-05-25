#!/usr/bin/env python3
"""Preprocessing diagnostic for the 3D elasticity AMG case."""

from __future__ import annotations

import argparse
import json

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import dirichletbc, form, functionspace, locate_dofs_topological
from dolfinx.mesh import CellType, GhostMode, create_box, locate_entities_boundary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preprocessing diagnostics for case 1.")
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--length", type=float, default=2.0)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD

    comm.Barrier()
    t0 = MPI.Wtime()

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
    fixed_dofs = locate_dofs_topological(V, entity_dim=fdim, entities=fixed_facets)
    bc = dirichletbc(np.zeros(3, dtype=PETSc.ScalarType), fixed_dofs, V=V)

    # Keep variables live so the diagnostic really exercises the full preprocessing path.
    _ = (a, L, bc)

    comm.Barrier()
    elapsed = MPI.Wtime() - t0
    elapsed_max = comm.allreduce(elapsed, op=MPI.MAX)

    cell_map = msh.topology.index_map(msh.topology.dim)
    vertex_map = msh.topology.index_map(0)
    block_size = V.dofmap.index_map_bs

    expected_fixed_nodes = (args.ny + 1) * (args.nz + 1) + (args.nx + 1) * (args.nz + 1) - (args.nz + 1)
    expected_fixed_components = expected_fixed_nodes * block_size

    rank_data = {
        "rank": comm.rank,
        "local_cells": int(cell_map.size_local),
        "local_vertices": int(vertex_map.size_local),
        "local_dofs": int(V.dofmap.index_map.size_local * block_size),
        "local_fixed_facets": int(len(fixed_facets)),
        "local_fixed_block_dofs": int(len(fixed_dofs)),
        "local_fixed_scalar_components": int(len(fixed_dofs) * block_size),
        "preprocess_elapsed_seconds": float(elapsed),
    }

    summary = {
        "mpi_size": comm.size,
        "mesh": [args.nx, args.ny, args.nz],
        "domain": [args.length, args.width, args.height],
        "cell_type": "tetrahedron",
        "ghost_mode": "shared_facet",
        "element": "vector Lagrange P1",
        "geometric_dim": int(msh.geometry.dim),
        "block_size": int(block_size),
        "global_cells": int(cell_map.size_global),
        "global_vertices": int(vertex_map.size_global),
        "global_dofs": int(V.dofmap.index_map.size_global * block_size),
        "sum_fixed_facets": int(comm.allreduce(len(fixed_facets), op=MPI.SUM)),
        "sum_fixed_block_dofs": int(comm.allreduce(len(fixed_dofs), op=MPI.SUM)),
        "sum_fixed_scalar_components": int(comm.allreduce(len(fixed_dofs) * block_size, op=MPI.SUM)),
        "expected_unique_fixed_nodes": int(expected_fixed_nodes),
        "expected_unique_fixed_scalar_components": int(expected_fixed_components),
        "E": E,
        "nu": nu,
        "mu": mu,
        "lambda": lambda_,
        "rho": rho,
        "omega": omega,
        "body_force_coefficient": rho * omega**2,
        "bc_value": [0.0, 0.0, 0.0],
        "forms_created": True,
        "preprocess_elapsed_seconds_max_rank": float(elapsed_max),
    }

    ranks = comm.gather(rank_data, root=0)
    if comm.rank == 0:
        print("PREPROCESS " + json.dumps({"summary": summary, "ranks": ranks}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
