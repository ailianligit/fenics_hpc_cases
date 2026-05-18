#!/usr/bin/env python3
"""Summarize strong-scaling metrics and export performance figures."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    label: str
    dirs: dict[int, str]
    dof_key: str
    norm_keys: tuple[str, ...]
    figure_name: str


CASES = (
    CaseConfig(
        case_id="case1",
        label="3D Linear Elasticity AMG",
        dirs={
            1: "case1_strong_1",
            2: "case1_strong_2",
            4: "case1_strong_4",
        },
        dof_key="global_dofs",
        norm_keys=("solution_norm",),
        figure_name="case1_strong_performance.png",
    ),
    CaseConfig(
        case_id="case2",
        label="Taylor-Hood Stokes FieldSplit",
        dirs={
            1: "case2_strong_1",
            2: "case2_strong_2",
            4: "case2_strong_4",
        },
        dof_key="total_dofs",
        norm_keys=("velocity_norm", "pressure_norm"),
        figure_name="case2_strong_performance.png",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="fenics_hpc_cases/results",
        help="Directory containing case*_strong_* result directories.",
    )
    parser.add_argument(
        "--figures-dir",
        default="fenics_hpc_cases/docs/figures",
        help="Directory where performance figures are written.",
    )
    parser.add_argument(
        "--csv",
        default="fenics_hpc_cases/results/strong_scaling_summary.csv",
        help="Path of the generated CSV summary.",
    )
    return parser.parse_args()


def load_metrics(results_dir: Path, cfg: CaseConfig) -> list[dict]:
    rows = []
    for mpi_size, dirname in cfg.dirs.items():
        metrics_path = results_dir / dirname / "metrics.json"
        with metrics_path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        if metrics["mpi_size"] != mpi_size:
            raise ValueError(f"{metrics_path}: expected mpi_size={mpi_size}, got {metrics['mpi_size']}")
        if metrics["converged_reason"] <= 0:
            raise ValueError(f"{metrics_path}: KSP did not converge")

        row = {
            "case_id": cfg.case_id,
            "case_label": cfg.label,
            "mpi_size": mpi_size,
            "mesh": " x ".join(str(v) for v in metrics["mesh"]),
            "global_cells": metrics["global_cells"],
            "dofs": metrics[cfg.dof_key],
            "iterations": metrics["iterations"],
            "elapsed_seconds": metrics["elapsed_seconds"],
            "ksp_type": metrics["ksp_type"],
            "pc_type": metrics["pc_type"],
            "converged_reason": metrics["converged_reason"],
            "write_xdmf": metrics["write_xdmf"],
        }
        for key in cfg.norm_keys:
            row[key] = metrics[key]
        if "divergence_l2" in metrics:
            row["divergence_l2"] = metrics["divergence_l2"]
        if "nullspace_test" in metrics:
            row["nullspace_test"] = metrics["nullspace_test"]
        rows.append(row)

    rows.sort(key=lambda item: item["mpi_size"])
    baseline = rows[0]
    baseline_time = baseline["elapsed_seconds"]
    for row in rows:
        speedup = baseline_time / row["elapsed_seconds"]
        row["speedup"] = speedup
        row["parallel_efficiency"] = speedup / row["mpi_size"]
        for key in cfg.norm_keys:
            base_norm = baseline[key]
            row[f"{key}_relative_error"] = abs(row[key] - base_norm) / max(abs(base_norm), 1.0e-30)
    return rows


def write_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "case_label",
        "mesh",
        "mpi_size",
        "global_cells",
        "dofs",
        "iterations",
        "elapsed_seconds",
        "speedup",
        "parallel_efficiency",
        "solution_norm",
        "solution_norm_relative_error",
        "velocity_norm",
        "velocity_norm_relative_error",
        "pressure_norm",
        "pressure_norm_relative_error",
        "divergence_l2",
        "ksp_type",
        "pc_type",
        "converged_reason",
        "nullspace_test",
        "write_xdmf",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_case(rows: list[dict], cfg: CaseConfig, figures_dir: Path) -> Path:
    mpi = [row["mpi_size"] for row in rows]
    elapsed = [row["elapsed_seconds"] for row in rows]
    speedup = [row["speedup"] for row in rows]
    efficiency = [100.0 * row["parallel_efficiency"] for row in rows]
    iterations = [row["iterations"] for row in rows]

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.8), dpi=180)
    fig.suptitle(cfg.label, fontsize=13)

    ax = axes[0, 0]
    ax.plot(mpi, elapsed, marker="o", color="#1f77b4")
    ax.set_title("Runtime")
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("seconds")
    ax.grid(True, alpha=0.28)

    ax = axes[0, 1]
    ax.plot(mpi, speedup, marker="o", color="#2ca02c", label="measured")
    ax.plot(mpi, mpi, linestyle="--", color="#7f7f7f", label="ideal")
    ax.set_title("Speedup")
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("T1 / TN")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(mpi, efficiency, marker="o", color="#ff7f0e")
    ax.axhline(100.0, linestyle="--", color="#7f7f7f", linewidth=1)
    ax.set_title("Parallel efficiency")
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("%")
    ax.grid(True, alpha=0.28)

    ax = axes[1, 1]
    ax.plot(mpi, iterations, marker="o", color="#9467bd")
    ax.set_title("KSP iterations")
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("iterations")
    ax.grid(True, alpha=0.28)

    for ax in axes.ravel():
        ax.set_xticks(mpi)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = figures_dir / cfg.figure_name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    all_rows: list[dict] = []
    figure_paths = []

    for cfg in CASES:
        rows = load_metrics(results_dir, cfg)
        all_rows.extend(rows)
        figure_paths.append(plot_case(rows, cfg, figures_dir))

    write_csv(all_rows, Path(args.csv))
    print(Path(args.csv))
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()
