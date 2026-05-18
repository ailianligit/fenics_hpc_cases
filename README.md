# FEniCSx 高性能案例

本目录包含 2 个 FEniCSx/DOLFINx 高性能科学计算案例，均使用中文文档和 Python 脚本交付。

## 目录结构

```text
fenics_hpc_cases/
├── case1_elasticity_amg.py
├── case2_stokes_fieldsplit.py
├── export_field_figures.py
├── docs/
│   ├── FEniCS高性能教程.md
│   ├── case1_elasticity_amg.md
│   ├── case2_stokes_fieldsplit.md
│   └── figures/
│       ├── case1_displacement_warp.png
│       ├── case1_von_mises.png
│       ├── case2_pressure.png
│       ├── case2_velocity.png
│       └── performance_summary.png
└── results/
    ├── case1_elasticity_amg/
    └── case2_stokes_fieldsplit/
```

## 环境

使用已有 Conda 环境：

```bash
conda run -n fenicsx-env python -c "import dolfinx, petsc4py, mpi4py; print(dolfinx.__version__)"
```

本机验证版本为 DOLFINx `0.10.0`。

## 快速运行

```bash
# 案例 1：三维线弹性 AMG
conda run -n fenicsx-env python fenics_hpc_cases/case1_elasticity_amg.py --small
conda run -n fenicsx-env mpirun -n 2 python fenics_hpc_cases/case1_elasticity_amg.py --small \
  --output-dir fenics_hpc_cases/results/case1_elasticity_amg_mpi2

# 案例 2：Taylor-Hood Stokes fieldsplit
conda run -n fenicsx-env python fenics_hpc_cases/case2_stokes_fieldsplit.py --small
conda run -n fenicsx-env mpirun -n 2 python fenics_hpc_cases/case2_stokes_fieldsplit.py --small \
  --output-dir fenics_hpc_cases/results/case2_stokes_fieldsplit_mpi2
```

## 文档

- `docs/FEniCS高性能教程.md`：整合版完整教程，包含两个案例、实验验证、性能图和可视化说明。
- `docs/case1_elasticity_amg.md`：问题分析、求解模型、算法设计、编程实现、应用组合、运行验证。
- `docs/case2_stokes_fieldsplit.md`：问题分析、求解模型、算法设计、编程实现、应用组合、运行验证。

输出文件为 XDMF，可用 ParaView 查看。每次运行还会生成 `metrics.json`，用于记录自由度数、迭代次数、范数和运行时间。

## 场变量插图导出

```bash
conda run -n fenicsx-env python fenics_hpc_cases/export_field_figures.py
```

该脚本会重新计算可视化规模的场变量，并导出位移、Von Mises 应力、速度和压力 PNG 到 `docs/figures/`。
