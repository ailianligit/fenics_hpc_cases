# FEniCSx 高性能案例

本目录包含 2 个 FEniCSx/DOLFINx 高性能科学计算案例，均使用 Python 脚本交付。

## 目录结构

```text
fenics_hpc_cases/
├── case1_elasticity_amg.md
├── case1_elasticity_amg.py
├── case2_stokes_fieldsplit.py
├── case2_stokes_fieldsplit.md
├── export_field_figures.py
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
conda run -n fenicsx-env python case1_elasticity_amg.py --small
conda run -n fenicsx-env mpirun -n 2 python case1_elasticity_amg.py --small \
  --output-dir results/case1_elasticity_amg_mpi2

# 案例 2：Taylor-Hood Stokes fieldsplit
conda run -n fenicsx-env python case2_stokes_fieldsplit.py --small
conda run -n fenicsx-env mpirun -n 2 python case2_stokes_fieldsplit.py --small \
  --output-dir results/case2_stokes_fieldsplit_mpi2
```

## 文档

- `case1_elasticity_amg.md`：问题分析、求解模型、算法设计、编程实现、应用组合、运行验证。
- `case2_stokes_fieldsplit.md`：问题分析、求解模型、算法设计、编程实现、应用组合、运行验证。

输出文件为 XDMF，可用 ParaView 查看。每次运行还会生成 `metrics.json`，用于记录自由度数、迭代次数、范数和运行时间。

## 场变量插图导出

```bash
conda run -n fenicsx-env python export_field_figures.py
```

该脚本会重新计算可视化规模的场变量，并导出位移、Von Mises 应力、速度和压力 PNG 到 `docs/figures/`。
