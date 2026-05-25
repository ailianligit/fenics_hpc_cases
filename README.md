# FEniCSx 高性能案例

本仓库包含 2 个基于 FEniCSx/DOLFINx 的高性能有限元案例，覆盖固体力学和流体力学两类典型问题。

## 1. 案例概览

| 案例 | 物理问题 | 离散与求解 | 高性能重点 | 主要输出 |
|---|---|---|---|---|
| 基于 AMG 的三维线弹性并行求解 | 三维弹性体位移和 Von Mises 应力 | $[P_1]^3$ 向量元，`CG + GAMG` | MPI 分布式网格、PETSc GAMG、刚体模态近零空间 | `displacement.xdmf`、`von_mises_stress.xdmf`、性能指标 |
| 基于 FieldSplit 的顶盖驱动 Stokes 流求解 | 二维不可压 Stokes 方腔流 | Taylor-Hood $[P_2]^2 \times P_1$，`MINRES + FieldSplit` | MatNest/VecNest、速度-压力块预条件、压力空空间 | `velocity.xdmf`、`pressure.xdmf`、性能指标 |

## 2. 目录结构

```text
fenics_hpc_cases/
├── README.md
├── case1_elasticity_amg.py
├── case1_preprocess_diagnostic.py
├── case2_stokes_fieldsplit.py
├── case2_preprocess_diagnostic.py
├── export_field_figures.py
└── results/
```

## 3. 环境准备

Ubuntu 开发机可按 FEniCSx PPA 安装核心依赖：

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common git wget
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt update
sudo apt install -y fenicsx
```

`fenicsx` 会安装 DOLFINx、UFL、Basix、FFCx、PETSc、petsc4py、mpi4py、NumPy 和 MPI 运行时等核心依赖。为了绘制性能图、导出 PNG 或交互查看 XDMF，建议额外安装：

```bash
sudo apt install -y python3-matplotlib paraview
```

验证安装：

```bash
python3 -c "import dolfinx, petsc4py, mpi4py, numpy, matplotlib; print(dolfinx.__version__)"
mpirun -n 2 python3 -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())"
```

## 4. 快速正确性验证

以下命令默认在仓库目录 `fenics_hpc_cases/` 下执行。

系统 Python 运行：

```bash
python3 case1_elasticity_amg.py --small
mpirun -n 2 python3 case1_elasticity_amg.py --small \
  --output-dir results/case1_small_np2

python3 case2_stokes_fieldsplit.py --small \
  mpirun -n 2 python3 case2_stokes_fieldsplit.py --small \
  --output-dir results/case2_small_np2
```

成功标志：终端输出 `METRICS {...}`，且 `converged_reason` 为正数。
