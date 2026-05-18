# 案例 1：三维线弹性 AMG 并行求解

## 1. 问题分析

本案例使用 FEniCSx/DOLFINx 求解三维各向同性小变形线弹性问题。计算域为长方体
\(\Omega=[0,L]\times[0,W]\times[0,H]\)，材料受到随空间坐标变化的体力载荷
\(f=\rho\omega^2(x,y,0)^T\)。边界 \(x=0\) 与 \(y=W\) 固定，其余边界自由。

该问题代表固体力学中常见的三维位移场求解。离散后得到大规模稀疏对称正定线性系统，适合使用 PETSc 的 Krylov 迭代和代数多重网格预条件器。案例重点展示：

- MPI 分布式三维网格与有限元装配；
- PETSc CG + GAMG 求解器；
- 三维弹性刚体模态近零空间；
- 位移场和 Von Mises 应力并行输出。

脚本位置：`case1_elasticity_amg.py`

## 2. 求解模型

强形式为：

\[
-\nabla\cdot\sigma(u)=f \quad \text{in } \Omega
\]

\[
u=0 \quad \text{on } \Gamma_D,\qquad \sigma(u)n=0 \quad \text{on } \Gamma_N
\]

线弹性本构关系为：

\[
\sigma(u)=2\mu\epsilon(u)+\lambda\operatorname{tr}(\epsilon(u))I
\]

\[
\epsilon(u)=\frac{1}{2}\left(\nabla u+\nabla u^T\right)
\]

其中

\[
\mu=\frac{E}{2(1+\nu)}, \qquad
\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}
\]

弱形式为：求 \(u\in V\)，满足 Dirichlet 边界条件，且对任意测试函数 \(v\in V_0\)：

\[
\int_\Omega \sigma(u):\epsilon(v)\,dx=\int_\Omega f\cdot v\,dx
\]

脚本中使用一阶连续 Lagrange 向量元：

\[
V_h = [P_1]^3
\]

后处理阶段计算偏应力和 Von Mises 应力：

\[
\sigma_\mathrm{vm}=\sqrt{\frac{3}{2}s:s},\qquad
s=\sigma-\frac{1}{3}\operatorname{tr}(\sigma)I
\]

## 3. 算法设计

1. 使用 `create_box` 创建可 MPI 划分的三维四面体网格。
2. 使用 UFL 直接描述应变、应力、双线性形式和线性形式。
3. 定位固定边界面，构造齐次 Dirichlet 边界条件。
4. 并行装配矩阵和右端项，并对 Dirichlet 条件执行 lifting 与 ghost update。
5. 构造三维刚体模态近零空间，包括 3 个平移模态和 3 个旋转模态。
6. 将近零空间附加到 PETSc 矩阵，使用 CG + GAMG 求解。
7. 将位移和 Von Mises 应力写出为 XDMF 文件。
8. 输出 JSON 指标，便于串并行一致性检查。

## 4. 编程实现

主要命令行参数：

```bash
conda run -n fenicsx-env python case1_elasticity_amg.py --help
```

常用运行方式：

```bash
# 小规模串行验证
conda run -n fenicsx-env python case1_elasticity_amg.py --small

# 2 进程 MPI 验证
conda run -n fenicsx-env mpirun -n 2 python case1_elasticity_amg.py --small \
  --output-dir results/case1_elasticity_amg_mpi2

# 增大网格规模
conda run -n fenicsx-env mpirun -n 4 python case1_elasticity_amg.py \
  --nx 24 --ny 12 --nz 12 --monitor
```

关键实现点：

- `build_nullspace(V)` 构造弹性算子的近零空间；
- `assemble_matrix` 和 `assemble_vector` 完成并行装配；
- `PETSc.KSP` 设置为 `cg`，预条件器设置为 `gamg`；
- `Expression` 将 Von Mises 应力插值到 DG0 空间；
- `metrics.json` 记录自由度数、迭代次数、范数和运行时间。

## 5. 应用组合

本案例组合了 FEniCSx 的多个核心组件：

- DOLFINx：网格、函数空间、边界条件、并行装配和 XDMF 输出；
- UFL：以接近数学公式的形式定义弱形式；
- Basix/FFCx：有限元基函数与变分形式编译；
- PETSc：稀疏矩阵、KSP 迭代器、GAMG 预条件器；
- MPI：网格分区、向量通信和并行求解；
- ParaView：读取 `displacement.xdmf` 与 `von_mises_stress.xdmf` 可视化。

## 6. 运行验证

环境检查：

```bash
conda run -n fenicsx-env python -c "import dolfinx, petsc4py, mpi4py; print(dolfinx.__version__)"
```

本机环境返回 DOLFINx `0.10.0`。小规模网格 `--small` 对应网格参数 `[4, 4, 4]`，全局单元数 384，全局自由度数 375。

| 进程数 | 迭代次数 | 收敛原因 | 位移向量范数 | 运行时间 s |
|---:|---:|---:|---:|---:|
| 1 | 12 | 2 | 0.007511165612501762 | 0.4431 |
| 2 | 13 | 2 | 0.0075111656151875305 | 0.0216 |
| 4 | 12 | 2 | 0.007511165612052932 | 0.0186 |

`converged_reason=2` 是 PETSc 正收敛原因，表示求解器正常收敛。串行、2 进程、4 进程得到的位移范数在 \(10^{-11}\) 量级内一致，说明 MPI 分布式装配和求解路径正确。

输出文件：

- `results/case1_elasticity_amg/displacement.xdmf`
- `results/case1_elasticity_amg/von_mises_stress.xdmf`
- `results/case1_elasticity_amg/metrics.json`

指标字段说明：

- `global_dofs`：全局有限元自由度数；
- `iterations`：KSP 迭代次数；
- `solution_norm`：位移系数向量范数，用于串并行一致性检查；
- `elapsed_seconds`：核心装配、求解和后处理耗时；
- `pc_type=gamg`：使用 PETSc 代数多重网格预条件。

## 参考资料

- DOLFINx Elasticity AMG demo: https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/demos/demo_elasticity.html
- FEniCSx tutorial overview: https://jsdokken.com/dolfinx-tutorial/fem.html
