# 案例 2：Taylor-Hood Stokes 块预条件求解

## 1. 问题分析

本案例使用 FEniCSx/DOLFINx 求解二维 lid-driven cavity Stokes 流。计算域为单位正方形
\(\Omega=[0,1]\times[0,1]\)。顶部边界以速度 \((1,0)^T\) 水平移动，其余边界满足无滑移条件。目标是求解速度场 \(u\) 和压力场 \(p\)。

Stokes 方程是不可压 Navier-Stokes 方程的线性化模型，也是流体力学、多孔介质、多物理耦合中的基础模块。离散后得到速度-压力鞍点系统，单纯直接求解不利于大规模并行扩展。本案例重点展示：

- Taylor-Hood 混合有限元；
- PETSc MatNest/VecNest 块矩阵结构；
- MINRES + fieldsplit 块预条件；
- 速度块 GAMG、压力块 Jacobi；
- 压力零均值空空间处理。

脚本位置：`case2_stokes_fieldsplit.py`

## 2. 求解模型

本案例采用 DOLFINx 官方示例中的对称号约定：

\[
-\nabla\cdot(\nabla u + pI)=f \quad \text{in } \Omega
\]

\[
\nabla\cdot u=0 \quad \text{in } \Omega
\]

边界条件为：

\[
u=(1,0)^T \quad \text{on } y=1
\]

\[
u=(0,0)^T \quad \text{on } x=0,\ x=1,\ y=0
\]

压力只确定到一个常数，因此需要给线性系统附加压力常数空空间。

弱形式为：求 \((u,p)\in V\times Q\)，对任意 \((v,q)\in V\times Q\)：

\[
\int_\Omega \nabla u:\nabla v\,dx
+ \int_\Omega p\nabla\cdot v\,dx
+ \int_\Omega q\nabla\cdot u\,dx
= \int_\Omega f\cdot v\,dx
\]

离散空间采用 Taylor-Hood 元：

\[
V_h=[P_2]^2,\qquad Q_h=P_1
\]

该组合满足常用的速度-压力稳定性要求，适合不可压流问题。

## 3. 算法设计

1. 使用 `create_rectangle` 创建二维三角形网格。
2. 分别构建速度空间 \(V_h=[P_2]^2\) 和压力空间 \(Q_h=P_1\)。
3. 定义顶部驱动边界和其他无滑移边界。
4. 用 UFL 定义块形式 `a_ufl` 和右端 `L_ufl`。
5. 构造块对角预条件器：速度块使用刚度矩阵，压力块使用质量矩阵。
6. 使用 `LinearProblem(kind="nest")` 创建 PETSc MatNest/VecNest 系统。
7. 附加压力常数 NullSpace，消除压力不唯一性。
8. 使用 MINRES + fieldsplit 求解，速度块用 GAMG，压力块用 Jacobi。
9. 将速度插值到一阶向量空间后输出 XDMF，压力直接输出 XDMF。

## 4. 编程实现

主要命令行参数：

```bash
conda run -n fenicsx-env python case2_stokes_fieldsplit.py --help
```

常用运行方式：

```bash
# 小规模串行验证
conda run -n fenicsx-env python case2_stokes_fieldsplit.py --small

# 2 进程 MPI 验证
conda run -n fenicsx-env mpirun -n 2 python case2_stokes_fieldsplit.py --small \
  --output-dir results/case2_stokes_fieldsplit_mpi2

# 增大网格规模
conda run -n fenicsx-env mpirun -n 4 python case2_stokes_fieldsplit.py \
  --nx 64 --ny 64 --monitor
```

关键实现点：

- `element("Lagrange", ..., degree=2, shape=(gdim,))` 定义二次速度元；
- `element("Lagrange", ..., degree=1)` 定义一次压力元；
- `LinearProblem(..., kind="nest")` 保持块结构；
- `pc_type=fieldsplit` 使用 PETSc 块预条件；
- `create_vector(extract_function_spaces(L), "nest")` 构造压力空空间向量；
- `metrics.json` 记录速度/压力自由度、迭代次数、范数和散度指标。

## 5. 应用组合

本案例体现了 FEniCSx 在多物理场快速原型中的典型组合方式：

- DOLFINx：网格、边界定位、混合空间、XDMF 输出；
- UFL：速度-压力耦合弱形式；
- Basix：Taylor-Hood 元定义；
- PETSc：MatNest、VecNest、MINRES、fieldsplit；
- MPI：并行网格、并行装配和并行求解；
- ParaView：读取 `velocity.xdmf` 和 `pressure.xdmf` 查看流场。

## 6. 运行验证

环境检查：

```bash
conda run -n fenicsx-env python -c "import dolfinx, petsc4py, mpi4py; print(dolfinx.__version__)"
```

本机环境返回 DOLFINx `0.10.0`。小规模网格 `--small` 对应网格参数 `[8, 8]`，全局单元数 128，总自由度数 659，其中速度自由度 578，压力自由度 81。

| 进程数 | 迭代次数 | 收敛原因 | 速度范数 | 压力范数 | 散度 L2 指标 | 运行时间 s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 79 | 2 | 5.210424816467188 | 65.22050552805148 | 0.3724618711102948 | 0.2463 |
| 2 | 79 | 2 | 5.210424816522794 | 65.22050551904131 | 0.37246187106874534 | 0.1352 |
| 4 | 80 | 2 | 5.210424816612111 | 65.22050552412779 | 0.3724618709772056 | 0.0371 |

`converged_reason=2` 是 PETSc 正收敛原因，`nullspace_test=true` 表示压力常数空空间通过矩阵一致性测试。串行、2 进程、4 进程速度范数和压力范数保持一致，说明 MatNest/fieldsplit 并行求解路径正确。散度 L2 指标用于串并行一致性比较；Taylor-Hood 方法满足弱不可压约束，不要求逐点散度为零。

输出文件：

- `results/case2_stokes_fieldsplit/velocity.xdmf`
- `results/case2_stokes_fieldsplit/pressure.xdmf`
- `results/case2_stokes_fieldsplit/metrics.json`

指标字段说明：

- `velocity_dofs`：速度空间全局自由度；
- `pressure_dofs`：压力空间全局自由度；
- `iterations`：MINRES 迭代次数；
- `nullspace_test`：压力空空间测试结果；
- `velocity_norm`、`pressure_norm`：用于串并行一致性检查；
- `pc_type=fieldsplit`：使用 PETSc 块预条件器。

## 参考资料

- DOLFINx Stokes Taylor-Hood demo: https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/demos/demo_stokes.html
- FEniCSx finite element solver gallery: https://jsdokken.com/dolfinx-tutorial/chapter2/intro.html
