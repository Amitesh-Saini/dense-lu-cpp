# Dense LU Decomposition in C++17: Accuracy, Stability, and Performance Analysis

A from-scratch dense direct solver implementing Gaussian elimination with partial pivoting in C++17. The implementation computes the factorization PA = LU, solves linear systems via forward and backward substitution, and is verified against a suite of structured matrix families. Performance is benchmarked against Eigen's `PartialPivLU` across matrix sizes up to n = 2048. The project studies numerical correctness, backward stability, conditioning effects, and computational scaling — grounded in the error analysis of Dahlquist and Björck [1].

---

## Table of Contents

1. [Mathematical Background](#1-mathematical-background)
2. [Implementation](#2-implementation)
3. [Verification](#3-verification)
4. [Test Matrix Families](#4-test-matrix-families)
5. [Benchmark Methodology](#5-benchmark-methodology)
6. [Results](#6-results)
7. [Building and Running](#7-building-and-running)
8. [Limitations](#8-limitations)
9. [Future Work](#9-future-work)
10. [Issues and Feedback](#10-issues-and-feedback)
11. [References](#11-references)

---

## 1. Mathematical Background

### 1.1 LU Factorization with Partial Pivoting

The solver decomposes a nonsingular matrix $A \in \mathbb{R}^{n \times n}$ as

$$PA = LU,$$

where $P$ is a permutation matrix (stored as a pivot index vector), $L$ is unit lower triangular ($L_{ii} = 1$), and $U$ is upper triangular. The existence and uniqueness of the unpivoted factorization follows from the leading-minor condition:

> **Theorem 5.3.1 [1].** Given $A \in \mathbb{R}^{n \times n}$, denote by $A_k$ the $k \times k$ leading principal submatrix. If $\det(A_k) \neq 0$ for $k = 1, \ldots, n-1$, then there exists a unique unit lower triangular $L$ and unique upper triangular $U$ such that $LU = A$.

Partial pivoting relaxes this condition by introducing row interchanges. At elimination step $k$, the pivot row $r$ is chosen as

$$|a_{rk}| = \max_{k \leq i \leq n} |a_{ik}|,$$

and rows $k$ and $r$ are swapped before the elimination multipliers are computed. This guarantees that all multipliers $|m_{ik}| \leq 1$, bounding growth during elimination.

**Why partial rather than complete pivoting?** Complete pivoting searches the entire remaining $(n-k) \times (n-k)$ submatrix and pivots on both rows and columns, providing stronger growth control at the cost of $O(n^2)$ comparisons per step versus $O(n)$ for partial pivoting. In practice, partial pivoting is used universally in production dense solvers (LAPACK's `dgetrf`, Eigen's `PartialPivLU`) because the theoretical worst-case growth advantage of complete pivoting is rarely realized and the extra search cost is not justified.

### 1.2 Operation Count

At elimination step $k$, the algorithm performs $n-k$ divisions (computing multipliers) and $(n-k)^2$ fused multiply-subtract operations (updating the trailing submatrix). Summing over $k = 1, \ldots, n-1$:

$$\text{Factorization} = \frac{2}{3}n^3 + O(n^2) \text{ flops},$$

where the leading $\frac{2}{3}n^3$ comprises $\frac{1}{3}n^3$ multiplications and $\frac{1}{3}n^3$ subtractions. Once the factorization is known, each triangular solve (forward substitution $Ly = Pb$, backward substitution $Ux = y$) costs $n^2$ flops, giving a total solve cost of $2n^2$ flops. This cubic-vs-quadratic asymmetry is the main motivation for factorizing once and reusing the factors across multiple right-hand sides.

### 1.3 Growth Factor and Backward Stability

The key quantity controlling backward stability under floating-point arithmetic is the **growth factor**

$$g_n = \frac{\displaystyle\max_{i,j,k} |a_{ij}^{(k)}|}{\displaystyle\max_{i,j} |a_{ij}^{(0)}|},$$

where $a_{ij}^{(k)}$ denotes the $(i,j)$ entry of the matrix after $k$ steps of elimination. The growth factor measures how much matrix entries grow relative to the original matrix during Gaussian elimination. The purpose of pivoting is precisely to limit $g_n$.

The backward error analysis of Dahlquist and Björck gives the following bounds:

> **Theorem 5.5.1 [1].** If $L$ and $U$ are the computed triangular factors of $A$ obtained by Gaussian elimination with partial or complete pivoting, using floating-point arithmetic with rounding unit $u$, then there exists a matrix $E$ satisfying
>
> $$\|E\|_\infty \leq k_1(n) \cdot u \cdot \|A\|_\infty, \qquad k_1(n) = n^2 g_n,$$
>
> such that $LU = A + E$.

> **Theorem 5.5.2 [1].** If $x'$ is the computed solution of $Ax = b$ obtained by forward and backward substitution using the factors from Theorem 5.5.1, then there exists $\Delta A$ depending on $A$ and $b$ satisfying
>
> $$\|\Delta A\|_\infty \leq k_2(n) \cdot u \cdot \|A\|_\infty, \qquad k_2(n) = (n^3 + 3n^2) g_n,$$
>
> such that $(A + \Delta A)x' = b$.

These theorems are the theoretical justification for the observed residual behavior in Section 6: a small solve residual means $x'$ solves a nearby perturbed system, and Theorem 5.5.2 quantifies how nearby in terms of $g_n$ and $u$.

**Growth factor bounds.** The theoretical worst-case bound for partial pivoting is $g_n \leq 2^{n-1}$, achieved by Wilkinson's adversarial construction. For complete pivoting the bound is $g_n \leq 1.8 \cdot n^{0.25 \ln n}$, substantially smaller at large $n$. However, empirical experience across decades of practical use shows that $g_n$ rarely exceeds 8 even under partial pivoting [1]. This empirical observation — not a proven bound — is why partial pivoting is considered stable in practice despite the exponential worst-case. In this project, the successful test cases show near-machine-precision residuals, suggesting that element growth was not the dominant source of error for the tested matrix families. A future benchmark could explicitly measure $g_n$ to confirm this behavior.

### 1.4 Norms

All residuals and errors in this project use the $\infty$-norm:

$$\|x\|_\infty = \max_i |x_i|, \qquad \|A\|_\infty = \max_i \sum_j |a_{ij}|.$$

The $\infty$-norm is chosen because it matches the norm used in the backward error bounds of Theorems 5.5.1 and 5.5.2, giving a direct connection between the measured residuals and the theoretical guarantees. Matrix norms satisfy submultiplicativity: $\|AB\|_\infty \leq \|A\|_\infty \|B\|_\infty$, ensuring that residual bounds compose correctly across the factorization and solve steps.

### 1.5 Conditioning and Forward Error

The condition number in the $\infty$-norm is

$$\kappa_\infty(A) = \|A\|_\infty \|A^{-1}\|_\infty.$$

Note that $\kappa_\infty(A)$ is not generally known without computing $A^{-1}$ explicitly, which costs $O(n^3)$. For a perturbed right-hand side $b + \Delta b$, the forward error satisfies

$$\frac{\|\Delta x\|_\infty}{\|x\|_\infty} \leq \kappa_\infty(A) \frac{\|\Delta b\|_\infty}{\|b\|_\infty},$$

with equality only for special right-hand sides. Combining with the backward error of Theorem 5.5.2:

$$\frac{\|x' - x\|_\infty}{\|x\|_\infty} \lesssim \kappa_\infty(A) \cdot k_2(n) \cdot u \cdot \frac{\|A\|_\infty \|x'\|_\infty}{\|b\|_\infty}.$$

A small solve residual is necessary but not sufficient for a small forward error. If $\kappa_\infty(A)$ is large, the amplification factor overwhelms the small backward error and the forward error can be $O(1)$ or larger. This is the precise mechanism behind the Hilbert matrix experiments in Section 6.2.

### 1.6 Residuals in Practice

Dahlquist and Björck note [1] that in many practical applications what matters is not the forward error in $x'$ but whether $r = b - Ax'$ is small. Theorem 5.5.2 guarantees

$$\|b - Ax'\|_\infty \leq k_2(n) \cdot u \cdot \|A\|_\infty \|x'\|_\infty,$$

so Gaussian elimination with partial pivoting guarantees a small solve residual regardless of conditioning, provided $g_n$ is not pathologically large. The solve residuals in this project remain near $u \cdot \|A\|_\infty \|x'\|_\infty$ for all tested matrix families — including the Hilbert cases where CLU does not abort — confirming that the backward error is controlled. It is the forward error that reflects ill-conditioning.

---

## 2. Implementation

### 2.1 Storage Layout

Matrices are stored as flat `std::vector<double>` arrays using row-major indexing:

$$A_{ij} = \texttt{data}[i \cdot n + j].$$

This gives a single contiguous memory allocation. A `std::vector<std::vector<double>>` representation would allocate each row on the heap separately, fragmenting the matrix across many allocations and degrading cache locality. Contiguous storage reduces TLB pressure, simplifies cache-line prediction, and matches the layout expected by BLAS and LAPACK routines.

### 2.2 In-Place Factorization

The LU factors overwrite the original matrix in-place:

- the strict lower triangular part stores the multipliers of $L$ (diagonal of $L$ is implicit, equal to 1);
- the diagonal and upper triangular part store $U$.

Row permutations are tracked in a pivot index vector of length $n$ rather than a full $n \times n$ permutation matrix, keeping storage at $8n^2 + 8n$ bytes rather than $8n^2 + 8n^2$.

### 2.3 Pivot Threshold

The factorization aborts with `factorization_pivot_failure` if any pivot $|u_{kk}|$ falls below a configured absolute tolerance. This prevents silent propagation of division-by-near-zero through the remaining elimination steps. The threshold is a fixed constant; a more principled design would tie it to a running estimate of $\|A\|_\infty$ or an incremental condition estimate (see Section 9).

---

## 3. Verification

Three independent checks verify each factorization and solve:

**Factorization residual** — explicitly forms $LU - PA$ and computes $\|PA - LU\|_\infty$. This is $O(n^3)$ (a dense matrix product) and is enabled only in validation and testing paths.

**Solve residual** — computes the normalized residual

$$
\frac{\|b - Ax'\|_\infty}
{\|A\|_\infty \|x'\|_\infty + \|b\|_\infty}.
$$

This is \(O(n^2)\), since it requires a matrix-vector product, and is cheap relative to factorization cost.

**Relative forward error** — when a manufactured exact solution $x$ is available, computes $\|x' - x\|_\infty / \|x\|_\infty$.

Together these three quantities distinguish the four qualitatively different failure modes: incorrect factorization, incorrect triangular solve, large forward error due to ill-conditioning (with small backward error), and explicit pivot failure.

---

## 4. Test Matrix Families

| Family | Purpose | Expected behavior |
|---|---|---|
| Random dense | General nonsingular baseline | Near-$\varepsilon_\text{mach}$ residuals, moderate forward error |
| Diagonally dominant | Well-conditioned baseline | Cleaner than random; element growth is expected to remain controlled |
| Pivot-stress | Verify row interchanges | Zero factorization residual; tests permutation path |
| Hilbert | Severe ill-conditioning | Small backward error, large forward error; pivot failure at large $n$ |
| Near-singular | Failure detection | CLU: pivot failure at all $n$; EPLU: proceeds with large forward error |

**Diagonally dominant** matrices satisfy $|a_{ii}| > \sum_{j \neq i} |a_{ij}|$ for every row. For such matrices, Gaussian elimination is safe without pivoting — diagonal dominance is preserved through all elimination steps and $g_n$ is bounded. This project still uses partial pivoting as the general strategy; diagonally dominant cases serve as a clean correctness baseline.

**Pivot-stress** matrices are constructed to require row interchanges at every step. Exact-zero factorization residuals on these cases confirm that the permutation vector is applied consistently across both the factorization and triangular solve paths — a solver without correct permutation handling fails badly on these inputs.

---

## 5. Benchmark Methodology

Timings are median over 5 independent trials. The custom implementation is referred to as **CLU**; Eigen's `PartialPivLU` as **EPLU**.

| Operation | Flop count | Complexity |
|---|---|---|
| LU factorization | $\frac{2}{3}n^3$ | $O(n^3)$ |
| Triangular solve (single RHS) | $2n^2$ | $O(n^2)$ |
| Factorization residual check | $\sim 2n^3$ (matrix product) | $O(n^3)$ |
| Solve residual check | $\sim 2n^2$ (matrix-vector) | $O(n^2)$ |
| Dense matrix storage | $8n^2$ bytes | $O(n^2)$ |

Results are written to CSV and plotted with Python/Matplotlib. The primary CSV benchmark covers n = 8–256; a separate large-n run extends factorization timing to n = 2048.

### Benchmark Environment

Benchmarks were run on:

- Machine: MacBook Air, Late 2020
- CPU: Apple M1
- Memory: 8 GB
- Operating system: macOS Sequoia 15.6.1
- Compiler: AppleClang 17.0.0 (clang-1700.6.4.2)
- Target: arm64-apple-darwin24.6.0
- Build system: CMake 4.1.0
- Build type: Release
- Eigen version: 5.0.1
- Editor: Visual Studio Code

Unless otherwise stated, timing results use a Release build.      

---

## 6. Results

### 6.1a Pivot-Stress Matrices — Permutation Correctness

Pivot-stress matrices are constructed to require a row interchange at every
elimination step. Exact-zero factorization and solve residuals confirm that
the permutation vector, multiplier computation, and triangular factor storage
are mutually consistent. A solver with incorrect permutation handling fails
badly on these inputs.

| n | CLU ‖PA−LU‖∞ | CLU solve residual | CLU forward error ‖x′−x*‖∞ / ‖x*‖∞ |
|---|---|---|---|
| 8 | 0 | 0 | 0 |
| 16 | 0 | 0 | 0 |
| 32 | 0 | 0 | 0 |
| 64 | 0 | 0 | 0 |
| 128 | 0 | 0 | ~1.11×10⁻¹⁶ |
| 256 | 0 | 0 | ~2.23×10⁻¹⁶ |

Forward error at n = 128 and n = 256 reaches ~2ε_mach —  consistent with Theorem 5.5.1. The tested matrix families behave as if element growth is not the dominant source of error; residuals remain near machine precision on the successful cases. A future benchmark could explicitly measure $g_n$ to confirm whether element growth remains small across the tested matrix families.

---

### 6.1b Random Dense Matrices — Backward Error Scaling

![Accuracy and residuals vs n](plots/accuracy_residuals_vs_n.png)

| n | CLU ‖PA−LU‖∞ | CLU solve residual (normalized) | CLU forward error ‖x′−x*‖∞ / ‖x*‖∞ |
|---|---|---|---|
| 8 | ~1.0×10⁻¹⁶ | ~5×10⁻¹⁷ | ~6×10⁻¹⁶ |
| 16 | ~1.7×10⁻¹⁶ | ~9×10⁻¹⁷ | ~1.5×10⁻¹⁵ |
| 32 | ~2.5×10⁻¹⁶ | ~1.0×10⁻¹⁶ | ~4×10⁻¹⁵ |
| 64 | ~6×10⁻¹⁶ | ~2×10⁻¹⁶ | ~1×10⁻¹⁴ |
| 128 | ~9×10⁻¹⁶ | ~3×10⁻¹⁶ | ~3×10⁻¹³ |
| 256 | ~1.5×10⁻¹⁵ | ~8×10⁻¹⁶ | ~2×10⁻¹³ |
| 512 | ~3×10⁻¹⁵ | ~1.5×10⁻¹⁵ | ~2×10⁻¹² |

Factorization and solve residuals remain within a small multiple of ε_mach and grow
slowly with n, consistent with the $k_1(n) = n^2 g_n$ factor in Theorem 5.5.1. The
near-machine-precision residuals across all tested sizes suggest that element growth
was not the dominant source of error for these matrix families, though $g_n$ was not
explicitly measured. Forward error grows faster because it is further amplified by
$\kappa_\infty(A)$, which itself increases with n as the spectral spread of a random
dense matrix grows. The separation between the backward error curves (blue, gold) and
the forward error curve (green) is precisely the conditioning gap $\kappa_\infty(A)$ in action.

### 6.2 Hilbert Matrix — Forward vs. Backward Error Under Ill-Conditioning

![Hilbert Solution Error](plots/hilbert_solution_error.png)

| n | CLU forward error | EPLU forward error | CLU status |
|---|---|---|---|
| 4 | ~3×10⁻¹³ | ~2×10⁻¹³ | success |
| 8 | ~7×10⁻⁸ | ~7×10⁻⁸ | success |
| 12 | — | ~5×10⁻² | pivot failure |
| 16 | — | ~9×10⁻¹ | pivot failure |
| 24 | — | ~1.5 | pivot failure |
| 32 | — | ~3 | pivot failure |

At n = 4 and n = 8, both implementations agree closely and both exhibit rapidly growing forward error despite small solve residuals. This is the condition-number bound in action: the Hilbert matrix $H_n$ has $\kappa_\infty(H_n)$ that grows exponentially with $n$, so the forward error bound $\kappa_\infty(A) \cdot k_2(n) \cdot u$ exceeds 1 already at small $n$.

From n = 12 onward, CLU aborts with pivot failure. EPLU continues; at n = 16 the forward error exceeds 1, and at n = 32 it reaches $O(1)$ — the computed solution is numerically meaningless. Both solvers maintain small solve residuals throughout (Theorem 5.5.2 guarantees this regardless of conditioning); it is the forward error that reveals the ill-conditioning. The choice to abort rather than silently return a meaningless result reflects the design principle that in verified scientific computing, an explicit failure is safer than an incorrect number.

### 6.3 Near-Singular Matrices

CLU reports `factorization_pivot_failure` for all near-singular test cases at every tested size (n = 8–256, 5 trials each). EPLU returns solutions at all sizes without flagging failure. The conservative pivot threshold makes this behavior deterministic: CLU prioritizes explicit failure detection over permissive continuation. EPLU's approach — proceeding and returning a result — may be appropriate when downstream code performs residual checking; CLU's approach is safer when the solver result is consumed without further verification.

### 6.4 Factorization Timing — $\frac{2}{3}n^3$ Scaling

![Factorization Time vs n](plots/factorization_time_vs_n.png)

![Large-n factorization comparison](plots/large_n_factorization_comparison.png)

| n | CLU (ms) | EPLU (ms) | CLU/EPLU |
|---|---|---|---|
| 8 | 0.0014 | 0.0211 | 0.069× |
| 16 | 0.0097 | 0.0729 | 0.133× |
| 32 | 0.0626 | 0.1879 | 0.333× |
| 64 | 0.470 | 0.765 | 0.615× |
| 128 | 3.63 | 4.19 | 0.866× |
| 256 | 28.7 | 22.5 | 1.274× |
| 512† | ~230 | ~160 | ~1.44× |
| 1024† | ~1700 | ~1100 | ~1.55× |
| 2048† | ~14000 | ~9000 | ~1.56× |

†From large-n benchmark run; primary CSV covers n = 8–256.

Both implementations follow $\frac{2}{3}n^3$ scaling: doubling n multiplies factorization time by approximately $8\times$ for n ≥ 64, matching the theoretical prediction.

At n < 128, CLU is faster than EPLU. This is not an algorithmic advantage — it reflects EPLU's higher fixed per-call overhead: Eigen's expression template instantiation, internal dispatch, and thread-safety checks add costs that dominate when the matrix is small and the total flop count is a few thousand. Both implementations perform the same floating-point work at these sizes.

The crossover is around n = 128–256. At n = 256, EPLU is 1.27× faster; at n = 2048 the gap stabilizes at approximately 1.56×. The source is structural:

- **CLU (unblocked):** the trailing-matrix update at step $k$ is a sequence of rank-1 outer products, $a_{ij}^{(k+1)} \leftarrow a_{ij}^{(k)} - m_{ik} \cdot a_{kj}^{(k)}$ for $i,j > k$, executed as $n-k$ vector operations of length $n-k$. Each step touches $O((n-k)^2)$ memory with poor reuse.
- **EPLU (blocked):** the same computation is reorganized into panel updates $A_{22} \leftarrow A_{22} - L_{21}U_{12}$, the dominant cost can be expressed as a BLAS-3-style matrix-matrix update. This form has a more favorable flop-to-byte ratio than repeated rank-1 updates and allows much better cache reuse.

The unblocked formulation has no analogous data reuse and is limited by memory bandwidth at large n. At n = 512, the CLU matrix occupies $8 \times 512^2 \approx 2$ MB but the unblocked access pattern still generates many cache misses per elimination step as the working set changes at each column. Blocking is the targeted fix (Section 9). 

### 6.5 Triangular Solve Timing — $2n^2$ Scaling

![Solve Time vs n](plots/solve_time_vs_n.png)

| n | CLU solve (ms) | EPLU solve (ms) | CLU/EPLU |
|---|---|---|---|
| 8 | 0.000502 | 0.00882 | 0.057× |
| 16 | 0.00140 | 0.01736 | 0.081× |
| 32 | 0.00419 | 0.03714 | 0.113× |
| 64 | 0.01518 | 0.08554 | 0.178× |
| 128 | 0.05901 | 0.21952 | 0.269× |
| 256 | 0.23287 | 0.63114 | 0.369× |

Within the tested range (n = 8–256), CLU's single-RHS triangular solve is consistently faster than EPLU's measured solve step, with both following $O(n^2)$ scaling (doubling n multiplies solve time by ~4×).

The CLU advantage at these sizes reflects the simpler dispatch path: the custom solve is a direct loop over the in-place LU storage with no expression template overhead. **This result must not be extrapolated to larger n**: EPLU uses highly optimized Eigen triangular-solve kernels, and depending on configuration may exploit vectorized or BLAS-like dense linear algebra paths.

In any case, this comparison is secondary: the triangular solve costs $2n^2$ flops while factorization costs $\frac{2}{3}n^3$. At n = 256, factorization takes 28.7 ms and the solve takes 0.233 ms — a ratio of over 120×. Solve performance has no material effect on end-to-end throughput for single-matrix problems.

### 6.6 Multiple Right-Hand Sides (n = 256)

![Multiple RHS solve time](plots/multiple_rhs_solve_loop_time.png)

![Multiple RHS Time per RHS](plots/multiple_rhs_time_per_rhs.png)

![Multiple RHS total time](plots/multiple_rhs_total_time.png)

| n_rhs | CLU solve-loop (ms) | EPLU solve-loop (ms) | CLU ms/RHS | EPLU ms/RHS |
|---|---|---|---|---|
| 1 | 0.235 | 0.683 | 0.235 | 0.683 |
| 2 | 0.471 | 1.065 | 0.235 | 0.533 |
| 4 | 0.942 | 1.237 | 0.235 | 0.309 |
| 8 | 1.884 | 2.027 | 0.235 | 0.254 |
| 16 | 3.769 | 3.725 | 0.236 | 0.233 |

CLU's time per RHS is constant at ~0.235 ms regardless of n_rhs, because each column is solved independently by sequential forward and backward substitution with no cross-column data reuse.

EPLU's time per RHS drops from 0.683 ms at n_rhs = 1 to 0.233 ms at n_rhs = 16, a 2.9× improvement. EPLU uses a matrix-level triangular solve strategy analogous to BLAS `trsm` for multiple RHS: all n_rhs columns are processed simultaneously as a dense matrix, reorganizing the computation into blocked matrix-matrix operations. The CLU column-by-column implementation cannot exploit this structure without a fundamentally different data access pattern. 

Total solve-loop times converge at n_rhs = 16 (3.769 ms CLU vs. 3.725 ms EPLU) on this hardware. At larger n_rhs the EPLU advantage would grow. However, for all tested cases the factorization dominates total time: at n_rhs = 16, CLU total is 28.998 + 3.769 = 32.8 ms vs. EPLU 22.511 + 3.725 = 26.2 ms — a gap driven entirely by the factorization stage.

### 6.7 Residual Verification Cost

![Residual Timing vs n](plots/residual_timing_vs_n.png)

| n | Fact. residual — debug (ms) | Fact. residual — fast (ms) | Solve residual (ms) |
|---|---|---|---|
| 8 | 0.0042 | 0.0033 | 0.00065 |
| 16 | 0.0214 | 0.0178 | 0.00193 |
| 32 | 0.136 | 0.124 | 0.00720 |
| 64 | 0.930 | 0.876 | 0.0273 |
| 128 | 6.82 | 6.59 | 0.107 |
| 256 | 52.2 | 51.3 | 0.425 |

The factorization residual check is $O(n^3)$: it explicitly forms $LU - PA$ via a dense matrix product and norms the result. At n = 256 this costs 51–52 ms — nearly $2\times$ the factorization time of 28.7 ms. The debug and fast implementation variants have nearly identical cost, confirming that both paths perform the same dominant matrix-matrix work; implementation differences are negligible compared to the algorithmic cost.

The solve residual check is $O(n^2)$ and remains below 0.5 ms through n = 256. At n = 256 it costs 0.425 ms — less than 1.5% of factorization time — making it suitable for standard error reporting paths.

These measurements make verification overhead explicit. In production numerical software, factorization residual checks are enabled only in testing, validation, or diagnostic modes; the solve residual check is cheap enough for normal use.

### 6.8 Memory Footprint

![Theoretical Memory vs n](plots/theoretical_memory_vs_n.png)

| n | Single RHS (MB) | 16-RHS (MB) |
|---|---|---|
| 8 | 0.0012 | 0.0041 |
| 32 | 0.0166 | 0.0260 |
| 64 | 0.0645 | 0.0896 |
| 128 | 0.254 | 0.300 |
| 256 | 1.008 | 1.100 |
| 512 | 4.016 | — |

Theoretical storage scales as $O(n^2)$, dominated by the in-place $n \times n$ LU matrix ($8n^2$ bytes for double precision).  At \(n = 256\), the modeled single-RHS footprint is approximately 1.0 MB. At \(n = 512\), one dense double-precision matrix alone occupies approximately 2 MB, while the modeled single-RHS footprint is approximately 4.0 MB because the benchmark stores both the original matrix and the in-place LU matrix.

Additional storage for multiple RHS ($8n \cdot n_\text{rhs}$ bytes) is negligible relative to the $n \times n$ matrix: at n = 256, n_rhs = 16, the extra storage is 0.25 MB versus 1.0 MB for the matrix itself.

Measured OS-level process RSS was approximately 2.5 MB throughout, flat across all matrix sizes, reflecting OS page allocation granularity and allocator overhead. The theoretical model is the authoritative measure of algorithmic storage complexity.

---

## 7. Building and Running

**Requirements:** C++17 compiler, CMake ≥ 3.14, Eigen ≥ 3.4. Python ≥ 3.8 with Matplotlib for plots only.

### Build

```bash
git clone <repository-url>
cd <repository-name>
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

If CMake cannot locate Eigen automatically:

```bash
# macOS (Homebrew)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DEigen3_DIR="/opt/homebrew/share/eigen3/cmake"

# Manual Eigen path
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DEigen3_DIR="<path-to-eigen3-cmake>"
```

Windows (Visual Studio):

```powershell
cmake -S . -B build
cmake --build build --config Release
```

### Run

```bash
# Linux / macOS
./build/solve_demo      # worked example with residual output
./build/test_lu         # correctness and accuracy test suite
./build/bench_lu        # performance benchmarks; writes CSV to results/

# Windows
.\build\Release\solve_demo.exe
.\build\Release\test_lu.exe
.\build\Release\bench_lu.exe
```

### Plot

```bash
python3 scripts/plot_bench.py   # Linux / macOS
py scripts\plot_bench.py        # Windows
```

Run `bench_lu` before plotting.

---

## 8. Limitations

**Unblocked factorization.** The trailing-matrix update is a sequence of rank-1 outer products with no BLAS-3 structure, making the implementation memory-bandwidth bound at large n. This is why EPLU is ~1.56× faster at n = 2048. Blocking is the primary performance gap to close.

**Column-by-column triangular solve.** Each RHS is solved independently with no cross-column data reuse, producing constant per-RHS time but missing the 2.9× per-RHS improvement EPLU achieves via blocked trsm for large n_rhs batches.

**Serial execution.** No OpenMP, SIMD intrinsics, or GPU offload. A correct and benchmarked serial implementation is a prerequisite for any parallel extension; the current implementation serves this role.

**Fixed pivot threshold.** The abort threshold is a compile-time constant rather than a function of $\|A\|_\infty$ or an incremental condition estimate. A principled threshold would scale with the matrix norm and connect directly to the Theorem 5.5.1/5.5.2 bounds.

**No condition number estimation.** $\kappa_\infty(A)$ is not estimated for arbitrary inputs; known ill-conditioned families (Hilbert matrices) serve as proxies. A LINPACK-style incremental condition estimator would make the solver diagnostically useful for arbitrary inputs.

**Benchmark range.** Primary CSV data covers n = 8–256. Large-n factorization data (n = 512–2048) comes from a separate benchmark run and is presented via plots only; solve and multiple-RHS benchmarks are not available above n = 256.

**Platform-specific memory measurement.** The measured process-memory utility is implemented only for Windows, Linux, and macOS. The theoretical memory model is platform-independent, but measured RSS depends on operating-system APIs, allocator behavior, and page granularity.

---

## 9. Future Work

**Blocked LU factorization** — restructure the trailing-matrix update as $A_{22} \leftarrow A_{22} - L_{21}U_{12}$ with panel width $b$ chosen to match L2/L3 cache capacity, enabling a BLAS-3-style matrix-matrix update for the dominant computation. This is the approach used in LAPACK's `dgetrf` and is the direct fix for the 1.56× factorization gap against EPLU.

**Matrix-level triangular solve (trsm)** — accumulate all RHS columns into a dense matrix and solve via a blocked trsm, matching EPLU's per-RHS scaling for large n_rhs batch workloads.

**OpenMP parallelization** — after blocking, the trailing-matrix dgemm call parallelizes cleanly across rows; residual verification loops also parallelize without synchronization. Thread-scaling experiments are a natural follow-on benchmark.

**Iterative refinement** — compute correction $\Delta x$ by solving $A\Delta x = r$ where $r = b - Ax'$, then update $x' \leftarrow x' + \Delta x$. This connects the Theorem 5.5.1/5.5.2 backward error bounds directly to forward error improvement and provides a controlled experimental setting for studying the backward-error / conditioning / forward-error triangle.

**Incremental condition estimation** — replace the fixed pivot threshold with a LINPACK-style norm estimator tracking $\kappa_\infty(A)$ during factorization, giving a data-driven stopping criterion tied directly to the Theorem 5.5.1 bound.

**PDE-driven linear systems** — the solver infrastructure (factorization, residual verification, benchmarking harness) applies directly to linear systems from finite difference and finite element discretizations. A natural extension uses CLU as the direct solver within an implicit time-stepping scheme for the heat equation or Poisson equation, connecting this dense direct solver to the spectral and finite-difference projects that follow it.

---

## 10. Issues and Feedback

If something in the build instructions, benchmark outputs, plots, documentation, or mathematical discussion does not work as expected, please open an issue on the repository.

Corrections, suggestions, and feedback are welcome, especially regarding numerical accuracy, mathematical statements, benchmarking methodology, portability, or documentation clarity.


---

## 11. References

[1] G. Dahlquist and Å. Björck, *Numerical Methods*. Dover Publications. — Primary reference for the LU existence theorem 5.3.1, backward error Theorems 5.5.1 and 5.5.2, growth factor bounds, and the practical residual criterion used throughout this project.

[2] E. Anderson et al., *LAPACK Users' Guide*, 3rd ed. SIAM, 1999. — Reference for `dgetrf` (blocked LU with partial pivoting), `dtrsm` (matrix triangular solve), and `dgecon` (LINPACK-style condition estimation).

[3] Eigen documentation, `Eigen::PartialPivLU`. https://eigen.tuxfamily.org — Reference implementation used for all performance and accuracy comparisons.