# Dense LU Decomposition in C++: Accuracy, Stability, and Performance Analysis

## 1. Overview

<!-- Write this section yourself. -->

## 2. Mathematical Background

<!-- Write this section yourself. -->

## 3. Implementation

<!-- Write this section yourself. -->

## 4. Verification

<!-- Write this section yourself. -->

## 5. Test Matrix Families

<!-- Write this section yourself. -->

## 6. Benchmark Methodology

<!-- Write this section yourself. -->

---

## 7. Results

All timings are median over 5 trials on a M1 Macbook Air with 8Gb of RAM. Reported residuals use the ∞-norm unless stated otherwise. The custom implementation is referred to as **CLU**; Eigen's `PartialPivLU` as **EPLU**.

---

### 7.1 Backward Error — Random Dense and Diagonally Dominant Matrices

| n   | CLU fact. residual ‖PA−LU‖∞ | CLU solve residual (normalized) | CLU forward error ‖x−x*‖∞ |
|-----|-----------------------------|---------------------------------|---------------------------|
| 8   | ~1.0×10⁻¹⁶                 | ~5×10⁻¹⁷                       | ~6×10⁻¹⁶                 |
| 64  | ~6×10⁻¹⁶                   | ~2×10⁻¹⁶                       | ~1×10⁻¹⁴                 |
| 256 | ~2×10⁻¹⁵                   | ~1×10⁻¹⁵                       | ~1×10⁻¹³                 |
| 512 | ~3×10⁻¹⁵                   | ~2×10⁻¹⁵                       | ~2×10⁻¹³                 |

Factorization and solve residuals remain within a small multiple of ε_mach and grow slowly with n, consistent with the O(nε_mach) backward error bound for partial pivoting LU on well-conditioned matrices. Forward error grows faster, as expected: it is bounded by κ(A) times the backward error, and even moderately conditioned random dense matrices accumulate nontrivial conditioning amplification at n = 512.

Diagonally dominant matrices produce smaller residuals and forward errors at every size, which is expected since diagonal dominance implies bounded growth factors during Gaussian elimination.

Pivot-stress matrices — structured to require row interchanges at every step — yield factorization residuals of exactly 0 across all tested sizes (n = 8 to 256), verifying that the permutation vector is consistently applied through both triangular factors and the forward/backward substitution paths.

---

### 7.2 Hilbert Matrix — Forward vs. Backward Error Under Ill-Conditioning

| n  | CLU forward error | EPLU forward error | CLU status    |
|----|-------------------|--------------------|---------------|
| 4  | ~3×10⁻¹³         | ~2×10⁻¹³          | success       |
| 8  | ~7×10⁻⁸          | ~7×10⁻⁸           | success       |
| 12 | —                 | ~5×10⁻²           | pivot failure |
| 16 | —                 | ~9×10⁻¹           | pivot failure |
| 24 | —                 | ~1.5               | pivot failure |
| 32 | —                 | ~3                 | pivot failure |

At n = 4 and n = 8 both implementations agree closely, correctly reflecting the known ill-conditioning of small Hilbert systems. From n = 12 onward CLU's pivot threshold detects that the factorization has become numerically unsafe and halts with an explicit failure. EPLU continues and returns a solution, but the forward error exceeds 1 at n = 16 and grows to O(1) at n = 32 — the computed solution is numerically meaningless, yet no failure is raised.

This experiment illustrates a fundamental point in numerical linear algebra: small backward error (‖b − Ax‖ small) does not bound forward error (‖x − x*‖) when κ(A) is large. The design choice here — explicit failure over silent inaccuracy — is appropriate for a solver intended to be used in verified scientific computations where an unreliable solution is worse than no solution.

---

### 7.3 Singular and Near-Singular Matrices

CLU reports `factorization_pivot_failure` for all tested singular and near-singular matrices at every size (n = 8 to 256, 5 trials each). EPLU returns solutions with forward errors in the range 10⁻⁷ to 10⁻⁵ for near-singular inputs — not flagged as failures, but produced from a numerically unsafe factorization. For truly singular matrices, EPLU similarly proceeds without error, returning solutions with large but finite forward errors. The difference reflects a design tradeoff between conservative failure detection and permissive continuation; neither is universally correct, but explicit failure is the safer default for a general-purpose solver.

---

### 7.4 Factorization Timing and O(n³) Scaling

| n    | CLU (ms) | EPLU (ms) | CLU/EPLU |
|------|----------|-----------|----------|
| 8    | 0.005    | 0.021     | 0.24×    |
| 32   | 0.063    | 0.188     | 0.33×    |
| 128  | 3.63     | 4.19      | 0.87×    |
| 256  | 28.7     | 22.5      | 1.27×    |
| 512  | ~230     | ~160      | ~1.44×   |
| 1024 | ~1700    | ~1100     | ~1.55×   |
| 2048 | ~14000   | ~9000     | ~1.56×   |

Both implementations exhibit O(n³) scaling across the full tested range (n = 8 to 2048): doubling n multiplies factorization time by approximately 8× once n is large enough that the cubic term dominates.

At small n (below ~128), CLU is faster than EPLU. This is not an algorithmic advantage — it is a consequence of EPLU's higher per-call overhead: template instantiation, expression template resolution, and internal dispatch in Eigen add fixed costs that dominate at small matrix sizes. The two implementations perform roughly the same amount of floating-point work at these sizes.

The crossover occurs around n = 128–256, after which EPLU is consistently faster. At n = 2048, EPLU is approximately 1.56× faster. This gap is structural: CLU is an unblocked LU factorization in which the trailing-matrix update is performed as a sequence of rank-1 outer products, while EPLU's implementation reorganizes the same computation into panel updates backed by highly optimized BLAS-3 matrix-matrix kernels. BLAS-3 routines achieve near-peak arithmetic throughput because their O(n³) flop count is served by O(n²) memory traffic, allowing the arithmetic units to stay saturated. The unblocked formulation does not exploit this ratio and is limited by memory bandwidth at large n.

Addressing this gap is the primary motivation for the blocked LU extension described in Section 10.

---

### 7.5 Triangular Solve Timing

| n   | CLU solve (ms) | EPLU solve (ms) | CLU/EPLU |
|-----|----------------|-----------------|----------|
| 8   | 0.00049        | 0.0088          | 0.056×   |
| 64  | 0.0152         | 0.0855          | 0.18×    |
| 256 | 0.233          | 0.631           | 0.37×    |
| 512 | ~0.92          | ~1.7            | ~0.54×   |

Within the tested range (n = 8 to 512), CLU's triangular solve is faster than EPLU's. Both follow O(n²) scaling, and the absolute times are small relative to the O(n³) factorization for any n large enough to matter in practice.

The observed CLU advantage at these sizes is likely a function of the simpler dispatch path: the custom implementation is a direct loop over the in-place LU storage with no abstraction overhead, while EPLU's solve involves additional expression template machinery. However, this advantage should not be extrapolated: at larger n, EPLU's trsv (triangular solve with a vector) kernel is backed by BLAS-2 routines with vectorized memory access patterns, and the gap would narrow or reverse. The benchmarked range is not sufficient to make claims about asymptotic solve performance.

In any case, the triangular solve is O(n²) while factorization is O(n³); for n ≥ 256 the factorization dominates total runtime by more than 100×, so the solve performance has no meaningful effect on end-to-end throughput.

---

### 7.6 Multiple Right-Hand Sides (n = 256)

A key use case for direct solvers is factorizing a matrix once and solving AX = B for many right-hand-side vectors by reusing the stored LU factors.

| n_rhs | CLU solve-loop (ms) | EPLU solve-loop (ms) | CLU ms/RHS | EPLU ms/RHS |
|-------|---------------------|----------------------|------------|-------------|
| 1     | 0.27                | 0.68                 | 0.270      | 0.680       |
| 2     | 0.49                | 1.08                 | 0.245      | 0.540       |
| 4     | 0.96                | 1.24                 | 0.240      | 0.310       |
| 8     | 1.90                | 2.03                 | 0.238      | 0.254       |
| 16    | 3.75                | 3.71                 | 0.234      | 0.232       |

CLU's time per RHS is nearly constant (~0.24 ms) regardless of n_rhs, because the implementation iterates over RHS columns independently: each solve is a sequential forward and backward substitution with no cross-column data reuse.

EPLU's per-RHS time drops from 0.68 ms at n_rhs = 1 to 0.23 ms at n_rhs = 16, a 3× improvement. This is a direct consequence of EPLU switching to a matrix-level triangular solve (trsm) when multiple RHS columns are available. A trsm call processes all columns simultaneously via blocked matrix-matrix operations, significantly improving cache reuse and arithmetic intensity relative to n_rhs independent trsv calls.

The total solve-loop times converge at n_rhs = 16 (3.75 ms CLU vs 3.71 ms EPLU) — at this batch size both are essentially equivalent on this hardware. At larger n_rhs the EPLU advantage would grow as the trsm kernel becomes increasingly efficient, while CLU's time would continue growing linearly. This identifies a clear structural limitation of the current implementation for batch-solve workloads.

---

### 7.7 Residual Verification Cost

| n   | Fact. residual check (ms) | Solve residual check (ms) | Fact. check / factorization |
|-----|---------------------------|---------------------------|-----------------------------|
| 8   | ~0.004                    | ~0.0006                   | ~0.8×                       |
| 64  | ~0.93                     | ~0.027                    | ~2.0×                       |
| 128 | ~6.8                      | ~0.107                    | ~1.9×                       |
| 256 | ~52                       | ~0.42                     | ~1.8×                       |

The factorization residual check — computing ‖PA − LU‖∞ by explicitly forming PA and LU and differencing them — is an O(n³) operation and costs roughly as much as the factorization itself. At n = 256 it adds ~52 ms to a ~29 ms factorization. The solve residual check (‖b − Ax‖∞) is O(n²) and negligible relative to factorization cost at all tested sizes.

The debug and fast variants of the factorization residual check produce nearly identical timings, indicating that the two implementation paths have the same dominant cost and that the difference between them is not a performance concern.

In production numerical software, factorization residual checks are computed only in validation, testing, or diagnostic modes and are never part of the hot path. This benchmark makes the verification overhead explicit and quantifies the cost of including such checks during development.

---

### 7.8 Memory Footprint

| n   | Single-RHS (MB) | 16-RHS (MB) |
|-----|-----------------|-------------|
| 8   | 0.0012          | 0.0041      |
| 64  | 0.064           | 0.087       |
| 128 | 0.254           | 0.300       |
| 256 | 1.008           | 1.100       |
| 512 | 4.016           | 4.109       |

Theoretical storage scales as O(n²), dominated by the n×n in-place LU matrix (two double-precision arrays of size n²). Additional storage for RHS and solution vectors is O(n·n_rhs) and becomes relevant only when n_rhs is large relative to n. At n = 256 the theoretical single-RHS footprint is ~1.0 MB, well within L3 cache on modern hardware; at n = 512 it is ~4.0 MB, at which point cache pressure begins to affect the unblocked factorization's memory access patterns.

Measured OS-level process RSS was flat across all matrix sizes at approximately 2.5 MB due to OS page allocation granularity and allocator overhead. The theoretical model is the authoritative measure of algorithmic storage complexity; OS-level measurements are retained as a practical diagnostic but not used for complexity analysis.

---

## 8. Building and Running

Requires a C++17 compiler, CMake ≥ 3.14, and Eigen ≥ 3.4. Python ≥ 3.8 with Matplotlib is required only for generating benchmark plots.

### Build

```bash
git clone <repository-url>
cd <repository-name>
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

If CMake cannot locate Eigen automatically:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DEigen3_DIR="<path-to-eigen3-cmake>"
cmake --build build
```

macOS (Homebrew):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DEigen3_DIR="/opt/homebrew/share/eigen3/cmake"
cmake --build build
```

Windows (Visual Studio):

```powershell
cmake -S . -B build
cmake --build build --config Release
```

### Run

```bash
# Linux / macOS
./build/lu_example      # worked example with residual output
./build/test_lu         # correctness and accuracy test suite
./build/bench_lu        # performance benchmarks (writes CSV to results/)

# Windows
.\build\Release\lu_example.exe
.\build\Release\test_lu.exe
.\build\Release\bench_lu.exe
```

### Plot

```bash
python3 scripts/plot_bench.py   # Linux / macOS
py scripts\plot_bench.py        # Windows
```

`bench_lu` must be run before plotting; the script reads the CSV files written to `results/`.

---

## 9. Limitations

- **Unblocked factorization.** The trailing-matrix update is performed as a sequence of rank-1 outer products, with no exploitation of BLAS-3 structure. This limits arithmetic intensity at large n: the algorithm is memory-bandwidth bound rather than compute-bound, which is why EPLU's factorization is ~1.56× faster at n = 2048. Blocking the factorization is the primary performance gap to address.

- **Column-by-column triangular solve.** Each RHS column is solved independently via sequential forward and backward substitution. This is correct and produces flat per-RHS time, but it cannot exploit the improved cache efficiency of a matrix-level trsm call. The consequence is visible in the multiple-RHS benchmark: CLU's per-RHS time is constant while EPLU's drops 3× over the tested range.

- **Serial execution.** The implementation is entirely single-threaded. No OpenMP, SIMD intrinsics, or GPU offload is used. The serial baseline was the explicit scope of this version; parallel extensions require a correct and benchmarked serial implementation as a reference.

- **Conservative pivot threshold.** The factorization aborts when a pivot falls below the configured tolerance, providing explicit failure detection at the cost of rejecting some near-singular matrices that a more permissive solver would attempt. The threshold is configurable but not yet tied to a scaling or condition estimate; a more principled approach would couple it to an incremental condition estimate during factorization.

- **No condition number estimation.** Residuals and forward errors are reported when a manufactured solution is available, but κ(A) is not estimated for arbitrary inputs. Known ill-conditioned families (Hilbert matrices) serve as proxies. A LINPACK-style condition estimator would make the solver's diagnostics useful for arbitrary user-supplied matrices.

---

## 10. Future Work

- **Blocked LU factorization** — restructure the trailing-matrix update as A₂₂ ← A₂₂ − L₂₁U₁₂ with panel width chosen to match L2/L3 cache capacity, enabling BLAS-3 dgemm for the dominant computation. This is the standard approach used in LAPACK's dgetrf and is the most direct path to closing the factorization performance gap against EPLU.

- **Matrix-based triangular solve (trsm)** — accumulate all RHS columns into a dense matrix and call a blocked triangular solve, matching EPLU's per-RHS scaling for large n_rhs batch workloads.

- **OpenMP parallelization** — after blocking, the trailing-matrix update parallelizes cleanly across panel columns; residual verification loops also parallelize without synchronization. Thread-level scaling experiments would be a natural follow-on benchmark.

- **Iterative refinement** — compute a correction x ← x + A⁻¹(b − Ax) after the initial solve to improve forward accuracy for moderately ill-conditioned systems. This would also provide a natural framework for connecting backward error, conditioning, and forward error in a single experiment, extending the Hilbert matrix analysis.

- **Incremental condition estimation** — couple a LINPACK-style or norm-estimation-based κ(A) estimate to the pivot threshold logic, replacing the current fixed tolerance with a data-driven stopping criterion.

- **Sparse and iterative solvers** — the dense direct solver infrastructure (factorization, residual verification, benchmarking harness) establishes patterns applicable to sparse LU, ILU preconditioning, and Krylov subspace methods, all of which are natural extensions toward PDE-scale problems.