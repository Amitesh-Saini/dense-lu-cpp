// bench_residuals.hpp
// Purpose:
//   Declare benchmark routines for residual-checking utilities.
//
// Contents:
//   - Factorization residual timing for debug and fast residual paths.
//   - Solve residual timing.
//   - Multiple-RHS residual timing.
//   - Per-case and full-suite residual benchmark runners.

#pragma once

#include "bench_types.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>


// Residual checker benchmark operations


// Benchmarks debug factorization residual only.
// What it does:
//   Times residual_factorization_debug on an already-factorized LU.
// Why needed:
//   Measures the cost of the clear but allocation-heavy verification path.


ResidualTimingResult benchmark_factorization_residual_debug_once(
    const LUProblem& problem, const std::vector<double>& LU, const std::vector<std::size_t>& piv, std::size_t trial, NormType norm, 
    const BenchmarkConfig& config);


// Benchmarks fast factorization residual only.
// What it does:
//   Times residual_factorization_fast on an already-factorized LU.
// Why needed:
//   Measures the optimized residual checker and allows comparison against the
//   debug residual checker.


ResidualTimingResult benchmark_factorization_residual_fast_once(
    const LUProblem& problem, const std::vector<double>& LU, const std::vector<std::size_t>& piv, std::size_t trial, NormType norm, 
    const BenchmarkConfig& config);



// Benchmarks solve residual only.
// What it does:
//   Times residual_solve after a completed solve.
// Why needed:
//   Measures the O(n^2) solve verification cost separately from the O(n^3)
//   factorization verification cost.

ResidualTimingResult benchmark_solve_residual_once(
    const LUProblem& problem, const std::vector<double>& x, std::size_t trial, NormType norm, const BenchmarkConfig& config);




// Runs one residual benchmark case.
// What it does:
//   Factorizes and solves a problem once, then times debug factorization
//   residual, fast factorization residual, and solve residual.
// Why needed:
//   Compares verification overheads under the same matrix and solution setup.


void run_residual_benchmark_case(
    const BenchmarkConfig& config, std::size_t n, MatrixType matrix_type, std::size_t trial, std::vector<ResidualTimingResult>& residual_results, 
    std::vector<MemoryCheckpoint>& memory_results);



// Runs the full residual benchmark suite.
// What it does:
//   Loops over configured sizes, matrix types, and trials for residual checker
//   timing.
// Why needed:
//   Provides the top-level residual benchmark entry point for benchmark_lu.cpp.
void run_residual_benchmarks(const BenchmarkConfig& config, std::vector<ResidualTimingResult>& residual_results, std::vector<MemoryCheckpoint>& memory_results);


 // Benchmarks multiple-RHS solve residual checking once.
// What it does:
//   Builds a multiple-RHS system, factorizes A once, solves all RHS columns,
//   then times only multiple_rhs_solve_residuals(...).
// Why needed:
//   Measures the verification overhead for A*X = B separately from solve timing.
MultipleRHSResidualTimingResult benchmark_multiple_rhs_residual_once(
    const LUProblem& problem, std::size_t nrhs, std::size_t trial, NormType norm, std::uint32_t seed, const BenchmarkConfig& config);

// Runs one multiple-RHS residual benchmark case.
// What it does:
//   Creates one LUProblem, loops over config.rhs_counts, and records residual
//   timing results for each nrhs.
// Why needed:
//   Keeps per-case residual benchmark setup out of the top-level benchmark file.
void run_multiple_rhs_residual_case(
    const BenchmarkConfig& config, std::size_t n, MatrixType matrix_type, std::size_t trial, std::vector<MultipleRHSResidualTimingResult>& residual_results, 
    std::vector<MemoryCheckpoint>& memory_results);

 // Runs all multiple-RHS residual benchmarks.
// What it does:
//   Loops over config.sizes, config.matrix_types, config.trials, and
//   config.rhs_counts.
// Why needed:
//   Provides the top-level driver for benchmarking multiple-RHS verification cost.
void run_multiple_rhs_residual_benchmarks(
    const BenchmarkConfig& config, std::vector<MultipleRHSResidualTimingResult>& residual_results, std::vector<MemoryCheckpoint>& memory_results);
