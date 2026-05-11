#pragma once

#include "bench_types.hpp"
#include "bench_common.hpp"
#include "lu.hpp"

#include <cstddef>
#include <string>
#include <vector>




// Accuracy helpers


// Computes custom LU accuracy after a completed solve.
// What it does:
//   Uses your existing residual_factorization_fast/residual_solve functions and
//   your existing relative solution error helpers to fill an AccuracyResult.
// Why needed:
//   Keeps residual checks outside timed regions while still recording numerical
//   correctness for every benchmark case.


AccuracyResult compute_custom_lu_accuracy(const LUProblem& problem, const std::vector<double>& LU,
 const std::vector<std::size_t>& piv, const std::vector<double>& x_computed, 
 std::size_t trial, NormType norm, const std::string& status);






// Custom LU benchmark operations


// Benchmarks custom LU factorization only.
// What it does:
//   Copies A0 into a working LU array, initializes pivots, times only
//   LU_factorize, and returns one TimingResult.
// Why needed:
//   Measures PA = LU cost separately from solve cost, residual checks, matrix
//   generation, memory recording, and CSV writing.


TimingResult benchmark_custom_factorization_once(const LUProblem& problem, std::size_t trial, const BenchmarkConfig& config);


// Benchmarks custom LU solve only.
// What it does:
//   Assumes LU and piv are already factorized outside the timed region, then
//   times only LU_solve.
// Why needed:
//   Measures forward/backward substitution cost separately from factorization.


TimingResult benchmark_custom_solve_once(const LUProblem& problem, const std::vector<double>& LU, const std::vector<std::size_t>& piv, std::size_t trial,
 const BenchmarkConfig& config);



// Benchmarks custom full solve.
// What it does:
//   Times the full operation of factorization followed by solve.
// Why needed:
//   Measures the practical end-to-end cost of solving one dense system Ax = b


//   from scratch.


TimingResult benchmark_custom_full_solve_once(const LUProblem& problem, std::size_t trial, const BenchmarkConfig& config);



// Runs one complete custom LU benchmark case.
// What it does:
//   For one n, matrix type, and trial, runs factorization timing, solve timing,
//   full-solve timing, accuracy checks, and memory checkpoints.
// Why needed:
//   Keeps benchmark_lu.cpp simple by collecting one complete case through a
//   single driver function.


void run_custom_lu_case(const BenchmarkConfig& config, std::size_t n, MatrixType matrix_type, std::size_t trial,
 std::vector<TimingResult>& timing_results, std::vector<AccuracyResult>& accuracy_results, std::vector<MemoryCheckpoint>& memory_results);



// Runs the full custom LU benchmark suite.
// What it does:
//   Loops over all configured sizes, matrix types, and trials, then calls
//   run_custom_lu_case for each case.
// Why needed:
//   Provides the top-level custom LU benchmark entry point for benchmark_lu.cpp.

void run_custom_lu_benchmarks(const BenchmarkConfig& config, std::vector<TimingResult>& timing_results, std::vector<AccuracyResult>& accuracy_results,
 std::vector<MemoryCheckpoint>& memory_results);
