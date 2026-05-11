#pragma once

#include "bench_types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>


// Multiple rhs benchmark operations

 // Benchmarks custom LU solving many RHS after one factorization.
// What it does:
//   Builds X_true and B = A*X_true, factorizes A once, then times solving each
//   RHS column using LU_solve.
// Why needed:
//   Demonstrates the main advantage of LU: reusing one factorization for many
//   right-hand sides.

MultipleRHSResult benchmark_custom_multiple_rhs_once(const LUProblem& problem, std::size_t nrhs, std::size_t trial,
 NormType norm, std::uint32_t seed, const BenchmarkConfig& config);



// Runs one custom LU multiple-RHS benchmark case.
// What it does:
//   Creates one LUProblem, loops over configured RHS counts, and appends custom
//   LU multiple-RHS results.
// Why needed:
//   Keeps the top-level benchmark loop clean.

void run_custom_multiple_rhs_case(const BenchmarkConfig& config, std::size_t n, MatrixType matrix_type, std::size_t trial,
 std::vector<MultipleRHSResult>& multiple_rhs_results, std::vector<MemoryCheckpoint>& memory_results);



 // Runs all custom LU multiple-RHS benchmarks.
// What it does:
//   Loops over config.sizes, config.matrix_types, config.trials, and
//   config.rhs_counts.
// Why needed:
//   Provides the top-level custom LU multiple-RHS benchmark entry point.

void run_custom_multiple_rhs_benchmarks(const BenchmarkConfig& config, std::vector<MultipleRHSResult>& multiple_rhs_results,
 std::vector<MemoryCheckpoint>& memory_results);
