#pragma once

#include "bench_types.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

// Converts your flat row-major std::vector matrix into an Eigen::MatrixXd.
// What it does:
//   Takes A stored as A[i*n + j] and copies it into an Eigen dense matrix.
// Why needed:
//   Eigen benchmarks need the same benchmark matrix as your custom LU, but in
//   Eigen's matrix format.
Eigen::MatrixXd to_eigen_matrix(const std::vector<double>& A, std::size_t n);

// Converts a std::vector<double> into an Eigen::VectorXd.
// What it does:
//   Copies vector entries from your benchmark format into Eigen's vector format.
// Why needed:
//   Eigen's solve function expects b as an Eigen::VectorXd.
Eigen::VectorXd to_eigen_vector(const std::vector<double>& x);

// Converts an Eigen::VectorXd back into std::vector<double>.
// What it does:
//   Copies Eigen's computed solution into your normal std::vector<double> format.
// Why needed:
//   Your existing residual and solution-error helpers work with std::vector<double>.
std::vector<double> from_eigen_vector(const Eigen::VectorXd& x);

// Benchmarks Eigen PartialPivLU factorization only.
// What it does:
//   Converts A0 to Eigen format, then times only Eigen's LU factorization.
// Why needed:
//   Provides a professional library baseline for your custom LU factorization timing.
TimingResult benchmark_eigen_factorization_once(const LUProblem& problem, std::size_t trial, const BenchmarkConfig& config);

// Benchmarks Eigen solve only after factorization.
// What it does:
//   Factorizes the matrix outside the timed region, then times only Eigen's solve.
// Why needed:
//   Compares Eigen's solve/reuse cost against your custom LU_solve benchmark.
TimingResult benchmark_eigen_solve_once(const LUProblem& problem, std::size_t trial, const BenchmarkConfig& config);

// Benchmarks Eigen full solve.
// What it does:
//   Times Eigen factorization followed by Eigen solve.
// Why needed:
//   Gives an end-to-end Eigen baseline for solving Ax = b from scratch.
TimingResult benchmark_eigen_full_solve_once(const LUProblem& problem, std::size_t trial, const BenchmarkConfig& config);

// Computes Eigen accuracy after a completed Eigen solve.
// What it does:
//   Uses your existing residual_solve and solution-error helpers to compare
//   Eigen's computed solution against b = A*x and x_true.
// Why needed:
//   Ensures Eigen comparison includes numerical correctness, not only speed.
AccuracyResult compute_eigen_accuracy(const LUProblem& problem, const std::vector<double>& x_eigen, std::size_t trial, NormType norm, const std::string& status);

// Runs all Eigen benchmark phases for one benchmark case.
// What it does:
//   For one n, matrix type, and trial, runs Eigen factorization timing,
//   solve timing, full-solve timing, accuracy checks, and memory checkpoints.
// Why needed:
//   Keeps Eigen-specific benchmark logic out of bench.cpp.
void run_eigen_case(const BenchmarkConfig& config, std::size_t n, MatrixType matrix_type, std::size_t trial, std::vector<TimingResult>& timing_results,
 std::vector<AccuracyResult>& accuracy_results, std::vector<MemoryCheckpoint>& memory_results);

// Runs the full Eigen benchmark suite.
// What it does:
//   Loops over all configured sizes, matrix types, and trials, then calls
//   run_eigen_case for each case.
// Why needed:
//   Provides the top-level Eigen benchmark entry point called from bench.cpp.
void run_eigen_benchmarks(const BenchmarkConfig& config, std::vector<TimingResult>& timing_results, std::vector<AccuracyResult>& accuracy_results,
 std::vector<MemoryCheckpoint>& memory_results);


 // Converts a column-major multiple-RHS block into an Eigen dense matrix.
// What it does:
//   Takes B or X stored as block[i + rhs*n] and copies it into Eigen matrix
//   entry M(i, rhs).
// Why needed:
//   Eigen can solve A*X = B directly when B is an n-by-nrhs matrix.

Eigen::MatrixXd to_eigen_rhs_matrix(const std::vector<double>& X, std::size_t n, std::size_t nrhs);


// Converts an Eigen dense RHS/solution matrix back into column-major std::vector.
// What it does:
//   Copies Eigen matrix M(i, rhs) into vector storage X[i + rhs*n].
// Why needed:
//   Your shared residual and solution-error helpers use std::vector<double>.

std::vector<double> from_eigen_rhs_matrix(const Eigen::MatrixXd& X);

// Benchmarks Eigen solving many RHS after one factorization.
// What it does:
//   Builds X_true and B = A*X_true, factorizes A once with Eigen PartialPivLU,
//   then times lu.solve(B) for all RHS columns at once.
// Why needed:
//   Provides the Eigen baseline for multiple-RHS solve reuse.

MultipleRHSResult benchmark_eigen_multiple_rhs_once(const LUProblem& problem, std::size_t nrhs, std::size_t trial,
 NormType norm, std::uint32_t seed, const BenchmarkConfig& config);


// Runs one Eigen multiple-RHS benchmark case.
// What it does:
//   Creates one LUProblem, loops over configured RHS counts, and appends Eigen
//   multiple-RHS benchmark results.
// Why needed:
//   Keeps Eigen multiple-RHS benchmark orchestration out of bench.cpp.

void run_eigen_multiple_rhs_case(const BenchmarkConfig& config, std::size_t n, MatrixType matrix_type, std::size_t trial,
 std::vector<MultipleRHSResult>& multiple_rhs_results, std::vector<MemoryCheckpoint>& memory_results);


 // Runs all Eigen multiple-RHS benchmarks.
// What it does:
//   Loops over config.sizes, config.matrix_types, config.trials, and
//   config.rhs_counts.
// Why needed:
//   Provides the top-level Eigen multiple-RHS benchmark entry point.
void run_eigen_multiple_rhs_benchmarks(const BenchmarkConfig& config, std::vector<MultipleRHSResult>& multiple_rhs_results, 
 std::vector<MemoryCheckpoint>& memory_results);