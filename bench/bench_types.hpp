// bench_types.hpp
// Purpose:
//   Define the shared benchmark enums, configuration structs, and result structs.
//
// Contents:
//   - Matrix and implementation labels.
//   - Benchmark phase labels.
//   - Benchmark configuration options.
//   - Timing, accuracy, memory, residual, and multiple-RHS result records.

#pragma once

#include "verify.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>


// Benchmark enums


// Describes which matrix family is used in a benchmark case.
// What it does:
//   Labels the type of matrix used: random dense, diagonally dominant, Hilbert,
//   pivot-stress, near-singular, or singular.
// Why needed:
//   Lets benchmark_lu.cpp request a matrix type without hardcoding the exact
//   matrix-generation function everywhere.
enum class MatrixType {
    RandomDense, DiagonallyDominant,
    Hilbert, PivotStress,
    NearSingular, Singular
};

// Describes which implementation is being benchmarked.
// What it does:
//   Labels whether a result came from your custom LU implementation or from
//   Eigen's PartialPivLU baseline.
// Why needed:
//   Lets the CSV clearly distinguish custom_lu rows from eigen rows.
enum class Implementation {
    CustomLU, EigenPartialPivLU
};

// Describes the benchmark phase being measured.
// What it does:
//   Labels whether the benchmark timed factorization, solve, full solve,
//   residual checking, or multiple-RHS solves.
// Why needed:
//   Prevents mixing different operations in the same timing category.
enum class BenchmarkPhase {
    Factorization, Solve,
    FullSolve, ResidualFactorizationDebug,
    ResidualFactorizationFast,
    ResidualSolve, MultipleRHS, 
    MultipleRHSResidual
};


// Benchmark data structures



// Stores one multiple-RHS benchmark result.
// What it does:
//   Records performance and accuracy when solving A*X = B for many RHS columns.
// Why needed:
//   Multiple RHS benchmarks need extra fields like nrhs, factorization time,
//   total solve time, and time per RHS that do not fit cleanly into TimingResult.

struct MultipleRHSResult {Implementation implementation; MatrixType matrix_type; std::size_t n; std::size_t nrhs; std::size_t trial;

    double factorization_time_ms;
    double solve_kernel_time_ms;
    double solve_loop_time_ms;
    double total_time_ms;
    double time_per_rhs_ms;

    double estimated_gflops;

    double solve_residual_max;
    double solve_residual_mean;
    double solution_error_inf;
    double solution_error_l2;

    std::string status;
};


// Stores global benchmark settings.
// What it does:
//   Holds sizes, matrix types, number of trials, tolerance, norm choice,
//   random seed, and output directory.
// Why needed:
//   Keeps benchmark_lu.cpp clean by passing one config object instead of many
//   separate arguments.

struct BenchmarkConfig {
    std::vector<std::size_t> sizes;
    std::vector<MatrixType> matrix_types;

    std::size_t trials = 1;

    // Timing repetitions inside each trial.
    // These stabilize small-n timings without increasing CSV row count.
    std::size_t factorization_repetitions = 1;
    std::size_t solve_repetitions = 1;
    std::size_t full_solve_repetitions = 1;
    std::size_t residual_repetitions = 1;
    std::size_t multiple_rhs_repetitions = 1;

    double eps = 1e-12;
    NormType norm = NormType::Infinity;
    std::uint32_t base_seed = 123456789;

    std::vector<std::size_t> rhs_counts;

    std::string output_dir;
};

// Stores one generated linear system A*x = b.
// What it does:
//   Holds the original matrix A0, known exact solution x_true, right-hand side b,
//   pivot vector, matrix size, tolerance, matrix type, and random seed.
// Why needed:
//   Ensures factorization, solve, full-solve, residual, and Eigen benchmarks all
//   start from a consistent problem setup.

struct LUProblem {
    std::vector<double> A0; std::vector<double> x_true;
    std::vector<double> b; std::vector<std::size_t> piv;
    std::size_t n; double eps; MatrixType matrix_type;
    std::uint32_t seed;
};

// Stores one timing result.
// What it does:
//   Records implementation, phase, matrix type, size, trial number, elapsed time,
//   estimated GFLOP/s, and status.
// Why needed:
//   Provides one clean row for timing CSV output.

struct TimingResult {
    Implementation implementation; BenchmarkPhase phase;
    MatrixType matrix_type; std::size_t n; std::size_t trial;
    double time_ms; double gflops; std::string status;
};

// Stores timing results for multiple-RHS residual checking.
// What it does:
//   Records how long it takes to verify A*X_computed = B for many RHS columns.
// Why needed:
//   Multiple-RHS residual checking has different scaling because it depends on
//   both n and nrhs, so it needs its own result type.

struct MultipleRHSResidualTimingResult {
    MatrixType matrix_type;
    NormType norm;
    std::size_t n;
    std::size_t nrhs;
    std::size_t trial;

    double time_ms;
    double residual_max;
    double residual_mean;

    std::string status;
};

// Stores one accuracy result.
// What it does:
//   Records factorization residual, solve residual, solution errors, and status.
// Why needed:
//   Keeps numerical correctness data separate from timing data.

struct AccuracyResult {
    Implementation implementation; MatrixType matrix_type;
    std::size_t n; std::size_t trial;
    double factorization_residual; double solve_residual;
    double solution_error_inf; double solution_error_l2;
    std::string status;
};

// Stores one memory measurement checkpoint.
// What it does:
//   Records actual process memory at a specific checkpoint such as before setup,
//   after allocation, after factorization, or after solve.
// Why needed:
//   Lets you track real process-level memory behavior without mixing memory
//   measurement into timing results.

struct MemoryCheckpoint {
    Implementation implementation; BenchmarkPhase phase;
    MatrixType matrix_type; std::size_t n;
    std::size_t trial; std::string checkpoint;
    std::size_t memory_bytes;
};

// Stores one residual-checker timing result.
// What it does:
//   Records timing and value for debug factorization residual, fast
//   factorization residual, or solve residual.
// Why needed:
//   Lets you compare verification overhead separately from LU performance.

struct ResidualTimingResult {
    MatrixType matrix_type; BenchmarkPhase phase;
    NormType norm; std::size_t n;
    std::size_t trial; double time_ms;
    double residual_value;
};
