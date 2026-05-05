#pragma once

#include "lu.hpp"
#include "verify.hpp"

#include <chrono>
#include <cstddef>
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
    ResidualSolve, MultipleRHS
};


// Benchmark data structures


// Stores global benchmark settings.
// What it does:
//   Holds sizes, matrix types, number of trials, tolerance, norm choice,
//   random seed, and output directory.
// Why needed:
//   Keeps benchmark_lu.cpp clean by passing one config object instead of many
//   separate arguments.

struct BenchmarkConfig {
    std::vector<std::size_t> sizes; std::vector<MatrixType> matrix_types;
    std::size_t trials; double eps;
    NormType norm; std::uint32_t base_seed;
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


// String conversion helpers


// Converts MatrixType to a CSV-friendly string.
// What it does:
//   Converts enum values like MatrixType::RandomDense into labels like
//   "random_dense".
// Why needed:
//   CSV files should contain readable names instead of integer enum values.

std::string to_string(MatrixType type);



// Converts Implementation to a CSV-friendly string.
// What it does:
//   Converts enum values like Implementation::CustomLU into labels like
//   "custom_lu".
// Why needed:
//   CSV rows need to identify which implementation produced the result.

std::string to_string(Implementation implementation);



// Converts BenchmarkPhase to a CSV-friendly string.
// What it does:
//   Converts enum values like BenchmarkPhase::Factorization into labels like
//   "factorization".
// Why needed:
//   CSV rows need to identify exactly what operation was timed.

std::string to_string(BenchmarkPhase phase);



// Converts NormType to a CSV-friendly string.
// What it does:
//   Converts the norm enum from your verification code into labels like
//   "infinity" or "frobenius".
// Why needed:
//   Residual benchmark CSVs should record which norm was used.

std::string to_string(NormType norm);



// Converts LU_factorize_Status to a CSV-friendly string.
// What it does:
//   Converts your factorization status enum into labels like "success",
//   "pivot_failure", or "elimination_failure".
// Why needed:
//   Benchmark CSVs should record whether factorization succeeded or failed.

std::string to_string(LU_factorize_Status status);



// Converts LU_Solve_Status to a CSV-friendly string.
// What it does:
//   Converts your solve status enum into labels like "success",
//   "forward_substitution_failure", or "backward_substitution_failure".
// Why needed:
//   Benchmark CSVs should record whether the solve phase succeeded or failed.

std::string to_string(LU_Solve_Status status);




// Problem setup helpers


// Creates a matrix that forces partial pivoting.
// What it does:
//   Builds a simple matrix where the first pivot is tiny/invalid and a better
//   pivot exists below it.
// Why needed:
//   Tests that partial pivoting is actually being exercised.


std::vector<double> make_pivot_stress_matrix(std::size_t n);

// Creates a nearly singular matrix.
// What it does:
//   Builds a matrix with two nearly dependent rows.
// Why needed:
//   Stress-tests numerical tolerance and stability behavior.

std::vector<double> make_near_singular_matrix(std::size_t n);

// Creates a singular matrix.
// What it does:
//   Builds a matrix with two identical rows.
// Why needed:
//   Tests that LU factorization fails correctly on singular systems.

std::vector<double> make_singular_matrix(std::size_t n);


// creates random dense matrix using seed

std::vector<double> generate_seeded_random_dense_matrix(double lower_bound, double upper_bound, std::size_t n, std::uint32_t seed);


// creates random diagonally dominant dense matrix using seed

std::vector<double> generate_seeded_random_diagonally_dominant_matrix(double lower_bound, double upper_bound, std::size_t n,
 double margin, std::uint32_t seed);

// generate random seeded vector 

std::vector<double> generate_seeded_random_vector(double lower_bound, double upper_bound, std::size_t n, std::uint32_t seed);



// Creates a benchmark matrix of the requested type.
// What it does:
//   Dispatches to your existing matrix generators based on MatrixType.
//   For example, RandomDense should call your existing random dense generator,
//   DiagonallyDominant should call your existing diagonally dominant generator,
//   and Hilbert should call your existing Hilbert generator.
// Why needed:
//   Keeps bench_lu.cpp independent of the details of matrix generation.

std::vector<double> make_benchmark_matrix(std::size_t n, MatrixType type, std::uint32_t seed);



// Creates a benchmark vector.
// What it does:
//   Dispatches to your existing random vector generator.
// Why needed:
//   Used to create x_true before forming b = A*x_true.

std::vector<double> make_benchmark_vector(std::size_t n, double lower, double upper, std::uint32_t seed);



// Creates the identity pivot vector [0, 1, 2, ..., n-1].
// What it does:
//   Allocates and initializes the pivot vector before LU factorization.
// Why needed:
//   Your LU factorization expects the pivot vector to already exist and have
//   valid initial row indices.

std::vector<std::size_t> make_identity_pivot(std::size_t n);



// Creates a full LU benchmark problem.
// What it does:
//   Builds A0, x_true, b = A0*x_true, pivot vector, size, tolerance, matrix
//   label, and seed.
// Why needed:
//   All benchmark phases need the same clean problem setup, and this avoids
//   repeating setup code in every benchmark driver.


LUProblem make_lu_problem(std::size_t n, MatrixType matrix_type, std::uint32_t seed, double eps);


// Timing helpers


// Returns the current high-resolution timestamp.
// What it does:
//   Wraps std::chrono::high_resolution_clock::now().
// Why needed:
//   Gives all benchmark functions a consistent way to mark start/end times.


std::chrono::steady_clock::time_point benchmark_now();



// Converts two timestamps into elapsed milliseconds.
// What it does:
//   Computes end - start and returns the duration in milliseconds.
// Why needed:
//   Ensures every timing result uses the same unit.


double elapsed_ms(std::chrono::steady_clock::time_point start,
 std::chrono::steady_clock::time_point end);



// Times a callable operation and returns elapsed milliseconds.
// What it does:
//   Starts a timer, runs the operation, stops the timer, and returns elapsed ms.
// Why needed:
//   Avoids repeating timing boilerplate in every benchmark function.
//
// Note:
//   This is a template, so it must be defined in the header.


template <typename Func>

double time_operation(Func&& operation) {
    const auto start = benchmark_now();
    operation();
    const auto end = benchmark_now();
    return elapsed_ms(start, end);
}

// Returns a repeat count for small benchmark sizes.
// What it does:
//   Gives larger repeat counts for tiny n where one run is too fast to time
//   accurately.
// Why needed:
//   Reduces timer noise for n = 8, 16, 32, etc.


std::size_t repeat_count_for_size(std::size_t n, BenchmarkPhase phase);



// Actual process memory helpers


// Returns the current process memory footprint in bytes.
// What it does:
//   On macOS, this should use mach_task_self() + task_info + TASK_VM_INFO,
//   preferably using phys_footprint.
// Why needed:
//   Lets the benchmark report actual process-level memory behavior instead of
//   only theoretical storage formulas.


std::size_t current_process_memory_bytes();


// Records one memory checkpoint.
// What it does:
//   Calls current_process_memory_bytes() and appends a MemoryCheckpoint row.
// Why needed:
//   Lets benchmark code record memory at points like before_setup,
//   after_A0_alloc, after_LU_copy, after_factorization, after_solve,
//   and after_residual.


void record_memory_checkpoint(std::vector<MemoryCheckpoint>& checkpoints, Implementation implementation, BenchmarkPhase phase,
 MatrixType matrix_type, std::size_t n, std::size_t trial, const std::string& checkpoint);



// Finds the largest memory value in a checkpoint list.
// What it does:
//   Scans memory checkpoint rows and returns the largest memory_bytes value.
// Why needed:
//   Lets you report peak observed memory for a benchmark case.


std::size_t peak_observed_memory(const std::vector<MemoryCheckpoint>& checkpoints);




// FLOP and performance helpers


// Estimates the flop count for dense LU factorization.
// What it does:
//   Returns approximately (2/3)*n^3 floating-point operations.
// Why needed:
//   Lets you compute approximate GFLOP/s for factorization benchmarks.


double estimate_lu_factorization_flops(std::size_t n);



// Estimates the flop count for one LU solve.
// What it does:
//   Returns an approximate O(n^2) flop count for forward and backward
//   substitution.
// Why needed:
//   Lets you compute approximate solve throughput.


double estimate_lu_solve_flops(std::size_t n);



// Converts flop count and elapsed time into GFLOP/s.
// What it does:
//   Computes flops / seconds / 1e9.
// Why needed:
//   GFLOP/s is a standard performance metric for numerical/HPC benchmarks.


double compute_gflops(double flops, double time_ms);


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


TimingResult benchmark_custom_factorization_once(const LUProblem& problem, std::size_t trial);


// Benchmarks custom LU solve only.
// What it does:
//   Assumes LU and piv are already factorized outside the timed region, then
//   times only LU_solve.
// Why needed:
//   Measures forward/backward substitution cost separately from factorization.


TimingResult benchmark_custom_solve_once(const LUProblem& problem, const std::vector<double>& LU, const std::vector<std::size_t>& piv, std::size_t trial);



// Benchmarks custom full solve.
// What it does:
//   Times the full operation of factorization followed by solve.
// Why needed:
//   Measures the practical end-to-end cost of solving one dense system Ax = b


//   from scratch.


TimingResult benchmark_custom_full_solve_once(const LUProblem& problem, std::size_t trial);



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




// Residual checker benchmark operations


// Benchmarks debug factorization residual only.
// What it does:
//   Times residual_factorization_debug on an already-factorized LU.
// Why needed:
//   Measures the cost of the clear but allocation-heavy verification path.


ResidualTimingResult benchmark_factorization_residual_debug_once(const LUProblem& problem, const std::vector<double>& LU, const std::vector<std::size_t>& piv,
 std::size_t trial, NormType norm);



// Benchmarks fast factorization residual only.
// What it does:
//   Times residual_factorization_fast on an already-factorized LU.
// Why needed:
//   Measures the optimized residual checker and allows comparison against the
//   debug residual checker.


ResidualTimingResult benchmark_factorization_residual_fast_once(const LUProblem& problem, const std::vector<double>& LU,
 const std::vector<std::size_t>& piv, std::size_t trial, NormType norm);



// Benchmarks solve residual only.
// What it does:
//   Times residual_solve after a completed solve.
// Why needed:
//   Measures the O(n^2) solve verification cost separately from the O(n^3)
//   factorization verification cost.

ResidualTimingResult benchmark_solve_residual_once(const LUProblem& problem, const std::vector<double>& x,
 std::size_t trial, NormType norm);




// Runs one residual benchmark case.
// What it does:
//   Factorizes and solves a problem once, then times debug factorization
//   residual, fast factorization residual, and solve residual.
// Why needed:
//   Compares verification overheads under the same matrix and solution setup.


void run_residual_benchmark_case(const BenchmarkConfig& config, std::size_t n, MatrixType matrix_type,
 std::size_t trial, std::vector<ResidualTimingResult>& residual_results, 
 std::vector<MemoryCheckpoint>& memory_results);




// Runs the full residual benchmark suite.
// What it does:
//   Loops over configured sizes, matrix types, and trials for residual checker
//   timing.
// Why needed:
//   Provides the top-level residual benchmark entry point for benchmark_lu.cpp.
void run_residual_benchmarks(const BenchmarkConfig& config, std::vector<ResidualTimingResult>& residual_results, 
 std::vector<MemoryCheckpoint>& memory_results);





// CSV writing helpers

// Writes timing results to CSV.
// What it does:
//   Writes one row per TimingResult with implementation, phase, n, matrix type,
//   trial, time, GFLOP/s, and status.
// Why needed:
//   Keeps CSV formatting out of benchmark_lu.cpp.

void write_timing_csv(const std::string& filename, const std::vector<TimingResult>& results);



// Writes accuracy results to CSV.
// What it does:
//   Writes one row per AccuracyResult with residuals, solution errors, and status.
// Why needed:
//   Stores numerical correctness separately from performance timing.

void write_accuracy_csv(const std::string& filename, const std::vector<AccuracyResult>& results);



// Writes memory checkpoint results to CSV.
// What it does:
//   Writes one row per MemoryCheckpoint.
// Why needed:
//   Stores actual process memory measurements for later plotting in Python.
void write_memory_csv(const std::string& filename, const std::vector<MemoryCheckpoint>& results);



// Writes residual timing results to CSV.
// What it does:
//   Writes one row per ResidualTimingResult.
// Why needed:
//   Stores debug-vs-fast residual checker timings for later plotting.
void write_residual_timing_csv(const std::string& filename, const std::vector<ResidualTimingResult>& results);