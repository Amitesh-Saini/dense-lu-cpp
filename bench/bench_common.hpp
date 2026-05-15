// bench_common.hpp
// Purpose:
//   Declare shared benchmark helper functions used across the LU benchmark suite.
//
// Contents:
//   - Enum-to-string conversion helpers for CSV output.
//   - Timing helpers for benchmark measurements.
//   - Process-memory measurement helpers.
//   - FLOP estimates and GFLOP/s calculations.
//   - Repetition-count helpers for stable benchmark timing.

#pragma once

#include "bench_types.hpp"
#include "lu.hpp"

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

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


// Actual process memory helpers


// Returns the current process memory footprint in bytes.
// What it does:
//   On macOS, this should use mach_task_self() + task_info + TASK_VM_INFO,
//   preferably using phys_footprint.
// Why needed:
//   Lets the benchmark report actual process-level memory behavior instead of
//   only theoretical storage formulas.
// process-level memory footprint, not exact algorithm memory allocation.


std::size_t current_process_memory_bytes();


// Records one memory checkpoint.
// What it does:
//   Calls current_process_memory_bytes() and appends a MemoryCheckpoint row.
// Why needed:
//   Lets benchmark code record memory at points like before_setup,
//   after_A0_alloc, after_LU_copy, after_factorization, after_solve,
//   and after_residual.


void record_memory_checkpoint(
    std::vector<MemoryCheckpoint>& checkpoints, Implementation implementation, BenchmarkPhase phase, MatrixType matrix_type, 
    std::size_t n, std::size_t trial, const std::string& checkpoint);



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



// Returns the configured repetition count for the given benchmark phase.
std::size_t repetitions_for_phase(const BenchmarkConfig& config, BenchmarkPhase phase);

// Throws if repetitions == 0. Prevents divide-by-zero timing bugs.
std::size_t checked_repetitions(std::size_t reps);
