#pragma once

#include "bench_types.hpp"

#include <string>
#include <vector>

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


// Writes multiple-RHS benchmark results to CSV.
// What it does:
//   Writes one row per MultipleRHSResult.
// Why needed:
//   Stores multiple-RHS timing and accuracy data separately from ordinary
//   factorization/solve/full-solve benchmark CSVs.

void write_multiple_rhs_csv(const std::string& filename, const std::vector<MultipleRHSResult>& results);

// Writes multiple-RHS residual timing results to CSV.
// What it does:
//   Writes one row per MultipleRHSResidualTimingResult.
// Why needed:
//   Stores verification-overhead data for plotting residual-check cost vs n/nrhs.
void write_multiple_rhs_residual_timing_csv(
    const std::string& filename,
    const std::vector<MultipleRHSResidualTimingResult>& results);