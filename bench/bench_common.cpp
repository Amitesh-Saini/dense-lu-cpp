// bench_common.cpp
// Purpose:
//   Implement shared benchmark utilities used by all LU benchmark drivers.
//
// Implementation notes:
//   - Converts benchmark enums into CSV-friendly strings.
//   - Provides timing helpers based on std::chrono::steady_clock.
//   - Measures current process memory on macOS, Linux, and Windows.
//   - Estimates theoretical FLOP counts for LU factorization and triangular solves.

#include "bench_common.hpp"

#include <fstream>
#include <stdexcept>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif


// Enum-to-string conversion helpers

std::string to_string(MatrixType type) {
    switch (type) {
        case MatrixType::RandomDense:
            return "random_dense";

        case MatrixType::DiagonallyDominant:
            return "diagonally_dominant";

        case MatrixType::Hilbert:
            return "hilbert";

        case MatrixType::PivotStress:
            return "pivot_stress";

        case MatrixType::NearSingular:
            return "near_singular";

        case MatrixType::Singular:
            return "singular";
    }

    return "unknown_matrix_type";
}



std::string to_string(Implementation implementation) {
    switch (implementation) {
        case Implementation::CustomLU:
            return "custom_lu";

        case Implementation::EigenPartialPivLU:
            return "eigen_partial_piv_lu";
    }

    return "unknown_implementation";
}




std::string to_string(BenchmarkPhase phase) {
    switch (phase) {
        case BenchmarkPhase::Factorization:
            return "factorization";

        case BenchmarkPhase::Solve:
            return "solve";

        case BenchmarkPhase::FullSolve:
            return "full_solve";

        case BenchmarkPhase::ResidualFactorizationDebug:
            return "residual_factorization_debug";

        case BenchmarkPhase::ResidualFactorizationFast:
            return "residual_factorization_fast";

        case BenchmarkPhase::ResidualSolve:
            return "residual_solve";

        case BenchmarkPhase::MultipleRHS:
            return "multiple_rhs";

        case BenchmarkPhase::MultipleRHSResidual:
            return "multiple_rhs_residual";
    }

    return "unknown_benchmark_phase";
}

std::string to_string(NormType norm) {
    switch (norm) {
        case NormType::Infinity:
            return "inf_norm";

        case NormType::Frobenius:
            return "frobenius_norm";
    }

    return "unknown_norm_type";
}


std::string to_string(LU_factorize_Status status) {
    switch (status) {
        case LU_factorize_Status::Success:
            return "success";

        case LU_factorize_Status::PivotFailure:
            return "pivot_failure";

        case LU_factorize_Status::EliminationFailure:
            return "elimination_failure";

        case LU_factorize_Status::GeneralFailure:
            return "general_failure";
    }

    return "unknown_factorization_status";
}



std::string to_string(LU_Solve_Status status) {
    switch (status) {
        case LU_Solve_Status::Success:
            return "success";

        case LU_Solve_Status::ForwardSubstitutionFailure:
            return "forward_substitution_failure";

        case LU_Solve_Status::BackwardSubstitutionFailure:
            return "backward_substitution_failure";

        case LU_Solve_Status::GeneralFailure:
            return "general_failure";
    }

    return "unknown_solve_status";
}


// Timing helpers


std::chrono::steady_clock::time_point benchmark_now(){

    return std::chrono::steady_clock::now();
}



double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end){

    std::chrono::duration<double, std::milli> elapsed = end - start;

    return elapsed.count();
}




// Memory helpers


std::size_t current_process_memory_bytes()
{
#if defined(__APPLE__)

    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;

    kern_return_t result = task_info(
        mach_task_self(),
        TASK_VM_INFO,
        reinterpret_cast<task_info_t>(&info),
        &count
    );

    if (result != KERN_SUCCESS) {
        return 0;
    }

    return static_cast<std::size_t>(info.phys_footprint);

#elif defined(__linux__)

    std::ifstream statm("/proc/self/statm");

    if (!statm.is_open()) {
        return 0;
    }

    long total_pages = 0;
    long resident_pages = 0;

    statm >> total_pages >> resident_pages;

    if (resident_pages <= 0) {
        return 0;
    }

    const long page_size = sysconf(_SC_PAGESIZE);

    if (page_size <= 0) {
        return 0;
    }

    return static_cast<std::size_t>(resident_pages) *
           static_cast<std::size_t>(page_size);

#elif defined(_WIN32)

    PROCESS_MEMORY_COUNTERS counters;

    BOOL success = GetProcessMemoryInfo(
        GetCurrentProcess(),
        &counters,
        sizeof(counters)
    );

    if (!success) {
        return 0;
    }

    return static_cast<std::size_t>(counters.WorkingSetSize);

#else

    return 0;

#endif
}


void record_memory_checkpoint(
    std::vector<MemoryCheckpoint>& checkpoints, Implementation implementation, BenchmarkPhase phase, MatrixType matrix_type, 
    std::size_t n, std::size_t trial, const std::string& checkpoint){

    MemoryCheckpoint row;
    row.implementation = implementation;
    row.phase = phase;
    row.matrix_type = matrix_type;
    row.n = n;
    row.trial = trial;
    row.checkpoint = checkpoint;
    row.memory_bytes = current_process_memory_bytes();

    checkpoints.push_back(row);
}



std::size_t peak_observed_memory(const std::vector<MemoryCheckpoint>& checkpoints){

    std::size_t max = 0;

    for(std::size_t i = 0; i < checkpoints.size(); ++i){

        if(checkpoints[i].memory_bytes > max) max = checkpoints[i].memory_bytes;
    }

    return max;
}





// Flop helpers


double estimate_lu_factorization_flops(std::size_t n){

    const double nd = static_cast<double>(n);

    return ((2.0/3.0) * nd * nd * nd);
}



double estimate_lu_solve_flops(std::size_t n){

    const double nd = static_cast<double>(n);

    return (2 * nd * nd);
}



double compute_gflops(double flops, double time_ms){

    if(time_ms <= 0.0) return 0.0;

    const double seconds = time_ms / 1000.0;

    return (flops / seconds / 1.0e9); 
}



std::size_t checked_repetitions(std::size_t reps)
{
    if (reps == 0) {
        throw std::invalid_argument("benchmark repetitions must be greater than 0");
    }

    return reps;
}


std::size_t repetitions_for_phase(const BenchmarkConfig& config,
                                  BenchmarkPhase phase)
{
    switch (phase) {
        case BenchmarkPhase::Factorization:
            return config.factorization_repetitions;

        case BenchmarkPhase::Solve:
            return config.solve_repetitions;

        case BenchmarkPhase::FullSolve:
            return config.full_solve_repetitions;

        case BenchmarkPhase::ResidualFactorizationDebug:
        case BenchmarkPhase::ResidualFactorizationFast:
        case BenchmarkPhase::ResidualSolve:
        case BenchmarkPhase::MultipleRHSResidual:
            return config.residual_repetitions;

        case BenchmarkPhase::MultipleRHS:
            return config.multiple_rhs_repetitions;
    }

    return 1;
}