#include "bench_csv.hpp"

#include "bench_common.hpp"

#include <fstream>
#include <stdexcept>


void write_timing_csv(const std::string& filename, const std::vector<TimingResult>& results)
{
    std::ofstream out(filename);

    if (!out.is_open()) {
        throw std::runtime_error("write_timing_csv: failed to open file: " + filename);
    }

    out << "implementation,"
        << "phase,"
        << "matrix_type,"
        << "n,"
        << "trial,"
        << "time_ms,"
        << "estimated_gflops,"
        << "status\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const TimingResult& result = results[i];

        out << to_string(result.implementation) << ','
            << to_string(result.phase) << ','
            << to_string(result.matrix_type) << ','
            << result.n << ','
            << result.trial << ','
            << result.time_ms << ','
            << result.gflops << ','
            << result.status << '\n';
    }
}

void write_accuracy_csv(const std::string& filename, const std::vector<AccuracyResult>& results)
{
    std::ofstream out(filename);

    if (!out.is_open()) {
        throw std::runtime_error("write_accuracy_csv: failed to open file: " + filename);
    }

    out << "implementation,"
        << "matrix_type,"
        << "n,"
        << "trial,"
        << "factorization_residual,"
        << "solve_residual,"
        << "solution_error_inf,"
        << "solution_error_l2,"
        << "status\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const AccuracyResult& result = results[i];

        out << to_string(result.implementation) << ','
            << to_string(result.matrix_type) << ','
            << result.n << ','
            << result.trial << ','
            << result.factorization_residual << ','
            << result.solve_residual << ','
            << result.solution_error_inf << ','
            << result.solution_error_l2 << ','
            << result.status << '\n';
    }
}

void write_memory_csv(const std::string& filename, const std::vector<MemoryCheckpoint>& results)
{
    std::ofstream out(filename);

    if (!out.is_open()) {
        throw std::runtime_error("write_memory_csv: failed to open file: " + filename);
    }

    out << "implementation,"
        << "phase,"
        << "matrix_type,"
        << "n,"
        << "trial,"
        << "checkpoint,"
        << "memory_bytes,"
        << "memory_mb\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const MemoryCheckpoint& result = results[i];

        const double memory_mb =
            static_cast<double>(result.memory_bytes) / (1024.0 * 1024.0);

        out << to_string(result.implementation) << ','
            << to_string(result.phase) << ','
            << to_string(result.matrix_type) << ','
            << result.n << ','
            << result.trial << ','
            << result.checkpoint << ','
            << result.memory_bytes << ','
            << memory_mb << '\n';
    }
}

void write_residual_timing_csv(const std::string& filename,
                               const std::vector<ResidualTimingResult>& results)
{
    std::ofstream out(filename);

    if (!out.is_open()) {
        throw std::runtime_error("write_residual_timing_csv: failed to open file: " + filename);
    }

    out << "matrix_type,"
        << "phase,"
        << "norm,"
        << "n,"
        << "trial,"
        << "time_ms,"
        << "residual_value\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const ResidualTimingResult& result = results[i];

        out << to_string(result.matrix_type) << ','
            << to_string(result.phase) << ','
            << to_string(result.norm) << ','
            << result.n << ','
            << result.trial << ','
            << result.time_ms << ','
            << result.residual_value << '\n';
    }
}


void write_multiple_rhs_csv(const std::string& filename,
                            const std::vector<MultipleRHSResult>& results)
{
    std::ofstream out(filename);

    if (!out.is_open()) {
        throw std::runtime_error("write_multiple_rhs_csv: failed to open file: " + filename);
    }

    out << "implementation,"
        << "matrix_type,"
        << "n,"
        << "nrhs,"
        << "trial,"
        << "factorization_time_ms,"
        << "solve_kernel_time_ms,"
        << "solve_loop_time_ms,"
        << "total_time_ms,"
        << "time_per_rhs_ms,"
        << "estimated_gflops,"
        << "solve_residual_max,"
        << "solve_residual_mean,"
        << "solution_error_inf,"
        << "solution_error_l2,"
        << "status\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const MultipleRHSResult& result = results[i];

        out << to_string(result.implementation) << ','
            << to_string(result.matrix_type) << ','
            << result.n << ','
            << result.nrhs << ','
            << result.trial << ','
            << result.factorization_time_ms << ','
            << result.solve_kernel_time_ms << ','
            << result.solve_loop_time_ms << ','
            << result.total_time_ms << ','
            << result.time_per_rhs_ms << ','
            << result.estimated_gflops << ','
            << result.solve_residual_max << ','
            << result.solve_residual_mean << ','
            << result.solution_error_inf << ','
            << result.solution_error_l2 << ','
            << result.status << '\n';
    }
}


void write_multiple_rhs_residual_timing_csv(
    const std::string& filename,
    const std::vector<MultipleRHSResidualTimingResult>& results)
{
    std::ofstream out(filename);

    if (!out.is_open()) {
        throw std::runtime_error("write_multiple_rhs_residual_timing_csv: failed to open file: " + filename);
    }

    out << "matrix_type,"
        << "norm,"
        << "n,"
        << "nrhs,"
        << "trial,"
        << "time_ms,"
        << "residual_max,"
        << "residual_mean,"
        << "status\n";

    for (std::size_t i = 0; i < results.size(); ++i) {
        const MultipleRHSResidualTimingResult& result = results[i];

        out << to_string(result.matrix_type) << ','
            << to_string(result.norm) << ','
            << result.n << ','
            << result.nrhs << ','
            << result.trial << ','
            << result.time_ms << ','
            << result.residual_max << ','
            << result.residual_mean << ','
            << result.status << '\n';
    }
}