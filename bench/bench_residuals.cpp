// bench_residuals.cpp
// Purpose:
//   Implement residual-checker benchmark routines.
//
// Implementation notes:
//   - Measures the cost of verification routines separately from LU solving.
//   - Compares debug and fast factorization residual paths.
//   - Times solve residual computation after completed solves.
//   - Benchmarks multiple-RHS residual computation for batched solve verification.

#include "bench_residuals.hpp"
#include "bench_common.hpp"
#include "bench_generators.hpp"
#include "verify.hpp"
#include "lu.hpp"

#include <limits>
#include <stdexcept>
#include <vector>


ResidualTimingResult benchmark_factorization_residual_debug_once(
    const LUProblem& problem, const std::vector<double>& LU, const std::vector<std::size_t>& piv, std::size_t trial, NormType norm, 
    const BenchmarkConfig& config){


    const std::size_t reps = checked_repetitions(repetitions_for_phase(config, BenchmarkPhase::ResidualFactorizationDebug));

    double residual_value = 0.0;
    double total_time_ms = 0.0;

    for (std::size_t rep = 0; rep < reps; ++rep) {
        std::vector<double> LU_multiplied(problem.n * problem.n);
        std::vector<double> PA(problem.n * problem.n);
        std::vector<double> diff(problem.n * problem.n);

        const auto start = benchmark_now();

        residual_value = residual_factorization_debug(problem.A0, LU_multiplied, piv, PA, LU, diff, problem.n, norm);

        const auto end = benchmark_now();

        total_time_ms += elapsed_ms(start, end);
    }

    const double time_ms = total_time_ms / static_cast<double>(reps);

    ResidualTimingResult result;

    result.matrix_type = problem.matrix_type;
    result.n = problem.n;
    result.norm = norm;
    result.phase = BenchmarkPhase::ResidualFactorizationDebug;
    result.time_ms = time_ms;
    result.trial = trial;
    result.residual_value = residual_value;

    return result;
}


ResidualTimingResult benchmark_factorization_residual_fast_once(
    const LUProblem& problem, const std::vector<double>& LU, const std::vector<std::size_t>& piv, std::size_t trial, NormType norm, 
    const BenchmarkConfig& config){


    const std::size_t reps = checked_repetitions(repetitions_for_phase(config, BenchmarkPhase::ResidualFactorizationFast));

    double residual_value = 0.0;
    double total_time_ms = 0.0;

    for (std::size_t rep = 0; rep < reps; ++rep) {
        const auto start = benchmark_now();

        residual_value = residual_factorization_fast(problem.A0, LU, piv, problem.n, norm);

        const auto end = benchmark_now();

        total_time_ms += elapsed_ms(start, end);
    }

    const double time_ms = total_time_ms / static_cast<double>(reps);

    ResidualTimingResult result;

    result.matrix_type = problem.matrix_type;
    result.n = problem.n;
    result.norm = norm;
    result.phase = BenchmarkPhase::ResidualFactorizationFast;
    result.time_ms = time_ms;
    result.trial = trial;
    result.residual_value = residual_value;

    return result;
}


ResidualTimingResult benchmark_solve_residual_once(
    const LUProblem& problem, const std::vector<double>& x, std::size_t trial, NormType norm, const BenchmarkConfig& config){


    const std::size_t reps = checked_repetitions(repetitions_for_phase(config, BenchmarkPhase::ResidualSolve));

    double residual_value = 0.0;
    double total_time_ms = 0.0;

    for (std::size_t rep = 0; rep < reps; ++rep) {
        const auto start = benchmark_now();

        residual_value = residual_solve(problem.A0, x, problem.b, problem.n, norm);

        const auto end = benchmark_now();

        total_time_ms += elapsed_ms(start, end);
    }

    const double time_ms = total_time_ms / static_cast<double>(reps);

    ResidualTimingResult result;

    result.matrix_type = problem.matrix_type;
    result.n = problem.n;
    result.norm = norm;
    result.phase = BenchmarkPhase::ResidualSolve;
    result.time_ms = time_ms;
    result.trial = trial;
    result.residual_value = residual_value;

    return result;
}


void run_residual_benchmark_case(
    const BenchmarkConfig& config, std::size_t n, MatrixType matrix_type, std::size_t trial, std::vector<ResidualTimingResult>& residual_results, 
    std::vector<MemoryCheckpoint>& memory_results){


    std::uint32_t seed = config.base_seed + static_cast<std::uint32_t>(trial) + static_cast<std::uint32_t>(1000 * n)
    + static_cast<std::uint32_t>(100000 * static_cast<int>(matrix_type));

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::ResidualFactorizationDebug, 
    matrix_type, n, trial, "before_residual_problem_setup");

    auto problem = make_lu_problem(n, matrix_type, seed, config.eps);

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::ResidualFactorizationDebug, 
    matrix_type, n, trial, "after_residual_problem_setup");

    std::vector<std::size_t> piv = problem.piv;
    std::vector<double> LU = problem.A0; 

    LU_factorize_Status factor_status = LU_factorize(LU, piv, problem.n, problem.eps);

    if(factor_status != LU_factorize_Status::Success) return;

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::ResidualFactorizationDebug, 
    matrix_type, n, trial, "after_residual_factorization_setup");

    residual_results.push_back(benchmark_factorization_residual_debug_once(problem, LU, piv, trial, config.norm, config));

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::ResidualFactorizationDebug, 
    matrix_type, n, trial, "after_debug_factorization_residual_benchmark");

    residual_results.push_back(benchmark_factorization_residual_fast_once(problem, LU, piv, trial, config.norm, config));

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::ResidualFactorizationFast, 
    matrix_type, n, trial, "after_fast_factorization_residual_benchmark");

    std::vector<double> x_computed(problem.n);

    LU_Solve_Status solve_status = LU_solve(LU, piv, problem.b, x_computed, problem.n, problem.eps);

    if(solve_status != LU_Solve_Status::Success) return;

    residual_results.push_back(benchmark_solve_residual_once(problem, x_computed, trial, config.norm, config));

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::ResidualSolve, 
    matrix_type, n, trial, "after_solve_residual_benchmark");
 }


void run_residual_benchmarks(const BenchmarkConfig& config, std::vector<ResidualTimingResult>& residual_results, std::vector<MemoryCheckpoint>& memory_results){

    for (const std::size_t n : config.sizes) {

        for (const MatrixType matrix_type : config.matrix_types) {

            for (std::size_t trial = 0; trial < config.trials; ++trial) {

                run_residual_benchmark_case(config, n, matrix_type, trial, residual_results, memory_results);
            }
        }
    }
 }





MultipleRHSResidualTimingResult benchmark_multiple_rhs_residual_once(
    const LUProblem& problem, std::size_t nrhs, std::size_t trial, NormType norm, std::uint32_t seed, const BenchmarkConfig& config){


    if (nrhs == 0) {
        throw std::invalid_argument("benchmark_multiple_rhs_residual_once: nrhs must be greater than 0");
    }

    if (problem.n == 0) {
        throw std::invalid_argument("benchmark_multiple_rhs_residual_once: n must be greater than 0");
    }

    const std::size_t reps = checked_repetitions(
        repetitions_for_phase(config, BenchmarkPhase::MultipleRHSResidual)
    );

    // ------------------------------------------------------------
    // 1. Build exact solution and RHS matrix once.
    // ------------------------------------------------------------

    std::vector<double> X_true =
        make_multiple_x_true(problem.n, nrhs, -1.0, 1.0, seed);

    std::vector<double> B(problem.n * nrhs);
    make_multiple_rhs(problem.A0, X_true, B, problem.n, nrhs);

    // ------------------------------------------------------------
    // 2. Factor once.
    // ------------------------------------------------------------

    std::vector<double> LU = problem.A0;
    std::vector<std::size_t> piv = problem.piv;

    LU_factorize_Status factor_status =
        LU_factorize(LU, piv, problem.n, problem.eps);

    if (factor_status != LU_factorize_Status::Success) {
        MultipleRHSResidualTimingResult result;

        result.matrix_type = problem.matrix_type;
        result.n = problem.n;
        result.norm = norm;
        result.nrhs = nrhs;
        result.residual_max = std::numeric_limits<double>::quiet_NaN();
        result.residual_mean = std::numeric_limits<double>::quiet_NaN();
        result.status = "factorization_" + to_string(factor_status);
        result.time_ms = std::numeric_limits<double>::quiet_NaN();
        result.trial = trial;

        return result;
    }

    // ------------------------------------------------------------
    // 3. Solve all RHS once to build X_computed.
    // ------------------------------------------------------------

    std::vector<double> X_computed(problem.n * nrhs);
    std::vector<double> x_i_computed(problem.n);
    std::vector<double> b_i(problem.n);

    for (std::size_t i = 0; i < nrhs; ++i) {
        b_i = extract_rhs_column(B, problem.n, nrhs, i);

        LU_Solve_Status solve_status =
            LU_solve(LU, piv, b_i, x_i_computed, problem.n, problem.eps);

        if (solve_status != LU_Solve_Status::Success) {
            MultipleRHSResidualTimingResult result;

            result.matrix_type = problem.matrix_type;
            result.n = problem.n;
            result.norm = norm;
            result.nrhs = nrhs;
            result.residual_max = std::numeric_limits<double>::quiet_NaN();
            result.residual_mean = std::numeric_limits<double>::quiet_NaN();
            result.status = "solve_" + to_string(solve_status) + "_rhs_" + std::to_string(i);
            result.time_ms = std::numeric_limits<double>::quiet_NaN();
            result.trial = trial;

            return result;
        }

        store_solution_column(X_computed, x_i_computed, problem.n, nrhs, i);
    }

    // ------------------------------------------------------------
    // 4. Benchmark only the multiple-RHS residual checker.
    // ------------------------------------------------------------

    std::vector<double> residual_values;
    double total_residual_time_ms = 0.0;

    for (std::size_t rep = 0; rep < reps; ++rep) {
        const auto start_residual_time = benchmark_now();

        residual_values =
            multiple_rhs_solve_residuals(problem.A0, X_computed, B, problem.n, nrhs, norm);

        const auto end_residual_time = benchmark_now();

        total_residual_time_ms += elapsed_ms(start_residual_time, end_residual_time);
    }

    const double multiple_rhs_residual_ms =
        total_residual_time_ms / static_cast<double>(reps);

    if (residual_values.size() != nrhs) {
        throw std::runtime_error("benchmark_multiple_rhs_residual_once: residual size mismatch");
    }

    // ------------------------------------------------------------
    // 5. Summarize residual values.
    // ------------------------------------------------------------

    double max_residual = 0.0;
    double residual_sum = 0.0;

    for (std::size_t j = 0; j < nrhs; ++j) {
        if (residual_values[j] > max_residual) {
            max_residual = residual_values[j];
        }

        residual_sum += residual_values[j];
    }

    MultipleRHSResidualTimingResult result;

    result.matrix_type = problem.matrix_type;
    result.n = problem.n;
    result.norm = norm;
    result.nrhs = nrhs;
    result.residual_max = max_residual;
    result.residual_mean = residual_sum / static_cast<double>(nrhs);
    result.status = "success";
    result.time_ms = multiple_rhs_residual_ms;
    result.trial = trial;

    return result;
}



void run_multiple_rhs_residual_case(
    const BenchmarkConfig& config, std::size_t n, MatrixType matrix_type, std::size_t trial, std::vector<MultipleRHSResidualTimingResult>& residual_results, 
    std::vector<MemoryCheckpoint>& memory_results){


    if (n == 0) {
        throw std::invalid_argument("run_multiple_rhs_residual_case: n must be greater than 0");
    }

    std::uint32_t seed = config.base_seed + static_cast<std::uint32_t>(trial) + static_cast<std::uint32_t>(1000 * n)
    + static_cast<std::uint32_t>(100000 * static_cast<int>(matrix_type));

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::MultipleRHSResidual, 
    matrix_type, n, trial, "before_problem_setup");

    auto problem = make_lu_problem(n, matrix_type, seed, config.eps);


    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::MultipleRHSResidual, 
    matrix_type, n, trial, "after_problem_setup");


    for(std::size_t i = 0; i < config.rhs_counts.size(); ++i){

        const std::size_t nrhs = config.rhs_counts[i];

        if (nrhs == 0) {
            throw std::invalid_argument("run_custom_multiple_rhs_case: nrhs must be greater than 0");
            }
        
        record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::MultipleRHSResidual, 
         matrix_type, n, trial, "before_multiples_rhs_residual_benchmarks");

        MultipleRHSResidualTimingResult result = benchmark_multiple_rhs_residual_once(problem, nrhs, trial, config.norm, seed, config);
        residual_results.push_back(result);

        record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::MultipleRHSResidual, 
         matrix_type, n, trial, "after_multiples_rhs_residual_benchmarks");
        
    }
 }


void run_multiple_rhs_residual_benchmarks(
    const BenchmarkConfig& config, std::vector<MultipleRHSResidualTimingResult>& residual_results, std::vector<MemoryCheckpoint>& memory_results){

    for (const std::size_t n : config.sizes) {

        for (const MatrixType matrix_type : config.matrix_types) {

            for (std::size_t trial = 0; trial < config.trials; ++trial) {

                run_multiple_rhs_residual_case(config, n, matrix_type, trial, residual_results, memory_results);
            }
        }
    }
 }
