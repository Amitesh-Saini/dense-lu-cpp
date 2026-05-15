// bench_multiple_rhs.cpp
// Purpose:
//   Implement custom LU multiple-RHS benchmark routines.
//
// Implementation notes:
//   - Demonstrates LU reuse: factorize A once, then solve many RHS vectors.
//   - Separates factorization timing from RHS solve-loop timing.
//   - Computes residuals and solution errors after timing.
//   - Records timing, accuracy, and status fields for CSV output.

#include "bench_multiple_rhs.hpp"
#include "bench_common.hpp"
#include "bench_generators.hpp"
#include "verify.hpp"
#include "lu.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

MultipleRHSResult benchmark_custom_multiple_rhs_once(
    const LUProblem& problem, std::size_t nrhs, std::size_t trial, NormType norm, std::uint32_t seed, const BenchmarkConfig& config){


    if (nrhs == 0) {
        throw std::invalid_argument("benchmark_custom_multiple_rhs_once: nrhs must be greater than 0");
    }

    if (problem.n == 0) {
        throw std::invalid_argument("benchmark_custom_multiple_rhs_once: n must be greater than 0");
    }

    const std::size_t reps = checked_repetitions(
        repetitions_for_phase(config, BenchmarkPhase::MultipleRHS)
    );

    // Build exact solution and RHS matrix once.
    std::vector<double> X_true = make_multiple_x_true(problem.n, nrhs, -1.0, 1.0, seed);

    std::vector<double> B(problem.n * nrhs);
    make_multiple_rhs(problem.A0, X_true, B, problem.n, nrhs);

    // ------------------------------------------------------------
    // 1. Benchmark factorization time with repetitions.
    // ------------------------------------------------------------

    double factorization_time_ms = 0.0;
    LU_factorize_Status factor_status = LU_factorize_Status::GeneralFailure;

    for (std::size_t rep = 0; rep < reps; ++rep) {
        std::vector<double> LU_rep = problem.A0;
        std::vector<std::size_t> piv_rep = problem.piv;

        const auto start_factorization = benchmark_now();

        factor_status = LU_factorize(LU_rep, piv_rep, problem.n, problem.eps);

        const auto end_factorization = benchmark_now();

        factorization_time_ms += elapsed_ms(start_factorization, end_factorization);
    }

    factorization_time_ms /= static_cast<double>(reps);

    // ------------------------------------------------------------
    // 2. Factor once for the actual solve benchmark and accuracy.
    // ------------------------------------------------------------

    std::vector<double> LU = problem.A0;
    std::vector<std::size_t> piv = problem.piv;

    factor_status = LU_factorize(LU, piv, problem.n, problem.eps);

    if (factor_status != LU_factorize_Status::Success) {
        const std::string status_string = "factorization_" + to_string(factor_status);

        MultipleRHSResult result;

        result.factorization_time_ms = factorization_time_ms;
        result.estimated_gflops = compute_gflops(
            estimate_lu_factorization_flops(problem.n),
            factorization_time_ms
        );
        result.implementation = Implementation::CustomLU;
        result.matrix_type = problem.matrix_type;
        result.n = problem.n;
        result.nrhs = nrhs;
        result.solution_error_inf = std::numeric_limits<double>::quiet_NaN();
        result.solution_error_l2 = std::numeric_limits<double>::quiet_NaN();
        result.solve_residual_max = std::numeric_limits<double>::quiet_NaN();
        result.solve_residual_mean = std::numeric_limits<double>::quiet_NaN();
        result.solve_kernel_time_ms = std::numeric_limits<double>::quiet_NaN();
        result.solve_loop_time_ms = std::numeric_limits<double>::quiet_NaN();
        result.time_per_rhs_ms = std::numeric_limits<double>::quiet_NaN();
        result.total_time_ms = factorization_time_ms;
        result.trial = trial;
        result.status = status_string;

        return result;
    }

    // ------------------------------------------------------------
    // 3. Benchmark multiple-RHS solve loop with repetitions.
    // ------------------------------------------------------------

    double rhs_solve_kernel_time_ms = 0.0;
    double rhs_solve_loop_time_ms = 0.0;

    LU_Solve_Status solve_status = LU_Solve_Status::Success;

    for (std::size_t rep = 0; rep < reps; ++rep) {
        std::vector<double> x_i_computed(problem.n);
        std::vector<double> b_i(problem.n);

        const auto start_solve_loop = benchmark_now();

        for (std::size_t i = 0; i < nrhs; ++i) {
            b_i = extract_rhs_column(B, problem.n, nrhs, i);

            const auto start_solve_kernel = benchmark_now();

            solve_status = LU_solve(LU, piv, b_i, x_i_computed, problem.n, problem.eps);

            const auto end_solve_kernel = benchmark_now();

            if (solve_status != LU_Solve_Status::Success) {
                const std::string status_string =
                    "solve_" + to_string(solve_status) + "_rhs_" + std::to_string(i);

                MultipleRHSResult result;

                result.factorization_time_ms = factorization_time_ms;
                result.estimated_gflops = std::numeric_limits<double>::quiet_NaN();
                result.implementation = Implementation::CustomLU;
                result.matrix_type = problem.matrix_type;
                result.n = problem.n;
                result.nrhs = nrhs;
                result.solution_error_inf = std::numeric_limits<double>::quiet_NaN();
                result.solution_error_l2 = std::numeric_limits<double>::quiet_NaN();
                result.solve_residual_max = std::numeric_limits<double>::quiet_NaN();
                result.solve_residual_mean = std::numeric_limits<double>::quiet_NaN();
                result.solve_kernel_time_ms = std::numeric_limits<double>::quiet_NaN();
                result.solve_loop_time_ms = std::numeric_limits<double>::quiet_NaN();
                result.total_time_ms = factorization_time_ms;
                result.time_per_rhs_ms = std::numeric_limits<double>::quiet_NaN();
                result.trial = trial;
                result.status = status_string;

                return result;
            }

            rhs_solve_kernel_time_ms += elapsed_ms(start_solve_kernel, end_solve_kernel);
        }

        const auto end_solve_loop = benchmark_now();

        rhs_solve_loop_time_ms += elapsed_ms(start_solve_loop, end_solve_loop);
    }

    rhs_solve_kernel_time_ms /= static_cast<double>(reps);
    rhs_solve_loop_time_ms /= static_cast<double>(reps);

    // ------------------------------------------------------------
    // 4. Solve once for actual X_computed used in accuracy/residual.
    // ------------------------------------------------------------

    std::vector<double> X_computed(problem.n * nrhs);
    std::vector<double> x_i_computed(problem.n);
    std::vector<double> b_i(problem.n);

    for (std::size_t i = 0; i < nrhs; ++i) {
        b_i = extract_rhs_column(B, problem.n, nrhs, i);

        LU_Solve_Status final_solve_status =
            LU_solve(LU, piv, b_i, x_i_computed, problem.n, problem.eps);

        if (final_solve_status != LU_Solve_Status::Success) {
            const std::string status_string =
                "solve_" + to_string(final_solve_status) + "_rhs_" + std::to_string(i);

            MultipleRHSResult result;

            result.factorization_time_ms = factorization_time_ms;
            result.estimated_gflops = std::numeric_limits<double>::quiet_NaN();
            result.implementation = Implementation::CustomLU;
            result.matrix_type = problem.matrix_type;
            result.n = problem.n;
            result.nrhs = nrhs;
            result.solution_error_inf = std::numeric_limits<double>::quiet_NaN();
            result.solution_error_l2 = std::numeric_limits<double>::quiet_NaN();
            result.solve_residual_max = std::numeric_limits<double>::quiet_NaN();
            result.solve_residual_mean = std::numeric_limits<double>::quiet_NaN();
            result.solve_kernel_time_ms = std::numeric_limits<double>::quiet_NaN();
            result.solve_loop_time_ms = std::numeric_limits<double>::quiet_NaN();
            result.total_time_ms = factorization_time_ms;
            result.time_per_rhs_ms = std::numeric_limits<double>::quiet_NaN();
            result.trial = trial;
            result.status = status_string;

            return result;
        }

        store_solution_column(X_computed, x_i_computed, problem.n, nrhs, i);
    }

    // ------------------------------------------------------------
    // 5. Compute residuals and solution errors once.
    // ------------------------------------------------------------

    std::vector<double> rhs_residuals =
        multiple_rhs_solve_residuals(problem.A0, X_computed, B, problem.n, nrhs, norm);

    if (rhs_residuals.size() != nrhs) {
        throw std::runtime_error("benchmark_custom_multiple_rhs_once: residual size mismatch");
    }

    double max_residual = 0.0;
    double avg_residuals = 0.0;

    for (std::size_t i = 0; i < nrhs; ++i) {
        if (rhs_residuals[i] > max_residual) {
            max_residual = rhs_residuals[i];
        }

        avg_residuals += rhs_residuals[i];
    }

    const double estimated_flops =
        estimate_lu_factorization_flops(problem.n)
        + static_cast<double>(nrhs) * estimate_lu_solve_flops(problem.n);

    const double total_time_ms =
        factorization_time_ms + rhs_solve_loop_time_ms;

    MultipleRHSResult result;

    result.estimated_gflops = compute_gflops(estimated_flops, total_time_ms);
    result.factorization_time_ms = factorization_time_ms;
    result.implementation = Implementation::CustomLU;
    result.matrix_type = problem.matrix_type;
    result.n = problem.n;
    result.nrhs = nrhs;
    result.solution_error_inf =
        multiple_rhs_solution_error_inf(X_true, X_computed, problem.n, nrhs);
    result.solution_error_l2 =
        multiple_rhs_solution_error_l2(X_true, X_computed, problem.n, nrhs);
    result.solve_residual_max = max_residual;
    result.solve_residual_mean = avg_residuals / static_cast<double>(nrhs);
    result.solve_kernel_time_ms = rhs_solve_kernel_time_ms;
    result.solve_loop_time_ms = rhs_solve_loop_time_ms;
    result.status = "success";
    result.time_per_rhs_ms = rhs_solve_loop_time_ms / static_cast<double>(nrhs);
    result.total_time_ms = total_time_ms;
    result.trial = trial;

    return result;
}



void run_custom_multiple_rhs_case(
    const BenchmarkConfig& config, std::size_t n, MatrixType matrix_type, std::size_t trial, std::vector<MultipleRHSResult>& multiple_rhs_results, 
    std::vector<MemoryCheckpoint>& memory_results){

    if (n == 0) {
        throw std::invalid_argument("run_custom_multiple_rhs_case: n must be greater than 0");
        }

    if (config.rhs_counts.empty()) {
        throw std::invalid_argument("run_custom_multiple_rhs_case: rhs_counts must not be empty");
       }

    std::uint32_t seed = config.base_seed + static_cast<std::uint32_t>(trial) + static_cast<std::uint32_t>(1000 * n)
    + static_cast<std::uint32_t>(100000 * static_cast<int>(matrix_type));

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::MultipleRHS, 
    matrix_type, n, trial, "before_multiple_rhs_problem_setup");

    auto problem = make_lu_problem(n, matrix_type, seed, config.eps);

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::MultipleRHS, 
    matrix_type, n, trial, "after_multiple_rhs_problem_setup");


    for(std::size_t i = 0; i < config.rhs_counts.size(); ++i){
        
        const std::size_t nrhs = config.rhs_counts[i];

        if (nrhs == 0) {
            throw std::invalid_argument("run_custom_multiple_rhs_case: nrhs must be greater than 0");
            }
        

        record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::MultipleRHS, 
         matrix_type, n, trial, "before_custom_multiple_rhs_benchmark");

        MultipleRHSResult result = benchmark_custom_multiple_rhs_once(problem, nrhs, trial, config.norm, seed, config);
        multiple_rhs_results.push_back(result);

        record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::MultipleRHS, 
         matrix_type, n, trial, "after_custom_multiple_rhs_benchmark");

    }
 }



void run_custom_multiple_rhs_benchmarks(
    const BenchmarkConfig& config, std::vector<MultipleRHSResult>& multiple_rhs_results, std::vector<MemoryCheckpoint>& memory_results){

    if (config.sizes.empty()) {
    throw std::invalid_argument("run_custom_multiple_rhs_benchmarks: sizes must not be empty");
    }

    if (config.matrix_types.empty()) {
        throw std::invalid_argument("run_custom_multiple_rhs_benchmarks: matrix_types must not be empty");
    }

    if (config.trials == 0) {
        throw std::invalid_argument("run_custom_multiple_rhs_benchmarks: trials must be greater than 0");
    }


    for (const std::size_t n : config.sizes) {

        for (const MatrixType matrix_type : config.matrix_types) {

            for (std::size_t trial = 0; trial < config.trials; ++trial) {

                run_custom_multiple_rhs_case(config, n, matrix_type, trial, multiple_rhs_results, memory_results);
            }
        }
    }
 }
