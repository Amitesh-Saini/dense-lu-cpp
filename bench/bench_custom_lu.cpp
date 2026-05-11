#include "bench_custom_lu.hpp"

#include "bench_common.hpp"
#include "bench_generators.hpp"
#include "verify.hpp"
#include "test_lu_utils.hpp"
#include "lu.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

AccuracyResult compute_custom_lu_accuracy(const LUProblem& problem, const std::vector<double>& LU,
 const std::vector<std::size_t>& piv, const std::vector<double>& x_computed, 
 std::size_t trial, NormType norm, const std::string& status){


    AccuracyResult result;

    result.implementation = Implementation::CustomLU;
    result.matrix_type = problem.matrix_type;
    result.n = problem.n;
    result.factorization_residual = residual_factorization_fast(problem.A0, LU, piv, problem.n, norm);
    result.solve_residual = residual_solve(problem.A0, x_computed, problem.b, problem.n, norm);
    result.solution_error_inf = relative_inf_error(problem.x_true, x_computed, problem.n);
    result.solution_error_l2 = relative_l2_error(problem.x_true, x_computed, problem.n);
    result.status = status;
    result.trial = trial;

    return result;
 }



TimingResult benchmark_custom_factorization_once(const LUProblem& problem, std::size_t trial, const BenchmarkConfig& config){

    const std::size_t reps = checked_repetitions(repetitions_for_phase(config, BenchmarkPhase::Factorization));

    LU_factorize_Status status = LU_factorize_Status::GeneralFailure;

    double total_time_ms = 0.0;

    std::vector<double> LU(problem.n * problem.n);
    std::vector<std::size_t> piv(problem.n);

    for (std::size_t rep = 0; rep < reps; ++rep) {
        std::copy(problem.A0.begin(), problem.A0.end(), LU.begin());
        std::copy(problem.piv.begin(), problem.piv.end(), piv.begin());

        const auto start = benchmark_now();

        status = LU_factorize(LU, piv, problem.n, problem.eps);

        const auto end = benchmark_now();

        total_time_ms += elapsed_ms(start, end);
    }

    const double time_ms = total_time_ms / static_cast<double>(reps);
    const double flops = estimate_lu_factorization_flops(problem.n);

    TimingResult result;

    result.implementation = Implementation::CustomLU;
    result.matrix_type = problem.matrix_type;
    result.n = problem.n;
    result.phase = BenchmarkPhase::Factorization;
    result.status = to_string(status);
    result.gflops = compute_gflops(flops, time_ms);
    result.time_ms = time_ms;
    result.trial = trial;

    return result;
}

 TimingResult benchmark_custom_solve_once(const LUProblem& problem, const std::vector<double>& LU, const std::vector<std::size_t>& piv, std::size_t trial,
 const BenchmarkConfig& config){


    const std::size_t reps = checked_repetitions(
        repetitions_for_phase(config, BenchmarkPhase::Solve)
    );

    LU_Solve_Status status = LU_Solve_Status::GeneralFailure;

    double total_time_ms = 0.0;

    for (std::size_t rep = 0; rep < reps; ++rep) {
        std::vector<double> x_computed(problem.n);

        const auto start = benchmark_now();

        status = LU_solve(LU, piv, problem.b, x_computed, problem.n, problem.eps);

        const auto end = benchmark_now();

        total_time_ms += elapsed_ms(start, end);
    }

    const double time_ms = total_time_ms / static_cast<double>(reps);
    const double flops = estimate_lu_solve_flops(problem.n);

    TimingResult result;

    result.gflops = compute_gflops(flops, time_ms);
    result.implementation = Implementation::CustomLU;
    result.matrix_type = problem.matrix_type;
    result.n = problem.n;
    result.phase = BenchmarkPhase::Solve;
    result.status = to_string(status);
    result.time_ms = time_ms;
    result.trial = trial;

    return result;
}



TimingResult benchmark_custom_full_solve_once(const LUProblem& problem, std::size_t trial, const BenchmarkConfig& config){

    const std::size_t reps = checked_repetitions(
        repetitions_for_phase(config, BenchmarkPhase::FullSolve)
    );

    LU_factorize_Status factor_status = LU_factorize_Status::GeneralFailure;
    LU_Solve_Status solve_status = LU_Solve_Status::GeneralFailure;
    bool solve_was_called = false;

    double total_time_ms = 0.0;

    std::vector<double> LU(problem.n * problem.n);
    std::vector<std::size_t> piv(problem.n);
    std::vector<double> x_computed(problem.n);

    for (std::size_t rep = 0; rep < reps; ++rep) {
        std::copy(problem.A0.begin(), problem.A0.end(), LU.begin());
        std::copy(problem.piv.begin(), problem.piv.end(), piv.begin());

        const auto start = benchmark_now();

        factor_status = LU_factorize(LU, piv, problem.n, problem.eps);

        if (factor_status == LU_factorize_Status::Success) {
            solve_was_called = true;
            solve_status = LU_solve(LU, piv, problem.b, x_computed, problem.n, problem.eps);
        }

        const auto end = benchmark_now();

        total_time_ms += elapsed_ms(start, end);
    }

    const double time_ms = total_time_ms / static_cast<double>(reps);

    std::string status_string;

    if (factor_status != LU_factorize_Status::Success) {
        status_string = "factorization_" + to_string(factor_status);
    }
    else if (solve_status != LU_Solve_Status::Success) {
        status_string = "solve_" + to_string(solve_status);
    }
    else {
        status_string = "success";
    }

    const double flops =
        solve_was_called
            ? estimate_lu_factorization_flops(problem.n) + estimate_lu_solve_flops(problem.n)
            : estimate_lu_factorization_flops(problem.n);

    TimingResult result;

    result.gflops = compute_gflops(flops, time_ms);
    result.implementation = Implementation::CustomLU;
    result.matrix_type = problem.matrix_type;
    result.n = problem.n;
    result.phase = BenchmarkPhase::FullSolve;
    result.status = status_string;
    result.time_ms = time_ms;
    result.trial = trial;

    return result;
}



void run_custom_lu_case(const BenchmarkConfig& config, std::size_t n, MatrixType matrix_type, std::size_t trial,
 std::vector<TimingResult>& timing_results, std::vector<AccuracyResult>& accuracy_results, 
 std::vector<MemoryCheckpoint>& memory_results){


    std::uint32_t seed = config.base_seed + static_cast<std::uint32_t>(trial) + static_cast<std::uint32_t>(1000 * n)
    + static_cast<std::uint32_t>(100000 * static_cast<int>(matrix_type));

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::FullSolve, 
    matrix_type, n, trial, "before_problem_setup");

    auto problem = make_lu_problem(n, matrix_type, seed, config.eps);

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::FullSolve, 
    matrix_type, n, trial, "after_problem_setup");

    TimingResult factorization_result = benchmark_custom_factorization_once(problem, trial, config);
    timing_results.push_back(factorization_result);

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::Factorization, 
    matrix_type, n, trial, "after_factorization_benchmark");

    std::vector<double> LU = problem.A0;
    std::vector<std::size_t> piv = problem.piv;
    std::vector<double> x_computed(problem.n);

    LU_factorize_Status factor_status = LU_factorize(LU, piv, problem.n, problem.eps);

    if(factor_status == LU_factorize_Status::Success){

        TimingResult solve_result = benchmark_custom_solve_once(problem, LU, piv, trial, config);

        timing_results.push_back(solve_result);

        LU_Solve_Status solve_status = LU_solve(LU, piv, problem.b, x_computed, problem.n, problem.eps);

        std::string accuracy_status;

        if(solve_status == LU_Solve_Status::Success){


            accuracy_status = "success";

            AccuracyResult accuracy_result = compute_custom_lu_accuracy(problem, LU, piv, x_computed, trial, config.norm, accuracy_status);

            accuracy_results.push_back(accuracy_result);

            record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::Solve,
             matrix_type, n, trial, "after_solve_benchmark_and_accuracy");
        }

        else{

            accuracy_status = "solve_" + to_string(solve_status);

             AccuracyResult accuracy_result;

            accuracy_result.implementation = Implementation::CustomLU;
            accuracy_result.matrix_type = matrix_type;
            accuracy_result.n = n;
            accuracy_result.trial = trial;
            accuracy_result.factorization_residual = std::numeric_limits<double>::quiet_NaN();
            accuracy_result.solution_error_inf = std::numeric_limits<double>::quiet_NaN();
            accuracy_result.solution_error_l2 = std::numeric_limits<double>::quiet_NaN();
            accuracy_result.solve_residual = std::numeric_limits<double>::quiet_NaN();
            accuracy_result.status = accuracy_status;

            accuracy_results.push_back(accuracy_result);
        }
    }

    else{

        AccuracyResult accuracy_result;

        accuracy_result.implementation = Implementation::CustomLU;
        accuracy_result.matrix_type = matrix_type;
        accuracy_result.n = n;
        accuracy_result.trial = trial;
        accuracy_result.factorization_residual = std::numeric_limits<double>::quiet_NaN();
        accuracy_result.solution_error_inf = std::numeric_limits<double>::quiet_NaN();
        accuracy_result.solution_error_l2 = std::numeric_limits<double>::quiet_NaN();
        accuracy_result.solve_residual = std::numeric_limits<double>::quiet_NaN();

        accuracy_result.status = "factorization_" + to_string(factor_status);

        accuracy_results.push_back(accuracy_result);
    }


    TimingResult full_result_solve = benchmark_custom_full_solve_once(problem, trial, config);

    timing_results.push_back(full_result_solve);

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::FullSolve, 
    matrix_type, n, trial, "after_full_solve_benchmark");
 }



void run_custom_lu_benchmarks(const BenchmarkConfig& config, std::vector<TimingResult>& timing_results,
 std::vector<AccuracyResult>& accuracy_results, std::vector<MemoryCheckpoint>& memory_results){

    

    for (const std::size_t n : config.sizes) {

        for (const MatrixType matrix_type : config.matrix_types) {

            for (std::size_t trial = 0; trial < config.trials; ++trial) {

                run_custom_lu_case(config, n, matrix_type, trial, timing_results, accuracy_results, memory_results);
            }
        }
    }
}
