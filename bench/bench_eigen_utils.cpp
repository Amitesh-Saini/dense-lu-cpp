#include "bench_eigen_utils.hpp"

#include "bench_common.hpp"
#include "bench_generators.hpp"
#include "verify.hpp"
#include "test_lu_utils.hpp"
#include "lu.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

Eigen::MatrixXd to_eigen_matrix(const std::vector<double>& A, std::size_t n)
{
    Eigen::MatrixXd A_eigen(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A_eigen(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = at(A, n, i, j);
        }
    }

    return A_eigen;
}


Eigen::VectorXd to_eigen_vector(const std::vector<double>& x)
{
    const std::size_t n = x.size();

    Eigen::VectorXd x_eigen(static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i) {
        x_eigen(static_cast<Eigen::Index>(i)) = x[i];
    }

    return x_eigen;
}


std::vector<double> from_eigen_vector(const Eigen::VectorXd& x)
{
    const Eigen::Index n_eigen = x.size();

    std::vector<double> x_vector(static_cast<std::size_t>(n_eigen));

    for (Eigen::Index i = 0; i < n_eigen; ++i) {
        x_vector[static_cast<std::size_t>(i)] = x(i);
    }

    return x_vector;
}


Eigen::MatrixXd to_eigen_rhs_matrix(const std::vector<double>& X, std::size_t n, std::size_t nrhs){

    if (n == 0) {
        throw std::invalid_argument("to_eigen_rhs_matrix: n must be greater than 0");
    }

    if (nrhs == 0) {
        throw std::invalid_argument("to_eigen_rhs_matrix: nrhs must be greater than 0");
    }

    if (X.size() != n * nrhs) {
        throw std::invalid_argument("to_eigen_rhs_matrix: X size mismatch");
    }

    Eigen::MatrixXd X_eigen(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(nrhs));

    for(std::size_t rhs = 0; rhs < nrhs; ++rhs){
        for(std::size_t i = 0; i < n; ++i){
             X_eigen(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(rhs)) = X[i + rhs * n];
        }
    }

    return X_eigen;
}


std::vector<double> from_eigen_rhs_matrix(const Eigen::MatrixXd& X){

    if (X.rows() <= 0) {
        throw std::invalid_argument("from_eigen_rhs_matrix: number of rows must be greater than 0");
    }

    if (X.cols() <= 0) {
        throw std::invalid_argument("from_eigen_rhs_matrix: number of RHS columns must be greater than 0");
    }

    const std::size_t n = static_cast<std::size_t>(X.rows());

    const std::size_t nrhs = static_cast<std::size_t>(X.cols());

    std::vector<double> X_vector(n * nrhs);

    for (std::size_t rhs = 0; rhs < nrhs; ++rhs) {
        for (std::size_t i = 0; i < n; ++i) {
            X_vector[i + rhs * n] = X(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(rhs));
        }
    }

    return X_vector;
}




TimingResult benchmark_eigen_factorization_once(const LUProblem& problem, std::size_t trial, const BenchmarkConfig& config){


    const std::size_t reps = checked_repetitions(
        repetitions_for_phase(config, BenchmarkPhase::Factorization)
    );

    std::string status = "success";
    double total_time_ms = 0.0;

    for (std::size_t rep = 0; rep < reps; ++rep) {
        Eigen::MatrixXd problem_matrix = to_eigen_matrix(problem.A0, problem.n);
        Eigen::PartialPivLU<Eigen::MatrixXd> lu;

        const auto start = benchmark_now();

        lu.compute(problem_matrix);

        const auto end = benchmark_now();

        total_time_ms += elapsed_ms(start, end);

        if (lu.info() != Eigen::Success) {
            status = "factorization_failure";
        }
    }

    const double time_ms = total_time_ms / static_cast<double>(reps);
    const double flops = estimate_lu_factorization_flops(problem.n);

    TimingResult result;

    result.gflops = compute_gflops(flops, time_ms);
    result.implementation = Implementation::EigenPartialPivLU;
    result.matrix_type = problem.matrix_type;
    result.n = problem.n;
    result.phase = BenchmarkPhase::Factorization;
    result.time_ms = time_ms;
    result.trial = trial;
    result.status = status;

    return result;
}



TimingResult benchmark_eigen_solve_once(const LUProblem& problem, std::size_t trial, const BenchmarkConfig& config){


    const std::size_t reps = checked_repetitions(repetitions_for_phase(config, BenchmarkPhase::Solve));

    Eigen::MatrixXd problem_matrix = to_eigen_matrix(problem.A0, problem.n);
    Eigen::VectorXd problem_b = to_eigen_vector(problem.b);

    Eigen::PartialPivLU<Eigen::MatrixXd> lu;
    lu.compute(problem_matrix);

    std::string status;
    double time_ms = 0.0;

    if (lu.info() != Eigen::Success) {
        status = "factorization_failure";
    }
    else {
        double total_time_ms = 0.0;

        for (std::size_t rep = 0; rep < reps; ++rep) {
            const auto start = benchmark_now();

            Eigen::VectorXd x = lu.solve(problem_b);

            const auto end = benchmark_now();

            total_time_ms += elapsed_ms(start, end);
        }

        time_ms = total_time_ms / static_cast<double>(reps);
        status = "success";
    }

    const double flops = estimate_lu_solve_flops(problem.n);

    TimingResult result;

    result.gflops = compute_gflops(flops, time_ms);
    result.implementation = Implementation::EigenPartialPivLU;
    result.matrix_type = problem.matrix_type;
    result.n = problem.n;
    result.phase = BenchmarkPhase::Solve;
    result.time_ms = time_ms;
    result.trial = trial;
    result.status = status;

    return result;
}

TimingResult benchmark_eigen_full_solve_once(const LUProblem& problem, std::size_t trial, const BenchmarkConfig& config){

    const std::size_t reps = checked_repetitions(repetitions_for_phase(config, BenchmarkPhase::FullSolve));

    std::string status = "success";
    double total_time_ms = 0.0;

    for (std::size_t rep = 0; rep < reps; ++rep) {
        Eigen::MatrixXd problem_matrix = to_eigen_matrix(problem.A0, problem.n);
        Eigen::VectorXd problem_b = to_eigen_vector(problem.b);

        Eigen::PartialPivLU<Eigen::MatrixXd> lu;

        const auto start = benchmark_now();

        lu.compute(problem_matrix);

        if (lu.info() != Eigen::Success) {
            status = "factorization_failure";
        }
        else {
            Eigen::VectorXd x = lu.solve(problem_b);
            status = "success";
        }

        const auto end = benchmark_now();

        total_time_ms += elapsed_ms(start, end);
    }

    const double time_ms = total_time_ms / static_cast<double>(reps);

    const double flops =
        estimate_lu_factorization_flops(problem.n)
        + estimate_lu_solve_flops(problem.n);

    TimingResult result;

    result.gflops = compute_gflops(flops, time_ms);
    result.implementation = Implementation::EigenPartialPivLU;
    result.matrix_type = problem.matrix_type;
    result.n = problem.n;
    result.phase = BenchmarkPhase::FullSolve;
    result.time_ms = time_ms;
    result.trial = trial;
    result.status = status;

    return result;
}


AccuracyResult compute_eigen_accuracy(const LUProblem& problem, const std::vector<double>& x_eigen, std::size_t trial, NormType norm, const std::string& status){



    AccuracyResult result; 


    result.factorization_residual = std::numeric_limits<double>::quiet_NaN();
    result.implementation = Implementation::EigenPartialPivLU;
    result.matrix_type = problem.matrix_type;
    result.n = problem.n;
    result.status = status;
    result.solve_residual = residual_solve(problem.A0, x_eigen, problem.b, problem.n, norm);
    result.solution_error_inf = relative_inf_error(problem.x_true, x_eigen, problem.n);
    result.solution_error_l2 = relative_l2_error(problem.x_true, x_eigen, problem.n);
    result.trial = trial;


    return result;
}



void run_eigen_case(const BenchmarkConfig& config, std::size_t n, MatrixType matrix_type, std::size_t trial, std::vector<TimingResult>& timing_results,
 std::vector<AccuracyResult>& accuracy_results, std::vector<MemoryCheckpoint>& memory_results){



    std::uint32_t seed = config.base_seed + static_cast<std::uint32_t>(trial) + static_cast<std::uint32_t>(1000 * n)
    + static_cast<std::uint32_t>(100000 * static_cast<int>(matrix_type));

    record_memory_checkpoint(memory_results, Implementation::EigenPartialPivLU, BenchmarkPhase::Factorization, 
    matrix_type, n, trial, "before_problem_setup");

    auto problem = make_lu_problem(n, matrix_type, seed, config.eps);
    
    record_memory_checkpoint(memory_results, Implementation::EigenPartialPivLU, BenchmarkPhase::Factorization, 
    matrix_type, n, trial, "after_problem_setup");

    TimingResult factorization_result = benchmark_eigen_factorization_once(problem , trial, config);
    timing_results.push_back(factorization_result);

    record_memory_checkpoint(memory_results, Implementation::EigenPartialPivLU, BenchmarkPhase::Factorization,
        matrix_type, n, trial, "after_factorization_benchmark");

    Eigen::MatrixXd problem_matrix = to_eigen_matrix(problem.A0, problem.n);
    Eigen::VectorXd problem_b = to_eigen_vector(problem.b);

    Eigen::PartialPivLU<Eigen::MatrixXd> lu;

    lu.compute(problem_matrix);

    std::string status;

    if(lu.info() == Eigen::Success){

        TimingResult solve_result = benchmark_eigen_solve_once(problem, trial, config);
        timing_results.push_back(solve_result);

        Eigen::VectorXd x_eigen_computed = lu.solve(problem_b);
        std::vector<double> x_computed = from_eigen_vector(x_eigen_computed);

        AccuracyResult accuracy_result = compute_eigen_accuracy(problem, x_computed, trial, config.norm, "success");

        accuracy_results.push_back(accuracy_result);

        record_memory_checkpoint(memory_results, Implementation::EigenPartialPivLU, BenchmarkPhase::Solve,
        matrix_type, n, trial, "after_solve_benchmark_and_accuracy");
    }

    else{

        AccuracyResult accuracy_result;

        accuracy_result.implementation = Implementation::EigenPartialPivLU;
        accuracy_result.matrix_type = matrix_type;
        accuracy_result.n = n;
        accuracy_result.trial = trial;
        accuracy_result.factorization_residual = std::numeric_limits<double>::quiet_NaN();
        accuracy_result.solution_error_inf = std::numeric_limits<double>::quiet_NaN();
        accuracy_result.solution_error_l2 = std::numeric_limits<double>::quiet_NaN();
        accuracy_result.solve_residual = std::numeric_limits<double>::quiet_NaN();

        accuracy_result.status = "factorization_failure";

        accuracy_results.push_back(accuracy_result);
    }


    TimingResult full_solve_result = benchmark_eigen_full_solve_once(problem, trial, config);
    timing_results.push_back(full_solve_result);


    record_memory_checkpoint(memory_results, Implementation::EigenPartialPivLU, BenchmarkPhase::FullSolve,
        matrix_type, n, trial, "after_full_solve_benchmark");


 }




 void run_eigen_benchmarks(const BenchmarkConfig& config, std::vector<TimingResult>& timing_results, std::vector<AccuracyResult>& accuracy_results,
 std::vector<MemoryCheckpoint>& memory_results){


    for (const std::size_t n : config.sizes) {

        for (const MatrixType matrix_type : config.matrix_types) {

            for (std::size_t trial = 0; trial < config.trials; ++trial) {

                run_eigen_case(config, n, matrix_type, trial, timing_results, accuracy_results, memory_results);
            }
        }
    }
 }



MultipleRHSResult benchmark_eigen_multiple_rhs_once(const LUProblem& problem, std::size_t nrhs, std::size_t trial,
 NormType norm, std::uint32_t seed, const BenchmarkConfig& config){


    if (nrhs == 0) {
        throw std::invalid_argument("benchmark_eigen_multiple_rhs_once: nrhs must be greater than 0");
    }

    if (problem.n == 0) {
        throw std::invalid_argument("benchmark_eigen_multiple_rhs_once: n must be greater than 0");
    }

    const std::size_t reps = checked_repetitions(
        repetitions_for_phase(config, BenchmarkPhase::MultipleRHS)
    );

    // Build exact solution and RHS matrix once.
    std::vector<double> X_true =
        make_multiple_x_true(problem.n, nrhs, -1.0, 1.0, seed);

    std::vector<double> B(problem.n * nrhs);
    make_multiple_rhs(problem.A0, X_true, B, problem.n, nrhs);

    Eigen::MatrixXd eigen_B = to_eigen_rhs_matrix(B, problem.n, nrhs);

    // ------------------------------------------------------------
    // 1. Benchmark Eigen factorization with repetitions.
    // ------------------------------------------------------------

    double factorization_time_ms = 0.0;
    std::string factorization_status = "success";

    for (std::size_t rep = 0; rep < reps; ++rep) {
        Eigen::MatrixXd eigen_A = to_eigen_matrix(problem.A0, problem.n);
        Eigen::PartialPivLU<Eigen::MatrixXd> lu;

        const auto start_factorization = benchmark_now();

        lu.compute(eigen_A);

        const auto end_factorization = benchmark_now();

        factorization_time_ms += elapsed_ms(start_factorization, end_factorization);

        if (lu.info() != Eigen::Success) {
            factorization_status = "factorization_failure";
        }
    }

    factorization_time_ms /= static_cast<double>(reps);

    // ------------------------------------------------------------
    // 2. Factor once for actual solve benchmark and accuracy.
    // ------------------------------------------------------------

    Eigen::MatrixXd eigen_A = to_eigen_matrix(problem.A0, problem.n);
    Eigen::PartialPivLU<Eigen::MatrixXd> lu;

    lu.compute(eigen_A);

    if (lu.info() != Eigen::Success) {
        MultipleRHSResult result;

        result.factorization_time_ms = factorization_time_ms;
        result.estimated_gflops = std::numeric_limits<double>::quiet_NaN();
        result.implementation = Implementation::EigenPartialPivLU;
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
        result.status = "factorization_failure";

        return result;
    }

    // ------------------------------------------------------------
    // 3. Benchmark Eigen multiple-RHS solve with repetitions.
    // ------------------------------------------------------------

    double solve_time_ms = 0.0;

    for (std::size_t rep = 0; rep < reps; ++rep) {
        const auto start_solve = benchmark_now();

        Eigen::MatrixXd X_computed_eigen = lu.solve(eigen_B);

        const auto end_solve = benchmark_now();

        solve_time_ms += elapsed_ms(start_solve, end_solve);
    }

    solve_time_ms /= static_cast<double>(reps);

    // ------------------------------------------------------------
    // 4. Solve once for actual residual/error computation.
    // ------------------------------------------------------------

    Eigen::MatrixXd X_computed_eigen = lu.solve(eigen_B);
    std::vector<double> X_computed_vector = from_eigen_rhs_matrix(X_computed_eigen);

    std::vector<double> rhs_residuals = multiple_rhs_solve_residuals(problem.A0, X_computed_vector, B, problem.n, nrhs, norm);

    if (rhs_residuals.size() != nrhs) {
        throw std::runtime_error("benchmark_eigen_multiple_rhs_once: residual size mismatch");
    }

    double max_residual = 0.0;
    double avg_residuals = 0.0;

    for (std::size_t i = 0; i < nrhs; ++i) {
        if (rhs_residuals[i] > max_residual) {
            max_residual = rhs_residuals[i];
        }

        avg_residuals += rhs_residuals[i];
    }

    const double estimated_flops = estimate_lu_factorization_flops(problem.n) + static_cast<double>(nrhs) * estimate_lu_solve_flops(problem.n);

    const double total_time_ms = factorization_time_ms + solve_time_ms;

    MultipleRHSResult result;

    result.estimated_gflops = compute_gflops(estimated_flops, total_time_ms);
    result.factorization_time_ms = factorization_time_ms;
    result.implementation = Implementation::EigenPartialPivLU;
    result.matrix_type = problem.matrix_type;
    result.n = problem.n;
    result.nrhs = nrhs;
    result.solution_error_inf = multiple_rhs_solution_error_inf(X_true, X_computed_vector, problem.n, nrhs);
    result.solution_error_l2 = multiple_rhs_solution_error_l2(X_true, X_computed_vector, problem.n, nrhs);
    result.solve_kernel_time_ms = solve_time_ms;
    result.solve_loop_time_ms = solve_time_ms;
    result.solve_residual_max = max_residual;
    result.solve_residual_mean = avg_residuals / static_cast<double>(nrhs);
    result.status = factorization_status;
    result.time_per_rhs_ms = solve_time_ms / static_cast<double>(nrhs);
    result.total_time_ms = total_time_ms;
    result.trial = trial;

    return result;
}



void run_eigen_multiple_rhs_case(const BenchmarkConfig& config, std::size_t n, MatrixType matrix_type, std::size_t trial,
 std::vector<MultipleRHSResult>& multiple_rhs_results, std::vector<MemoryCheckpoint>& memory_results){


    if (n == 0) {
        throw std::invalid_argument("run_eigen_multiple_rhs_case: n must be greater than 0");
    }

    if (config.rhs_counts.empty()) {
        throw std::invalid_argument("run_eigen_multiple_rhs_case: rhs_counts must not be empty");
    }

    std::uint32_t seed = config.base_seed + static_cast<std::uint32_t>(trial) + static_cast<std::uint32_t>(1000 * n)
    + static_cast<std::uint32_t>(100000 * static_cast<int>(matrix_type));

    record_memory_checkpoint(memory_results, Implementation::EigenPartialPivLU, BenchmarkPhase::MultipleRHS, 
    matrix_type, n, trial, "before_multiple_rhs_problem_setup");

    auto problem = make_lu_problem(n, matrix_type, seed, config.eps);

    record_memory_checkpoint(memory_results, Implementation::EigenPartialPivLU, BenchmarkPhase::MultipleRHS, 
    matrix_type, n, trial, "after_multiple_rhs_problem_setup");


    for(std::size_t i = 0; i < config.rhs_counts.size(); ++i){
        
        const std::size_t nrhs = config.rhs_counts[i];

        if (nrhs == 0) {
            throw std::invalid_argument("run_eigen_multiple_rhs_case: nrhs must be greater than 0");
        }

        record_memory_checkpoint(memory_results, Implementation::EigenPartialPivLU, BenchmarkPhase::MultipleRHS, 
         matrix_type, n, trial, "before_eigen_multiple_rhs_benchmark");

        MultipleRHSResult result = benchmark_eigen_multiple_rhs_once(problem, nrhs, trial, config.norm, seed, config);
        multiple_rhs_results.push_back(result);

        record_memory_checkpoint(memory_results, Implementation::EigenPartialPivLU, BenchmarkPhase::MultipleRHS, 
         matrix_type, n, trial, "after_eigen_multiple_rhs_benchmark");

    }
 }


 void run_eigen_multiple_rhs_benchmarks(const BenchmarkConfig& config, std::vector<MultipleRHSResult>& multiple_rhs_results, 
 std::vector<MemoryCheckpoint>& memory_results){

    if (config.sizes.empty()) {
    throw std::invalid_argument("run_eigen_multiple_rhs_benchmarks: sizes must not be empty");
    }

    if (config.matrix_types.empty()) {
        throw std::invalid_argument("run_eigen_multiple_rhs_benchmarks: matrix_types must not be empty");
    }

    if (config.trials == 0) {
        throw std::invalid_argument("run_eigen_multiple_rhs_benchmarks: trials must be greater than 0");
    }

    for (const std::size_t n : config.sizes) {

        for (const MatrixType matrix_type : config.matrix_types) {

            for (std::size_t trial = 0; trial < config.trials; ++trial) {

                run_eigen_multiple_rhs_case(config, n, matrix_type, trial, multiple_rhs_results, memory_results);
            }
        }
    }
}
