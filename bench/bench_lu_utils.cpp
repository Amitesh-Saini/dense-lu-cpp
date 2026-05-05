#include "lu.hpp"
#include "verify.hpp"
#include "bench_lu_utils.hpp"
#include "test_lu_utils.hpp"

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <limits>
#include <fstream>
#include <stdexcept>
#include <cstdint>

#ifdef __APPLE__
#include <mach/mach.h>
#endif


// Implement string to string ofr enum clas types


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
    }

    return "unknown_benchmark_phase";
}

std::string to_string(NormType norm) {
    switch (norm) {
        case NormType::Infinity:
            return "infinity";

        case NormType::Frobenius:
            return "frobenius";
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



// Problem setup helpers


std::vector<double> make_pivot_stress_matrix(std::size_t n){


    if (n == 0) {
        throw std::invalid_argument("make_pivot_stress_matrix: n must be greater than 0");
    }

    std::vector<double> A(n * n, 0.0);

    // Start with identity.
    for (std::size_t i = 0; i < n; ++i) {
        A[i * n + i] = 1.0;
    }

    if (n >= 2) {
        // Force the first pivot to be bad in row 0.
        A[0 * n + 0] = 1e-16;

        // Put a strong pivot below it, requiring a row swap.
        A[1 * n + 0] = 1.0;

        // Add some harmless off-diagonal structure.
        A[0 * n + 1] = 2.0;
        A[1 * n + 1] = 3.0;
    }

    return A;
}


std::vector<double> make_near_singular_matrix(std::size_t n){

    if (n == 0) {
        throw std::invalid_argument("make_near_singular_matrix: n must be greater than 0");
    }

    std::vector<double> A(n * n, 0.0);

    // Start with identity.
    for (std::size_t i = 0; i < n; ++i) {
        A[i * n + i] = 1.0;
    }

    if (n >= 2) {
        // Make row 1 almost equal to row 0, but not exactly.
        for (std::size_t j = 0; j < n; ++j) {
            A[1 * n + j] = A[0 * n + j];
        }

        A[1 * n + 1] += 1e-12;
    }

    return A;
}


std::vector<double> make_singular_matrix(std::size_t n){


    if (n == 0) {
        throw std::invalid_argument("make_singular_matrix: n must be greater than 0");
    }

    std::vector<double> A(n * n, 0.0);

    // Start with identity.
    for (std::size_t i = 0; i < n; ++i) {
        A[i * n + i] = 1.0;
    }

    if (n >= 2) {
        // Make row 1 exactly equal to row 0.
        for (std::size_t j = 0; j < n; ++j) {
            A[1 * n + j] = A[0 * n + j];
        }
    }

    return A;
}

std::vector<double> generate_seeded_random_dense_matrix(double lower_bound, double upper_bound, std::size_t n, std::uint32_t seed){

    if (n == 0) {
    throw std::invalid_argument("generate_seeded_random_dense_matrix: n must be greater than 0");
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);

    std::vector<double> A(n * n);

    for (std::size_t i = 0; i < n * n; ++i) {
        A[i] = dist(gen);
    }

    return A;
}

std::vector<double> generate_seeded_random_vector(double lower_bound, double upper_bound, std::size_t n, std::uint32_t seed){

    if (n == 0) {
    throw std::invalid_argument("generate_seeded_random_vector: n must be greater than 0");
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);

    std::vector<double> x(n);

    for (std::size_t i = 0; i < n; ++i) {
        x[i] = dist(gen);
    }

    return x;
}

std::vector<double> generate_seeded_random_diagonally_dominant_matrix(double lower_bound, double upper_bound, std::size_t n,
 double margin, std::uint32_t seed){


    if (n == 0) {
        throw std::invalid_argument("generate_seeded_random_diagonally_dominant_matrix: n must be greater than 0");
    }

    if (margin <= 0.0) {
        throw std::invalid_argument("generate_seeded_random_diagonally_dominant_matrix: margin must be positive");
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);

    std::vector<double> A(n * n);

    for (std::size_t i = 0; i < n; ++i) {
        double non_diagonal_sum = 0.0;

        for (std::size_t j = 0; j < n; ++j) {
            A[i * n + j] = dist(gen);

            if (i != j) {
                non_diagonal_sum += std::abs(A[i * n + j]);
            }
        }

        const double sign = (dist(gen) < 0.0) ? -1.0 : 1.0;
        A[i * n + i] = sign * (non_diagonal_sum + margin);
    }

    return A;
 }





std::vector<double> make_benchmark_matrix(std::size_t n, MatrixType type, std::uint32_t seed){


    if (n == 0) {
        throw std::invalid_argument("make_benchmark_matrix: n must be greater than 0");
    }

    // Fixed benchmark-generation defaults.
    // Random dense and diagonally dominant matrices use the same value range
    // so benchmark cases are controlled and comparable.
    constexpr double lower = -1.0;
    constexpr double upper = 1.0;
    constexpr double diagonal_dominance_margin = 1.0;

    switch (type) {
        case MatrixType::RandomDense:
            return generate_seeded_random_dense_matrix(lower, upper, n, seed);

        case MatrixType::DiagonallyDominant:
            return generate_seeded_random_diagonally_dominant_matrix(lower, upper, n, diagonal_dominance_margin, seed);

        case MatrixType::Hilbert:
            return make_hilbert_matrix(n);

        case MatrixType::PivotStress:
            return make_pivot_stress_matrix(n);

        case MatrixType::NearSingular:
            return make_near_singular_matrix(n);

        case MatrixType::Singular:
            return make_singular_matrix(n);
    }

    throw std::invalid_argument("make_benchmark_matrix: unknown MatrixType");
}


std::vector<double> make_benchmark_vector(std::size_t n, double lower, double upper, std::uint32_t seed){

    if(n == 0) {
        throw std::invalid_argument("make_benchmark_vector: n must be greater than 0");
    }

    return generate_seeded_random_vector(lower, upper, n, seed);
}


std::vector<std::size_t> make_identity_pivot(std::size_t n){

     if(n == 0) {
        throw std::invalid_argument("make_identity_pivot: n must be greater than 0");
    }

    std::vector<std::size_t> piv(n);

    std::iota(piv.begin(), piv.end(), 0);

    return piv;

}



LUProblem make_lu_problem(std::size_t n, MatrixType matrix_type, std::uint32_t seed, double eps){


    if (n == 0) {
        throw std::invalid_argument("make_lu_problem: n must be greater than 0");
    }

    if (eps <= 0.0) {
        throw std::invalid_argument("make_lu_problem: eps must be positive");
    }

    constexpr double lower = -1.0;
    constexpr double upper = 1.0;

    LUProblem problem;

    problem.A0 = make_benchmark_matrix(n, matrix_type, seed);

     // Use a different seed from the matrix so x_true is independently generated.
    problem.x_true = make_benchmark_vector(n, lower, upper, seed+1);

    matrix_vector_mul(problem.A0, problem.x_true, problem.b, n);

    problem.piv = make_identity_pivot(n);

    problem.eps = eps;

    problem.matrix_type = matrix_type;

    problem.n = n;

    problem.seed = seed;

    return problem;

}




std::chrono::steady_clock::time_point benchmark_now(){

    return std::chrono::steady_clock::now();
}



double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end){

    std::chrono::duration<double, std::milli> elapsed = end - start;

    return elapsed.count();
}



std::size_t current_process_memory_bytes()
{
#ifdef __APPLE__
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
#else
    return 0;
#endif
}


void record_memory_checkpoint(std::vector<MemoryCheckpoint>& checkpoints, Implementation implementation,
 BenchmarkPhase phase, MatrixType matrix_type, std::size_t n, std::size_t trial, const std::string& checkpoint){

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



 TimingResult benchmark_custom_factorization_once(const LUProblem& problem, std::size_t trial){

    std::vector<double> LU = problem.A0;
    std::vector<std::size_t> piv = problem.piv;

    LU_factorize_Status status;

    auto start = benchmark_now();

    status = LU_factorize(LU, piv, problem.n, problem.eps);

    auto end = benchmark_now();

    const double time_ms = elapsed_ms(start, end);

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


 TimingResult benchmark_custom_solve_once(const LUProblem& problem, const std::vector<double>& LU, 
  const std::vector<std::size_t>& piv, std::size_t trial){


    std::vector<double> x_computed(problem.n);

    LU_Solve_Status status;

    auto start = benchmark_now();

    status = LU_solve(LU, piv, problem.b, x_computed, problem.n, problem.eps);

    auto end = benchmark_now();

    const double time_ms = elapsed_ms(start, end);
    
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



TimingResult benchmark_custom_full_solve_once(const LUProblem& problem, std::size_t trial){


    std::vector<double> LU = problem.A0;
    std::vector<std::size_t> piv = problem.piv;
    std::vector<double> x_computed(problem.n);

    LU_factorize_Status factor_status;
    LU_Solve_Status solve_status = LU_Solve_Status::GeneralFailure;
    bool solve_was_called = false;

    auto start = benchmark_now();

    factor_status = LU_factorize(LU, piv, problem.n, problem.eps);

    if (factor_status == LU_factorize_Status::Success) {
        solve_was_called = true;
        solve_status = LU_solve(LU, piv, problem.b, x_computed, problem.n, problem.eps);
    }

    auto end = benchmark_now();

    const double time_ms = elapsed_ms(start, end);

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

    TimingResult factorization_result = benchmark_custom_factorization_once(problem, trial);
    timing_results.push_back(factorization_result);

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::Factorization, 
    matrix_type, n, trial, "after_factorization_benchmark");

    std::vector<double> LU = problem.A0;
    std::vector<std::size_t> piv = problem.piv;
    std::vector<double> x_computed(problem.n);

    LU_factorize_Status factor_status = LU_factorize(LU, piv, problem.n, problem.eps);

    if(factor_status == LU_factorize_Status::Success){

        TimingResult solve_result = benchmark_custom_solve_once(problem, LU, piv, trial);

        timing_results.push_back(solve_result);

        LU_Solve_Status solve_status = LU_solve(LU, piv, problem.b, x_computed, problem.n, problem.eps);

        std::string accuracy_status;

        if(solve_status == LU_Solve_Status::Success) accuracy_status = "success";

        else{
            accuracy_status = "solve_" + to_string(solve_status);
        }

        AccuracyResult accuracy_result = compute_custom_lu_accuracy(problem, LU, piv, x_computed, trial, 
        config.norm, accuracy_status);

        accuracy_results.push_back(accuracy_result);

        record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::Solve, 
        matrix_type, n, trial, "after_solve_benchmark_and_accuracy");
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


    TimingResult full_result_solve = benchmark_custom_full_solve_once(problem, trial);

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



ResidualTimingResult benchmark_factorization_residual_debug_once(const LUProblem& problem, const std::vector<double>& LU, const std::vector<std::size_t>& piv,
 std::size_t trial, NormType norm){


    std::vector<double> LU_multiplied(problem.n * problem.n);
    std::vector<double> PA(problem.n * problem.n);
    std::vector<double> diff(problem.n * problem.n);


    auto start = benchmark_now();

    const double residual_value = residual_factorization_debug(problem.A0, LU_multiplied, piv, PA, LU, diff, problem.n, norm);

    auto end = benchmark_now();

    const double time_ms = elapsed_ms(start, end);

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


ResidualTimingResult benchmark_factorization_residual_fast_once(const LUProblem& problem, 
 const std::vector<double>& LU, const std::vector<std::size_t>& piv, std::size_t trial, NormType norm){

    auto start = benchmark_now();

    const double residual_value = residual_factorization_fast(problem.A0, LU, piv, problem.n, norm);

    auto end = benchmark_now();

    const double time_ms = elapsed_ms(start, end);

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


 ResidualTimingResult benchmark_solve_residual_once(const LUProblem& problem, const std::vector<double>& x,
 std::size_t trial, NormType norm){

    auto start = benchmark_now();

    const double residual_value = residual_solve(problem.A0, x, problem.b, problem.n, norm);

    auto end = benchmark_now();

    const double time_ms = elapsed_ms(start, end);

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


 void run_residual_benchmark_case(const BenchmarkConfig& config, std::size_t n, MatrixType matrix_type,
 std::size_t trial, std::vector<ResidualTimingResult>& residual_results, 
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

    residual_results.push_back(benchmark_factorization_residual_debug_once(problem, LU, piv, trial, config.norm));

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::ResidualFactorizationDebug, 
    matrix_type, n, trial, "after_debug_factorization_residual_benchmark");

    residual_results.push_back(benchmark_factorization_residual_fast_once(problem, LU, piv, trial, config.norm));

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::ResidualFactorizationFast, 
    matrix_type, n, trial, "after_fast_factorization_residual_benchmark");

    std::vector<double> x_computed(problem.n);

    LU_Solve_Status solve_status = LU_solve(LU, piv, problem.b, x_computed, problem.n, problem.eps);

    if(solve_status != LU_Solve_Status::Success) return;

    residual_results.push_back(benchmark_solve_residual_once(problem, x_computed, trial, config.norm));

    record_memory_checkpoint(memory_results, Implementation::CustomLU, BenchmarkPhase::ResidualSolve, 
    matrix_type, n, trial, "after_solve_residual_benchmark");
 }


 void run_residual_benchmarks(const BenchmarkConfig& config, std::vector<ResidualTimingResult>& residual_results, 
 std::vector<MemoryCheckpoint>& memory_results){


    for (const std::size_t n : config.sizes) {

        for (const MatrixType matrix_type : config.matrix_types) {

            for (std::size_t trial = 0; trial < config.trials; ++trial) {

                run_residual_benchmark_case(config, n, matrix_type, trial, residual_results, memory_results);
            }
        }
    }
 }



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