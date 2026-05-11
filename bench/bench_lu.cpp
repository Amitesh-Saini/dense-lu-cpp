#include "bench_common.hpp"
#include "bench_generators.hpp"
#include "bench_custom_lu.hpp"
#include "bench_multiple_rhs.hpp"
#include "bench_residuals.hpp"
#include "bench_eigen_utils.hpp"
#include "bench_csv.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

void run_benchmark_suite(const std::string& suite_name,
                         BenchmarkConfig config,
                         bool run_residual_benchmarks_for_suite)
{
    std::cout << "\n========================================\n";
    std::cout << "Running benchmark suite: " << suite_name << '\n';
    std::cout << "Output directory: " << config.output_dir << '\n';
    std::cout << "========================================\n";

    std::filesystem::create_directories(config.output_dir);

    std::vector<TimingResult> timing_results;
    std::vector<AccuracyResult> accuracy_results;
    std::vector<MemoryCheckpoint> memory_results;
    std::vector<ResidualTimingResult> residual_results;

    std::cout << "Running custom LU benchmarks...\n";
    run_custom_lu_benchmarks(config, timing_results, accuracy_results, memory_results);

    if (run_residual_benchmarks_for_suite) {
        std::cout << "Running residual checker benchmarks...\n";
        run_residual_benchmarks(config, residual_results, memory_results);
    }

    std::cout << "Running Eigen benchmarks...\n";
    run_eigen_benchmarks(config, timing_results, accuracy_results, memory_results);

    std::cout << "Writing CSV files...\n";

    write_timing_csv(config.output_dir + "/timing.csv", timing_results);
    write_accuracy_csv(config.output_dir + "/accuracy.csv", accuracy_results);
    write_memory_csv(config.output_dir + "/memory.csv", memory_results);

    if (run_residual_benchmarks_for_suite) {
        write_residual_timing_csv(config.output_dir + "/residual_timing.csv", residual_results);
    }

    std::cout << "Suite complete: " << suite_name << '\n';
}


void run_multiple_rhs_benchmark_suite(const std::string& suite_name,
                                      BenchmarkConfig config,
                                      bool run_multiple_rhs_residual_benchmarks_for_suite)
{
    std::cout << "\n========================================\n";
    std::cout << "Running benchmark suite: " << suite_name << '\n';
    std::cout << "Output directory: " << config.output_dir << '\n';
    std::cout << "========================================\n";

    std::filesystem::create_directories(config.output_dir);

    std::vector<MultipleRHSResult> multiple_rhs_results;
    std::vector<MemoryCheckpoint> memory_results;
    std::vector<MultipleRHSResidualTimingResult> multiple_rhs_residual_results;

    std::cout << "Running custom LU multiple-RHS benchmarks...\n";
    run_custom_multiple_rhs_benchmarks(config, multiple_rhs_results, memory_results);

    std::cout << "Running Eigen multiple-RHS benchmarks...\n";
    run_eigen_multiple_rhs_benchmarks(config, multiple_rhs_results, memory_results);

    if (run_multiple_rhs_residual_benchmarks_for_suite) {
        std::cout << "Running multiple-RHS residual checker benchmarks...\n";
        run_multiple_rhs_residual_benchmarks(config, multiple_rhs_residual_results, memory_results);
    }

    std::cout << "Writing multiple-RHS CSV files...\n";

    write_multiple_rhs_csv(config.output_dir + "/multiple_rhs.csv", multiple_rhs_results);
    write_memory_csv(config.output_dir + "/memory.csv", memory_results);

    if (run_multiple_rhs_residual_benchmarks_for_suite) {
        write_multiple_rhs_residual_timing_csv(
            config.output_dir + "/multiple_rhs_residual_timing.csv",
            multiple_rhs_residual_results
        );
    }

    std::cout << "Suite complete: " << suite_name << '\n';
}


int main()
{
    try {
        // ============================================================
        // Main performance/correctness suite
        // ============================================================

        BenchmarkConfig main_config;

        main_config.sizes = {
            8, 16, 32, 64, 128, 256, 512
        };

        main_config.matrix_types = {
            MatrixType::RandomDense,
            MatrixType::DiagonallyDominant
        };

        main_config.trials = 5;

        // Repetition settings:
        // trials = independent CSV samples
        // repetitions = averaged kernel repeats inside each trial
        main_config.factorization_repetitions = 20;
        main_config.solve_repetitions = 200;
        main_config.full_solve_repetitions = 20;
        main_config.residual_repetitions = 3;
        main_config.multiple_rhs_repetitions = 1;

        main_config.eps = 1e-12;
        main_config.norm = NormType::Infinity;
        main_config.base_seed = 123456789;
        main_config.output_dir = "data/benchmarks_main";

        run_benchmark_suite("main_performance", main_config, true);


        // ============================================================
        // Pivot and near-singular stress suite
        // ============================================================

        BenchmarkConfig stress_config;

        stress_config.sizes = {
            8, 16, 32, 64, 128, 256
        };

        stress_config.matrix_types = {
            MatrixType::PivotStress,
            MatrixType::NearSingular
        };

        stress_config.trials = 5;

        stress_config.factorization_repetitions = 30;
        stress_config.solve_repetitions = 300;
        stress_config.full_solve_repetitions = 30;
        stress_config.residual_repetitions = 3;
        stress_config.multiple_rhs_repetitions = 1;

        stress_config.eps = 1e-12;
        stress_config.norm = NormType::Infinity;
        stress_config.base_seed = 987654321;
        stress_config.output_dir = "data/benchmarks_stress";

        run_benchmark_suite("pivot_and_near_singular_stress", stress_config, true);


        // ============================================================
        // Hilbert ill-conditioning suite
        // Keep repetitions low because this is a numerical-behavior
        // test, not a performance-stability test.
        // ============================================================

        BenchmarkConfig hilbert_config;

        hilbert_config.sizes = {
            4, 8, 12, 16, 24, 32
        };

        hilbert_config.matrix_types = {
            MatrixType::Hilbert
        };

        hilbert_config.trials = 1;

        hilbert_config.factorization_repetitions = 1;
        hilbert_config.solve_repetitions = 1;
        hilbert_config.full_solve_repetitions = 1;
        hilbert_config.residual_repetitions = 1;
        hilbert_config.multiple_rhs_repetitions = 1;

        hilbert_config.eps = 1e-12;
        hilbert_config.norm = NormType::Infinity;
        hilbert_config.base_seed = 246813579;
        hilbert_config.output_dir = "data/benchmarks_hilbert";

        run_benchmark_suite("hilbert_ill_conditioning", hilbert_config, true);


        // ============================================================
        // Singular failure-behavior suite
        // Keep repetitions low because factorization is expected to fail.
        // ============================================================

        BenchmarkConfig singular_config;

        singular_config.sizes = {
            8, 16, 32, 64, 128
        };

        singular_config.matrix_types = {
            MatrixType::Singular
        };

        singular_config.trials = 1;

        singular_config.factorization_repetitions = 1;
        singular_config.solve_repetitions = 1;
        singular_config.full_solve_repetitions = 1;
        singular_config.residual_repetitions = 1;
        singular_config.multiple_rhs_repetitions = 1;

        singular_config.eps = 1e-12;
        singular_config.norm = NormType::Infinity;
        singular_config.base_seed = 135792468;
        singular_config.output_dir = "data/benchmarks_singular";

        // Residual benchmarks are disabled for singular matrices because
        // factorization should fail, so PA - LU and solve residuals are not meaningful.
        run_benchmark_suite("singular_failure_behavior", singular_config, false);


        // ============================================================
        // Multiple-RHS factorization reuse suite
        // This is now large enough to show meaningful trends.
        // ============================================================

        BenchmarkConfig multiple_rhs_config;

        multiple_rhs_config.sizes = {
            8, 16, 32, 64, 128, 256
        };

        multiple_rhs_config.rhs_counts = {
            1, 2, 4, 8, 16
        };

        multiple_rhs_config.matrix_types = {
            MatrixType::RandomDense,
            MatrixType::DiagonallyDominant
        };

        multiple_rhs_config.trials = 5;

        // These are used by the multiple-RHS benchmark functions.
        multiple_rhs_config.factorization_repetitions = 1;
        multiple_rhs_config.solve_repetitions = 1;
        multiple_rhs_config.full_solve_repetitions = 1;
        multiple_rhs_config.residual_repetitions = 3;
        multiple_rhs_config.multiple_rhs_repetitions = 100;

        multiple_rhs_config.eps = 1e-12;
        multiple_rhs_config.norm = NormType::Infinity;
        multiple_rhs_config.base_seed = 1122334455;
        multiple_rhs_config.output_dir = "data/benchmarks_multiple_rhs";

        run_multiple_rhs_benchmark_suite(
            "multiple_rhs_factorization_reuse",
            multiple_rhs_config,
            true
        );


        std::cout << "\nAll benchmark suites complete.\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << '\n';
        return 1;
    }
}