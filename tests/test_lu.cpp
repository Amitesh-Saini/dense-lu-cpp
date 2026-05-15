// test_lu.cpp
// Purpose:
//   Unit tests for LU factorization + solve.
//
// What to test:
//   - Factorization correctness via residual:
//       ||P*A - L*U|| / ||A||
//   - Solve correctness via backward residual:
//       ||b - A*x|| / (||A||*||x|| + ||b||)
//   - Edge cases:
//       * singular matrices should fail factorization
//       * small sizes (2x2, 3x3) with known answers
//       * random matrices across several sizes
//   - Eigen comparison

#include "test_lu_utils.hpp"
#include "lu.hpp"
#include "verify.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>


int main()
{
    std::vector<std::string> failed_tests;   // Initialize array to hold which tests failed

    const double pivot_tol    = 1e-12;
    const double residual_tol = 1e-10;
    const double solution_tol = 1e-10;

    std::cout << "=== Running LU solver tests ===\n\n";

    // Small deterministic correctness tests

    {
        std::vector<double> A = make_known_dense_3x3();
        std::vector<double> x_true = {2.0, 3.0, -1.0};

        if(!run_known_solution(A, x_true, 3, pivot_tol, residual_tol, solution_tol)){
            failed_tests.push_back("known_dense_3x3");
        }
    }

    {
        std::vector<double> A = make_identity_matrix(3);
        std::vector<double> x_true = {1.0, -2.0, 4.0};

        if(!run_known_solution(A, x_true, 3, pivot_tol, residual_tol, solution_tol)){
            failed_tests.push_back("identity_3x3");
        }
    }

    {
        std::vector<double> A = make_pivot_required_3x3();
        std::vector<double> x_true = {1.0, -1.0, 2.0};

        if(!run_known_solution(A, x_true, 3, pivot_tol, residual_tol, solution_tol)){
            failed_tests.push_back("pivot_required_3x3");
        }
    }

    {
        std::vector<double> A = make_singular_repeated_row_3x3();

        if(!run_expected_factorization_failure(A, 3, pivot_tol)){
            failed_tests.push_back("singular_repeated_row_3x3");
        }
    }


    // Random diagonally dominant tests

    for(std::size_t n : {4, 8, 16, 32, 64, 128, 256}) {
        std::vector<double> A = generate_random_diagonally_dominant_matrix(-1.0, 1.0, n, 1.0);
        std::vector<double> x_true = generate_random_vector(-1.0, 1.0, n);

        if(!run_known_solution(A, x_true, n, pivot_tol, residual_tol, solution_tol)){
            failed_tests.push_back("random_diagonally_dominant_n_" + std::to_string(n));
        }
    }

    for(std::size_t n : {512, 1024}) {
        std::vector<double> A = generate_random_diagonally_dominant_matrix(-1.0, 1.0, n, 1.0);
        std::vector<double> x_true = generate_random_vector(-1.0, 1.0, n);

        if(!run_known_solution(A, x_true, n, pivot_tol, residual_tol, solution_tol)) {
            failed_tests.push_back("heavy_random_diagonally_dominant_n_" + std::to_string(n));
        }
    }


    // Random dense tests

    for(std::size_t n : {4, 8, 16, 32, 64, 128}) {
        std::vector<double> A = generate_random_dense_matrix(-1.0, 1.0, n);
        std::vector<double> x_true = generate_random_vector(-1.0, 1.0, n);

        if(!run_known_solution(A, x_true, n, pivot_tol, residual_tol, solution_tol)){
            failed_tests.push_back("random_dense_n_" + std::to_string(n));
        }
    }

    for(std::size_t n : {512, 1024}) {
        std::vector<double> A = generate_random_dense_matrix(-1.0, 1.0, n);
        std::vector<double> x_true = generate_random_vector(-1.0, 1.0, n);

        if(!run_known_solution(A, x_true, n, pivot_tol, residual_tol, solution_tol)){
            failed_tests.push_back("heavy_dense_n_" + std::to_string(n));
        }
    }


    // Singular and near-singular failure tests

    {
        std::vector<double> A = make_singular_multiple_row_3x3();

        if(!run_expected_factorization_failure(A, 3, pivot_tol)){
            failed_tests.push_back("singular_multiple_row_3x3");
        }
    }

    {
        std::vector<double> A = make_near_singular_3x3();
        std::vector<double> x_true = {1.0, 2.0, -1.0};

        if(!run_known_solution(A, x_true, 3, pivot_tol, 1e-7, 1e-7)){
            failed_tests.push_back("near_singular_3x3");
        }
    }

    {
        std::vector<double> A = make_zero_row_3x3();

        if(!run_expected_factorization_failure(A, 3, pivot_tol)){
            failed_tests.push_back("zero_row_3x3");
        }
    }

    {
        std::vector<double> A = make_zero_column_3x3();

        if(!run_expected_factorization_failure(A, 3, pivot_tol)){
            failed_tests.push_back("zero_column_3x3");
        }
    }

    {
        std::vector<double> A = make_diagonal_4x4();
        std::vector<double> x_true = {1.0, -2.0, 3.0, 0.5};

        if(!run_known_solution(A, x_true, 4, pivot_tol, residual_tol, solution_tol)){
            failed_tests.push_back("diagonal_4x4");
        }
    }

    {
        std::vector<double> A = make_upper_triangular_4x4();
        std::vector<double> x_true = {2.0, -1.0, 0.5, 3.0};

        if(!run_known_solution(A, x_true, 4, pivot_tol, residual_tol, solution_tol)){
            failed_tests.push_back("upper_triangular_4x4");
        }
    }

    for(std::size_t n : {512, 1024}){
        auto A = generate_random_dense_matrix(-1.0, 1.0, n);
        auto x = generate_random_vector(-1.0, 1.0, n);
        if(!run_known_solution(A, x, n, pivot_tol, /*residual_tol=*/1e-6, /*solution_tol=*/1e-6)){
            failed_tests.push_back("heavy_dense_n_relaxed_tol_" + std::to_string(n));
        }
    }


    // Edge/Stress Cases


    // 1×1 matrix – trivial boundary, must not crash or return wrong answer.
    {
        auto A = make_1x1(7.5);
        std::vector<double> x = {3.0};
        if(!run_known_solution(A, x, 1, pivot_tol, residual_tol, solution_tol)){
            failed_tests.push_back("n1_trivial"); 
        }
    }
 
    // 1×1 near-zero – should fail factorization.
    {   
        auto A = make_1x1(1e-14);
        if(!run_expected_factorization_failure(A, 1, pivot_tol)){
            failed_tests.push_back("n1_near_zero_should_fail"); 
        }
    }
 
    // Lower triangular – exercises forward substitution directly with no pivoting.
    { 
        auto A = make_lower_triangular_4x4();
        std::vector<double> x = {1.0, -1.0, 2.0, 0.5};
        if(!run_known_solution(A, x, 4, pivot_tol, residual_tol, solution_tol)){
            failed_tests.push_back("lower_triangular_4x4"); 
        }
    }
 
    // Permutation matrix – highly structured, maximum pivoting activity.
    { 
        auto A = make_permutation_4x4();
        std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
        if(!run_known_solution(A, x, 4, pivot_tol, residual_tol, solution_tol)){
            failed_tests.push_back("permutation_4x4"); 
        }
    }

    // Hilbert matrices – canonical ill-conditioning benchmark.
    // We only check the backward residual (not solution error) because
    // the solution itself is numerically unreliable for n >= 8 in double.
    // A correct solver should still satisfy ||b - A*x||/(||A||*||x||+||b||) < 1e-6.
    std::cout << "\n[Hilbert matrix tests — backward residual only]\n";
    for(std::size_t n : {4, 6, 8}){
        auto H  = make_hilbert_matrix(n);
        auto x  = std::vector<double>(n, 1.0);   // x_true = all-ones
        std::vector<double> b(n);
        matrix_vector_mul(H, x, b, n);
        if(!run_residual_only(H, b, n, pivot_tol, /*residual_tol=*/1e-6)){
            failed_tests.push_back("hilbert_n_" + std::to_string(n));
        } else {
            std::cout << "  hilbert n=" << n << "  PASS\n";
        }
    }

    // A correct relative residual formula is scale-invariant.
    // Multiply a well-conditioned 8×8 system by 1e12 and 1e-12; residuals
    // must remain the same as the unscaled version.
    {
        const std::size_t n = 8;
        auto A_base = generate_random_diagonally_dominant_matrix(-1.0, 1.0, n, 1.0);
        auto x_true = generate_random_vector(-1.0, 1.0, n);
 
        for(double scale : {1e12, 1e-12}){
            auto A_scaled = scale_matrix(A_base, scale);
            // x_true is the same; b = A_scaled * x_true = scale * (A_base * x_true)
            if(!run_known_solution(A_scaled, x_true, n, pivot_tol, residual_tol, solution_tol)){
                failed_tests.push_back("scale_" + std::to_string((int)std::log10(scale)) + "_n8");
            }
        }
    }

    // Multiple RHS vectors (one factorization, many solves)
    //
    // This is how LU is used in practice.  We factorize once, then solve
    // for 5 different right-hand sides and verify each solution independently.
    {
        std::cout << "\n[Multiple RHS test (n=32, 5 vectors)]\n";
        const std::size_t n = 32;
        auto A = generate_random_diagonally_dominant_matrix(-1.0, 1.0, n, 1.0);
 
        // Factorize once
        std::vector<double> LU = A;
        std::vector<std::size_t> piv(n);
        std::iota(piv.begin(), piv.end(), 0);
 
        bool fac_ok = (LU_factorize(LU, piv, n, pivot_tol) == LU_factorize_Status::Success);
        if(!fac_ok){
            failed_tests.push_back("multi_rhs_factorize");
        } else {
            bool all_pass = true;
            for(std::size_t k = 0; k < 5; k++){
                auto x_true = generate_random_vector(-1.0, 1.0, n);
                std::vector<double> b(n), x_computed;
                matrix_vector_mul(A, x_true, b, n);
 
                if(LU_solve(LU, piv, b, x_computed, n, pivot_tol) != LU_Solve_Status::Success){
                    all_pass = false; break;
                }
                double err = relative_inf_error(x_true, x_computed, n);
                if(err > solution_tol){ all_pass = false; break; }
            }
            if(!all_pass) failed_tests.push_back("multi_rhs_solve");
            else std::cout << "  all 5 RHS vectors solved correctly\n";
        }
    }

    // n=0 must return GeneralFailure immediately.
    {
        std::vector<double> LU;
        std::vector<std::size_t> piv;
        auto status = LU_factorize(LU, piv, 0, pivot_tol);
        if(status != LU_factorize_Status::GeneralFailure){
            failed_tests.push_back("contract_n0");
        } else {
            std::cout << "  n=0 -> GeneralFailure  PASS\n";
        }
    }

    // Mismatched array size: LU.size() != n*n.
    {
        std::vector<double> LU(6);   // 6 != 3*3 = 9
        std::vector<std::size_t> piv(3);
        std::iota(piv.begin(), piv.end(), 0);
        auto status = LU_factorize(LU, piv, 3, pivot_tol);
        if(status != LU_factorize_Status::GeneralFailure){
            failed_tests.push_back("contract_mismatched_size");
        } else {
            std::cout << "  mismatched array size -> GeneralFailure  PASS\n";
        }
    }
 
    // eps=0 must return GeneralFailure.
    {
        auto A = make_known_dense_3x3();
        std::vector<double> LU = A;
        std::vector<std::size_t> piv(3);
        std::iota(piv.begin(), piv.end(), 0);
        auto status = LU_factorize(LU, piv, 3, /*eps=*/0.0);
        if(status != LU_factorize_Status::GeneralFailure){
            failed_tests.push_back("contract_eps_zero");
        } else {
            std::cout << "  eps=0 -> GeneralFailure  PASS\n";
        }
    }
 
    // eps<0 must return GeneralFailure.
    {
        auto A = make_known_dense_3x3();
        std::vector<double> LU = A;
        std::vector<std::size_t> piv(3);
        std::iota(piv.begin(), piv.end(), 0);
        auto status = LU_factorize(LU, piv, 3, /*eps=*/-1.0);
        if(status != LU_factorize_Status::GeneralFailure){
            failed_tests.push_back("contract_eps_negative");
        } else {
            std::cout << "  eps<0 -> GeneralFailure  PASS\n";
        }
    }

    {
        std::cout << "\n[Eigen PartialPivLU comparison tests — diagonally dominant]\n";

        const double eigen_residual_tol = 1e-8;
        const double eigen_solution_tol = 1e-8;
        const double eigen_compare_tol  = 1e-8;

        for(std::size_t n : {16, 32, 64, 128, 256, 512}) {
            auto A = generate_random_diagonally_dominant_matrix(-1.0, 1.0, n, 1.0);
            auto x_true = generate_random_vector(-1.0, 1.0, n);

            if(!run_eigen_comparison(A, x_true, n, pivot_tol, eigen_residual_tol, eigen_solution_tol, eigen_compare_tol)) {
                failed_tests.push_back("eigen_comparison_diagonally_dominant_n_" + std::to_string(n));
            } else {
                std::cout << "  eigen comparison n=" << n << "  PASS\n";
            }
        }
    }

    {
        std::vector<double> LU;
        std::vector<std::size_t> piv;

        // Simulates an invalid signed-to-unsigned conversion.
        // The solver should reject it because the input sizes do not match n*n.

        std::size_t bad_n = static_cast<std::size_t>(-1);

        auto status = LU_factorize(LU, piv, bad_n, pivot_tol);

        if (status != LU_factorize_Status::GeneralFailure) {
            failed_tests.push_back("contract_invalid_large_n");
        } else {
            std::cout << "  invalid large n -> GeneralFailure  PASS\n";
        }
    }

    if(failed_tests.empty()) {
        std::cout << "All tests passed.\n";
        return 0;
    }

    std::cout << failed_tests.size() << " test(s) failed:\n";
    for(const auto& test_name : failed_tests) {
        std::cout << " - " << test_name << "\n";
    }    

    return 1;
}