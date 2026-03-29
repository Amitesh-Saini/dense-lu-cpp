// test_lu.cpp
// Purpose:
//   Unit tests for LU factorization + solve.
//
// What to test:
//   - Factorization correctness via residual:
//       ||P*A - L*U|| / ||A||
//   - Solve correctness:
//       ||A*x - b|| / ||b||
//   - Edge cases:
//       * singular matrices should fail factorization
//       * small sizes (2x2, 3x3) with known answers
//       * random matrices across several sizes
//
// Wire this into CTest via CMake.


//cmake --build build
//./build/test_lu


#include "lu.hpp"
#include "verify.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>


void matrix_vector_mul(const std::vector<double>& A, const std::vector<double>& x, std::vector<double>& b, int n){
    
    for(int i = 0; i<n; i++){
        
        double dot_prod = 0.0;

        for(int j = 0; j<n; j++){
            
            dot_prod += A[i*n+j] * x[j];
        }

        b[i] = dot_prod;
    }
}


double relative_l2_error(const std::vector<double>& x_true,
                         const std::vector<double>& x_computed,
                         int n)
{
    double numerator = 0.0;
    double denominator = 0.0;

    for(int i = 0; i < n; i++) {
        const double diff = x_computed[i] - x_true[i];
        numerator += diff * diff;
        denominator += x_true[i] * x_true[i];
    }

    numerator = std::sqrt(numerator);
    denominator = std::sqrt(denominator);

    if(denominator == 0.0) {
        return numerator;  
    }

    return numerator / denominator;
}


double relative_inf_error(const std::vector<double>& x_true,
                          const std::vector<double>& x_computed,
                          int n)
{
    double numerator_max = 0.0;
    double denominator_max = 0.0;

    for(int i = 0; i < n; i++) {
        numerator_max = std::max(numerator_max, std::abs(x_computed[i] - x_true[i]));
        denominator_max = std::max(denominator_max, std::abs(x_true[i]));
    }

    if(denominator_max == 0.0) {
        return numerator_max;
    }

    return numerator_max / denominator_max;
}



std::vector<double> generate_random_dense_matrix(double lower_bound, double upper_bound, int n){

    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);

    std::vector<double> A(n*n);

    for(int i = 0; i< n*n; i++){
        A[i] = dist(gen);
    }

    return A;
}

std::vector<double> generate_random_vector(double lower_bound, double upper_bound, int n){

    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);

    std::vector<double> x(n);

    for(int i = 0; i< n; i++){
        x[i] = dist(gen);
    }

    return x;
}

std::vector<double> generate_random_diagonally_dominant_matrix(double lower_bound, double upper_bound, int n, double margin){

    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);

    std::vector<double> A(n*n);

    for(int i = 0; i < n; i++){

        double non_diagonal_sum = 0.0;

        for(int j = 0; j<n; j++){

            A[i*n+j] = dist(gen);

            if(i != j){
                non_diagonal_sum += std::abs(A[i*n+j]);
            }
        }

        double sign = (dist(gen) < 0.0) ? -1.0 : 1.0;
        A[i*n+i] = sign * (non_diagonal_sum + margin);
    }

    return A;
}


bool run_known_solution(const std::vector<double>& A, const std::vector<double>& x_true, int n, double pivot_tol, double residual_tol = 1e-10, double solution_tol = 1e-10){
    
    
    std::vector<double> LU = A;         
    std::vector<double> b(n);
    std::vector<int> piv(n);
    std::vector<double> x_computed;

    matrix_vector_mul(A, x_true, b, n);

    std::iota(piv.begin(), piv.end(), 0);

    LU_factorize_Status fac_status = LU_factorize(LU, piv, n, pivot_tol);
    if(fac_status != LU_factorize_Status::Success) {
        std::cout << "Factorization failed for n = " << n << "\n";
        return false;
    }

    LU_Solve_Status solve_status = LU_solve(LU, piv, b, x_computed, n, pivot_tol);
    if(solve_status != LU_Solve_Status::Success) {
        std::cout << "Solve failed for n = " << n << "\n";
        return false;
    }

    double factorization_inf  = residual_factorization_fast(A, LU, piv, n, NormType::Infinity);
    double factorization_frob = residual_factorization_fast(A, LU, piv, n, NormType::Frobenius);

    double solve_inf  = residual_solve(A, x_computed, b, n, NormType::Infinity);
    double solve_frob = residual_solve(A, x_computed, b, n, NormType::Frobenius);

    double rel_l2  = relative_l2_error(x_true, x_computed, n);
    double rel_inf = relative_inf_error(x_true, x_computed, n);

    bool pass =
        factorization_inf  < residual_tol &&
        factorization_frob < residual_tol &&
        solve_inf          < residual_tol &&
        solve_frob         < residual_tol &&
        rel_l2             < solution_tol &&
        rel_inf            < solution_tol;

    if(!pass) {
        std::cout << std::setprecision(16);
        std::cout << "Known-solution test failed for n = " << n << "\n";
        std::cout << "factorization_inf  = " << factorization_inf  << "\n";
        std::cout << "factorization_frob = " << factorization_frob << "\n";
        std::cout << "solve_inf          = " << solve_inf          << "\n";
        std::cout << "solve_frob         = " << solve_frob         << "\n";
        std::cout << "relative_l2        = " << rel_l2             << "\n";
        std::cout << "relative_inf       = " << rel_inf            << "\n";
    }


    return pass;

}

// Variant that only checks residuals (no x_true available, e.g. Hilbert matrix)
bool run_residual_only(const std::vector<double>& A, const std::vector<double>& b, int n, double pivot_tol, double residual_tol){

    std::vector<double> LU = A;
    std::vector<int>    piv(n);
    std::vector<double> x_computed;
 
    std::iota(piv.begin(), piv.end(), 0);
 
    if(LU_factorize(LU, piv, n, pivot_tol) != LU_factorize_Status::Success) return false;
    if(LU_solve(LU, piv, b, x_computed, n, pivot_tol) != LU_Solve_Status::Success) return false;
 
    double sol_inf  = residual_solve(A, x_computed, b, n, NormType::Infinity);
    double sol_frob = residual_solve(A, x_computed, b, n, NormType::Frobenius);
 
    bool pass = sol_inf < residual_tol && sol_frob < residual_tol;
    if(!pass)
        std::cout << "  FAIL residual_only n=" << n
                  << "  sol_inf=" << sol_inf
                  << "  sol_frob=" << sol_frob << "\n";
    return pass;
}

bool run_expected_factorization_failure(const std::vector<double>& A,
                                        int n,
                                        double pivot_tol)
{
    std::vector<double> LU = A;
    std::vector<int> piv(n);

    std::iota(piv.begin(), piv.end(), 0);

    return LU_factorize(LU, piv, n, pivot_tol) != LU_factorize_Status::Success;
}


std::vector<double> make_known_dense_3x3()
{
    return {
         2.0,  1.0, -1.0,
        -3.0, -1.0,  2.0,
        -2.0,  1.0,  2.0
    };
}

std::vector<double> make_pivot_required_3x3()
{
    return {
        0.0,  2.0,  3.0,
        4.0,  5.0,  6.0,
        7.0,  8.0, 10.0
    };
}

std::vector<double> make_singular_repeated_row_3x3()
{
    return {
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };
}

std::vector<double> make_identity_matrix(int n)
{
    std::vector<double> I(n * n, 0.0);
    for(int i = 0; i < n; i++) {
        I[i * n + i] = 1.0;
    }
    return I;
}

std::vector<double> make_diagonal_4x4()
{
    return {
        4.0, 0.0, 0.0, 0.0,
        0.0, 7.0, 0.0, 0.0,
        0.0, 0.0, 2.5, 0.0,
        0.0, 0.0, 0.0, 9.0
    };
}

std::vector<double> make_upper_triangular_4x4()
{
    return {
        2.0, -1.0,  3.0,  4.0,
        0.0,  5.0, -2.0,  1.0,
        0.0,  0.0,  6.0, -3.0,
        0.0,  0.0,  0.0,  8.0
    };
}

std::vector<double> make_zero_row_3x3()
{
    return {
        1.0, 2.0, 3.0,
        0.0, 0.0, 0.0,
        4.0, 5.0, 6.0
    };
}

std::vector<double> make_zero_column_3x3()
{
    return {
        0.0, 1.0, 2.0,
        0.0, 3.0, 4.0,
        0.0, 5.0, 6.0
    };
}

std::vector<double> make_singular_multiple_row_3x3()
{
    return {
        1.0, 2.0, 3.0,
        2.0, 4.0, 6.0,
        4.0, 5.0, 6.0
    };
}

std::vector<double> make_near_singular_3x3()
{
    return {
        1.0, 1.0, 1.0,
        1.0, 1.0 + 1e-10, 1.0,
        1.0, 1.0, 1.0 + 1e-10
    };
}

// Hilbert matrix: H(i,j) = 1/(i+j+1), 0-indexed.
// Famously ill-conditioned: cond(H_8) ≈ 1.5e10.
// We don't use x_true here because the solution error is expected to
// be large; we only check that the backward residual ||b-Ax||/... is
// reasonable (say < 1e-6 for n<=8).
std::vector<double> make_hilbert_matrix(int n)
{
    std::vector<double> H(n*n);
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            H[i*n+j] = 1.0 / (i + j + 1.0);
    return H;
}
 
// Lower triangular 4×4 – tests forward substitution in isolation;
// no pivoting is applied so L is exercised directly.
std::vector<double> make_lower_triangular_4x4()
{
    return { 3.0,  0.0,  0.0,  0.0,
            -1.0,  5.0,  0.0,  0.0,
             2.0, -3.0,  7.0,  0.0,
             1.0,  4.0, -2.0,  4.0 };
}
 
// Permutation matrix that cycles [0,1,2,3] → [1,2,3,0].
// Exercises the pivot path on a perfectly structured matrix.
std::vector<double> make_permutation_4x4()
{
    return { 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0,
             1.0, 0.0, 0.0, 0.0 };
}
 
// 1×1 matrix – boundary case that must not crash.
std::vector<double> make_1x1(double val) { return {val}; }
 
// Scale a matrix by a scalar (used for the large/small entry tests).
std::vector<double> scale_matrix(std::vector<double> A, double s)
{
    for(auto& v : A) v *= s;
    return A;
}
 

int main()
{
    std::vector<std::string> failed_tests;

    const double pivot_tol    = 1e-12;
    const double residual_tol = 1e-10;
    const double solution_tol = 1e-10;

    std::cout << "=== Running LU solver tests ===\n\n";

    {
        std::vector<double> A = make_known_dense_3x3();
        std::vector<double> x_true = {2.0, 3.0, -1.0};

        if(!run_known_solution(A, x_true, 3, pivot_tol, residual_tol, solution_tol)) {
            failed_tests.push_back("known_dense_3x3");
        }
    }


    {
        std::vector<double> A = make_identity_matrix(3);
        std::vector<double> x_true = {1.0, -2.0, 4.0};

        if(!run_known_solution(A, x_true, 3, pivot_tol, residual_tol, solution_tol)) {
            failed_tests.push_back("identity_3x3");
        }
    }

    {
        std::vector<double> A = make_pivot_required_3x3();
        std::vector<double> x_true = {1.0, -1.0, 2.0};

        if(!run_known_solution(A, x_true, 3, pivot_tol, residual_tol, solution_tol)) {
            failed_tests.push_back("pivot_required_3x3");
        }
    }


    {
        std::vector<double> A = make_singular_repeated_row_3x3();

        if(!run_expected_factorization_failure(A, 3, pivot_tol)) {
            failed_tests.push_back("singular_repeated_row_3x3");
        }
    }

    for(int n : {4, 8, 16, 32, 64, 128, 256}) {
        std::vector<double> A = generate_random_diagonally_dominant_matrix(-1.0, 1.0, n, 1.0);
        std::vector<double> x_true = generate_random_vector(-1.0, 1.0, n);

        if(!run_known_solution(A, x_true, n, pivot_tol, residual_tol, solution_tol)) {
            failed_tests.push_back("random_diagonally_dominant_n_" + std::to_string(n));
        }
    }

    for(int n : {512, 1024}) {
        std::vector<double> A = generate_random_diagonally_dominant_matrix(-1.0, 1.0, n, 1.0);
        std::vector<double> x_true = generate_random_vector(-1.0, 1.0, n);

        if(!run_known_solution(A, x_true, n, pivot_tol, residual_tol, solution_tol)) {
            failed_tests.push_back("heavy_random_diagonally_dominant_n_" + std::to_string(n));
        }
    }

    for(int n : {4, 8, 16, 32, 64, 128}) {
        std::vector<double> A = generate_random_dense_matrix(-1.0, 1.0, n);
        std::vector<double> x_true = generate_random_vector(-1.0, 1.0, n);

        if(!run_known_solution(A, x_true, n, pivot_tol, residual_tol, solution_tol)) {
            failed_tests.push_back("random_dense_n_" + std::to_string(n));
        }
    }

    for(int n : {512, 1024}) {
        std::vector<double> A = generate_random_dense_matrix(-1.0, 1.0, n);
        std::vector<double> x_true = generate_random_vector(-1.0, 1.0, n);

        if(!run_known_solution(A, x_true, n, pivot_tol, residual_tol, solution_tol)) {
            failed_tests.push_back("heavy_dense_n_" + std::to_string(n));
        }
    }

    {
    std::vector<double> A = make_singular_multiple_row_3x3();

    if(!run_expected_factorization_failure(A, 3, pivot_tol)) {
        failed_tests.push_back("singular_multiple_row_3x3");
        }
    }

    {

    std::vector<double> A = make_near_singular_3x3();
    std::vector<double> x_true = {1.0, 2.0, -1.0};

    if(!run_known_solution(A, x_true, 3, pivot_tol, 1e-7, 1e-7)) {
        failed_tests.push_back("near_singular_3x3");
        }
    }

    {
    std::vector<double> A = make_zero_row_3x3();

    if(!run_expected_factorization_failure(A, 3, pivot_tol)) {
        failed_tests.push_back("zero_row_3x3");
        }
    }

    {
    std::vector<double> A = make_zero_column_3x3();

    if(!run_expected_factorization_failure(A, 3, pivot_tol)) {
        failed_tests.push_back("zero_column_3x3");
         }
    }

    {
    std::vector<double> A = make_diagonal_4x4();
    std::vector<double> x_true = {1.0, -2.0, 3.0, 0.5};

    if(!run_known_solution(A, x_true, 4, pivot_tol, residual_tol, solution_tol)) {
        failed_tests.push_back("diagonal_4x4");
       }
    }

    {
    std::vector<double> A = make_upper_triangular_4x4();
    std::vector<double> x_true = {2.0, -1.0, 0.5, 3.0};

    if(!run_known_solution(A, x_true, 4, pivot_tol, residual_tol, solution_tol)) {
        failed_tests.push_back("upper_triangular_4x4");
        }
    }

    for(int n : {512, 1024}){
        auto A = generate_random_dense_matrix(-1.0, 1.0, n);
        auto x = generate_random_vector(-1.0, 1.0, n);
        if(!run_known_solution(A, x, n, pivot_tol, /*residual_tol=*/1e-6, /*solution_tol=*/1e-6))
            failed_tests.push_back("heavy_dense_n_relaxed_tol" + std::to_string(n));
    }

    // 1×1 matrix – trivial boundary, must not crash or return wrong answer.
    { auto A = make_1x1(7.5);
      std::vector<double> x = {3.0};
      if(!run_known_solution(A, x, 1, pivot_tol, residual_tol, solution_tol))
          failed_tests.push_back("n1_trivial"); 
    }
 
    // 1×1 near-zero – should fail factorization.
    { auto A = make_1x1(1e-14);
      if(!run_expected_factorization_failure(A, 1, pivot_tol))
          failed_tests.push_back("n1_near_zero_should_fail"); 
    }
 
    // Lower triangular – exercises forward substitution directly with no pivoting.
    { auto A = make_lower_triangular_4x4();
      std::vector<double> x = {1.0, -1.0, 2.0, 0.5};
      if(!run_known_solution(A, x, 4, pivot_tol, residual_tol, solution_tol))
          failed_tests.push_back("lower_triangular_4x4"); 
    }
 
    // Permutation matrix – highly structured, maximum pivoting activity.
    { auto A = make_permutation_4x4();
      std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
      if(!run_known_solution(A, x, 4, pivot_tol, residual_tol, solution_tol))
          failed_tests.push_back("permutation_4x4"); 
    }

    // Hilbert matrices – canonical ill-conditioning benchmark.
    // We only check the backward residual (not solution error) because
    // the solution itself is numerically unreliable for n >= 8 in double.
    // A correct solver should still satisfy ||b - A*x||/(||A||*||x||+||b||) < 1e-6.
    std::cout << "\n[Hilbert matrix tests — backward residual only]\n";
    for(int n : {4, 6, 8}){
        auto H  = make_hilbert_matrix(n);
        auto x  = std::vector<double>(n, 1.0);   // x_true = all-ones
        std::vector<double> b(n);
        matrix_vector_mul(H, x, b, n);
        if(!run_residual_only(H, b, n, pivot_tol, /*residual_tol=*/1e-6))
            failed_tests.push_back("hilbert_n_" + std::to_string(n));
        else
            std::cout << "  hilbert n=" << n << "  PASS\n";
    }


    // A correct relative residual formula is scale-invariant.
    // Multiply a well-conditioned 8×8 system by 1e12 and 1e-12; residuals
    // must remain the same as the unscaled version.
    {
        const int n = 8;
        auto A_base = generate_random_diagonally_dominant_matrix(-1.0, 1.0, n, 1.0);
        auto x_true = generate_random_vector(-1.0, 1.0, n);
 
        for(double scale : {1e12, 1e-12}){
            auto A_scaled = scale_matrix(A_base, scale);
            // x_true is the same; b = A_scaled * x_true = scale * (A_base * x_true)
            if(!run_known_solution(A_scaled, x_true, n, pivot_tol, residual_tol, solution_tol))
                failed_tests.push_back("scale_" + std::to_string((int)std::log10(scale)) + "_n8");
        }
    }


    // ── NEW: multiple RHS vectors (one factorization, many solves) ───────
    //
    // This is how LU is used in practice.  We factorize once, then solve
    // for 5 different right-hand sides and verify each solution independently.
    {
        std::cout << "\n[Multiple RHS test (n=32, 5 vectors)]\n";
        const int n = 32;
        auto A = generate_random_diagonally_dominant_matrix(-1.0, 1.0, n, 1.0);
 
        // Factorize once
        std::vector<double> LU = A;
        std::vector<int>    piv(n);
        std::iota(piv.begin(), piv.end(), 0);
 
        bool fac_ok = (LU_factorize(LU, piv, n, pivot_tol) == LU_factorize_Status::Success);
        if(!fac_ok){
            failed_tests.push_back("multi_rhs_factorize");
        } else {
            bool all_pass = true;
            for(int k = 0; k < 5; k++){
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
        std::vector<int>    piv;
        auto status = LU_factorize(LU, piv, 0, pivot_tol);
        if(status != LU_factorize_Status::GeneralFailure)
            failed_tests.push_back("contract_n0");
        else
            std::cout << "  n=0 -> GeneralFailure  PASS\n";
    }


    // Mismatched array size: LU.size() != n*n.
    {
        std::vector<double> LU(6);   // 6 != 3*3 = 9
        std::vector<int>    piv(3);
        std::iota(piv.begin(), piv.end(), 0);
        auto status = LU_factorize(LU, piv, 3, pivot_tol);
        if(status != LU_factorize_Status::GeneralFailure)
            failed_tests.push_back("contract_mismatched_size");
        else
            std::cout << "  mismatched array size -> GeneralFailure  PASS\n";
    }
 
    // eps=0 must return GeneralFailure.
    {
        auto A = make_known_dense_3x3();
        std::vector<double> LU = A;
        std::vector<int>    piv(3);
        std::iota(piv.begin(), piv.end(), 0);
        auto status = LU_factorize(LU, piv, 3, /*eps=*/0.0);
        if(status != LU_factorize_Status::GeneralFailure)
            failed_tests.push_back("contract_eps_zero");
        else
            std::cout << "  eps=0 -> GeneralFailure  PASS\n";
    }
 
    // eps<0 must return GeneralFailure.
    {
        auto A = make_known_dense_3x3();
        std::vector<double> LU = A;
        std::vector<int>    piv(3);
        std::iota(piv.begin(), piv.end(), 0);
        auto status = LU_factorize(LU, piv, 3, /*eps=*/-1.0);
        if(status != LU_factorize_Status::GeneralFailure)
            failed_tests.push_back("contract_eps_negative");
        else
            std::cout << "  eps<0 -> GeneralFailure  PASS\n";
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