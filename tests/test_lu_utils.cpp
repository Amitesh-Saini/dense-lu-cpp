#include <cmath>
#include <iostream>
#include <random>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>
#include <cstddef>
#include <vector>
#include <Eigen/Dense>


#include "lu.hpp"
#include "verify.hpp"
#include "test_lu_utils.hpp"


static std::mt19937& test_rng()
{
    static std::mt19937 gen(123456789);
    return gen;
}





void matrix_vector_mul(const std::vector<double>& A, const std::vector<double>& x, std::vector<double>& b, std::size_t n){

    b.resize(n);
    
    for(std::size_t i = 0; i < n; i++){
        
        double dot_prod = 0.0;

        for(std::size_t j = 0; j < n; j++){
            
            dot_prod += A[i*n+j] * x[j];
        }

        b[i] = dot_prod;
    }
}

double relative_l2_error(const std::vector<double>& x_true,
                         const std::vector<double>& x_computed,
                         std::size_t n)
{
    double numerator = 0.0;
    double denominator = 0.0;

    for(std::size_t i = 0; i < n; i++) {
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
                          std::size_t n)
{
    double numerator_max = 0.0;
    double denominator_max = 0.0;

    for(std::size_t i = 0; i < n; i++) {
        numerator_max = std::max(numerator_max, std::abs(x_computed[i] - x_true[i]));
        denominator_max = std::max(denominator_max, std::abs(x_true[i]));
    }

    if(denominator_max == 0.0) {
        return numerator_max;
    }

    return numerator_max / denominator_max;
}

std::vector<double> generate_random_dense_matrix(double lower_bound, double upper_bound, std::size_t n){


    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);
    auto& gen = test_rng();

    std::vector<double> A(n*n);

    for(std::size_t i = 0; i < n*n; i++){
        A[i] = dist(gen);
    }

    return A;
}

std::vector<double> generate_random_vector(double lower_bound, double upper_bound, std::size_t n){

    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);
    auto& gen = test_rng();

    std::vector<double> x(n);

    for(std::size_t i = 0; i < n; i++){
        x[i] = dist(gen);
    }

    return x;
}

std::vector<double> generate_random_diagonally_dominant_matrix(double lower_bound, double upper_bound, std::size_t n, double margin){

    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);
    auto& gen = test_rng();

    std::vector<double> A(n*n);

    for(std::size_t i = 0; i < n; i++){

        double non_diagonal_sum = 0.0;

        for(std::size_t j = 0; j < n; j++){

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

bool run_known_solution(const std::vector<double>& A, const std::vector<double>& x_true, std::size_t n, double pivot_tol, double residual_tol, double solution_tol){
    
    
    std::vector<double> LU = A;         
    std::vector<double> b(n);
    std::vector<std::size_t> piv(n);
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

bool run_residual_only(const std::vector<double>& A, const std::vector<double>& b, std::size_t n, double pivot_tol, double residual_tol){

    std::vector<double> LU = A;
    std::vector<std::size_t> piv(n);
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
                                        std::size_t n,
                                        double pivot_tol)
{
    std::vector<double> LU = A;
    std::vector<std::size_t> piv(n);

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

std::vector<double> make_identity_matrix(std::size_t n)
{
    std::vector<double> I(n * n, 0.0);
    for(std::size_t i = 0; i < n; i++) {
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

std::vector<double> make_hilbert_matrix(std::size_t n)
{
    if (n == 0) {
        throw std::invalid_argument("make_hilbert_matrix: n must be greater than 0");
    }

    std::vector<double> H(n * n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            H[i * n + j] = 1.0 / static_cast<double>(i + j + 1);
        }
    }

    return H;
}
 
std::vector<double> make_lower_triangular_4x4()
{
    return { 3.0,  0.0,  0.0,  0.0,
            -1.0,  5.0,  0.0,  0.0,
             2.0, -3.0,  7.0,  0.0,
             1.0,  4.0, -2.0,  4.0 };
}
 
std::vector<double> make_permutation_4x4()
{
    return { 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0,
             1.0, 0.0, 0.0, 0.0 };
}
 
std::vector<double> make_1x1(double val) { return {val}; }
 
std::vector<double> scale_matrix(std::vector<double> A, double s)
{
    for(auto& v : A) v *= s;
    return A;
}


bool run_eigen_comparison(const std::vector<double>& A, const std::vector<double>& x_true, std::size_t n, double pivot_tol, double residual_tol, double solution_tol, double eigen_compare_tol)
{
    std::vector<double> LU = A;
    std::vector<double> b(n);
    std::vector<std::size_t> piv(n);
    std::vector<double> x_my;
    std::vector<double> x_eigen(n);

    matrix_vector_mul(A, x_true, b, n);

    std::iota(piv.begin(), piv.end(), 0);

    LU_factorize_Status fac_status = LU_factorize(LU, piv, n, pivot_tol);
    if(fac_status != LU_factorize_Status::Success) {
        std::cout << "Eigen comparison test: your factorization failed for n = " << n << "\n";
        return false;
    }

    LU_Solve_Status solve_status = LU_solve(LU, piv, b, x_my, n, pivot_tol);
    if(solve_status != LU_Solve_Status::Success) {
        std::cout << "Eigen comparison test: your solve failed for n = " << n << "\n";
        return false;
    }

    Eigen::MatrixXd A_eigen(n, n);
    Eigen::VectorXd b_eigen(n);

    for(std::size_t i = 0; i < n; ++i) {
        b_eigen(static_cast<Eigen::Index>(i)) = b[i];

        for(std::size_t j = 0; j < n; ++j) {
            A_eigen(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = A[i*n + j];
        }
    }

    Eigen::PartialPivLU<Eigen::MatrixXd> eigen_lu(A_eigen);
    Eigen::VectorXd eigen_solution = eigen_lu.solve(b_eigen);

    for(std::size_t i = 0; i < n; ++i) {
        x_eigen[i] = eigen_solution(static_cast<Eigen::Index>(i));
    }

    double my_l2              = relative_l2_error(x_true, x_my, n);
    double my_inf             = relative_inf_error(x_true, x_my, n);

    double eigen_l2           = relative_l2_error(x_true, x_eigen, n);
    double eigen_inf          = relative_inf_error(x_true, x_eigen, n);

    double my_vs_eigen_l2     = relative_l2_error(x_eigen, x_my, n);
    double my_vs_eigen_inf    = relative_inf_error(x_eigen, x_my, n);

    double my_res_inf         = residual_solve(A, x_my, b, n, NormType::Infinity);
    double my_res_frob        = residual_solve(A, x_my, b, n, NormType::Frobenius);

    double eigen_res_inf      = residual_solve(A, x_eigen, b, n, NormType::Infinity);
    double eigen_res_frob     = residual_solve(A, x_eigen, b, n, NormType::Frobenius);

    bool pass =
        my_l2           < solution_tol &&
        my_inf          < solution_tol &&
        eigen_l2        < solution_tol &&
        eigen_inf       < solution_tol &&
        my_vs_eigen_l2  < eigen_compare_tol &&
        my_vs_eigen_inf < eigen_compare_tol &&
        my_res_inf      < residual_tol &&
        my_res_frob     < residual_tol &&
        eigen_res_inf   < residual_tol &&
        eigen_res_frob  < residual_tol;

    if(!pass) {
        std::cout << std::setprecision(16);
        std::cout << "Eigen comparison test failed for n = " << n << "\n";
        std::cout << "my_l2              = " << my_l2              << "\n";
        std::cout << "my_inf             = " << my_inf             << "\n";
        std::cout << "eigen_l2           = " << eigen_l2           << "\n";
        std::cout << "eigen_inf          = " << eigen_inf          << "\n";
        std::cout << "my_vs_eigen_l2     = " << my_vs_eigen_l2     << "\n";
        std::cout << "my_vs_eigen_inf    = " << my_vs_eigen_inf    << "\n";
        std::cout << "my_res_inf         = " << my_res_inf         << "\n";
        std::cout << "my_res_frob        = " << my_res_frob        << "\n";
        std::cout << "eigen_res_inf      = " << eigen_res_inf      << "\n";
        std::cout << "eigen_res_frob     = " << eigen_res_frob     << "\n";
    }

    return pass;
}

