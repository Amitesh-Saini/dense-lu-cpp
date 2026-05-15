// verify.cpp
// Purpose:
//   Implement verification utilities for LU factorization and solve accuracy.
//
// Implementation notes:
//   - Computes matrix norms and residuals.
//   - Verifies PA = LU factorization quality.
//   - Verifies single-RHS and multiple-RHS solve accuracy.
//   - Uses dense row-major storage consistent with the LU implementation.

#include "verify.hpp"
#include "lu.hpp"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

// Helpers

void matmul(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, std::size_t n){

    if (n == 0) {
        throw std::invalid_argument("matmul: n must be greater than 0");
    }

    C.resize(n*n);

    for(std::size_t i = 0; i < n; ++i){
        for(std::size_t j = 0; j < n; ++j){

            at(C, n, i, j) = 0.0;

            for(std::size_t k = 0; k < n; ++k){

                at(C, n, i, j) += at(A, n, i, k)*at(B, n, k, j);
            }
        }    
    }
}

void matsub(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, std::size_t n){

    if (n == 0) {
        throw std::invalid_argument("matsub: n must be greater than 0");
    }

    C.resize(n*n);

    for(std::size_t i = 0; i < n; ++i){
        for(std::size_t j = 0; j < n; ++j) at(C, n, i, j) = at(A, n, i, j) - at(B, n, i, j);
    }
}

void apply_piv_rows(const std::vector<double>& A, const std::vector<std::size_t>& piv, std::vector<double>& PA, std::size_t n){


    if (n == 0) {
        throw std::invalid_argument("apply_piv_rows: n must be greater than 0");
    }

    PA.resize(n*n);

    for(std::size_t i = 0; i < n; i++){
        for(std::size_t j = 0; j < n; j++){

            at(PA, n, i, j) = at(A, n, piv[i], j);
        }
    }
}

double fro_norm(const std::vector<double>& A, std::size_t n){

    if (n == 0) {
        throw std::invalid_argument("fro_norm: n must be greater than 0");
    }

    double sum = 0.0;

    for(std::size_t i = 0; i < n*n; i++) sum += A[i]*A[i];

    return std::sqrt(sum);
}

double inf_norm(const std::vector<double>& A, std::size_t n){

    if (n == 0) {
        throw std::invalid_argument("inf_norm: n must be greater than 0");
    }

    double max = 0.0;

    for(std::size_t i = 0; i < n; i++){

        double sum = 0; 

        for(std::size_t j = 0; j < n; j++) sum += std::abs(A[i*n + j]);
        
        if(sum > max) max = sum;
    }

    return max;
}

void compute_L_times_U(const std::vector<double>& A, std::vector<double>& LU, std::size_t n){

    if (n == 0) {
        throw std::invalid_argument("compute_L_times_U: n must be greater than 0");
    }

    double L = 0.0;
    double U = 0.0;

    LU.resize(n*n);

   
    for(std::size_t i = 0; i < n; ++i){
        for(std::size_t j = 0; j < n; ++j){

            double sum = 0.0;

            for(std::size_t k = 0; k <= std::min(i,j); ++k){

                if(k == i) L = 1.0;
                else if (k < i) L = at(A, n, i, k);
                else L = 0.0;

                if(k <= j) U = at(A, n, k, j);
                else U = 0.0;

                sum += L*U;
            }

            LU[i*n+j] = sum;
        }
    }
}



std::vector<double> extract_rhs_column(const std::vector<double>& B, std::size_t n, std::size_t nrhs, std::size_t rhs_index){

    if (n == 0) {
        throw std::invalid_argument("extract_rhs_column: n must be greater than 0");
    }

    if (nrhs == 0) {
        throw std::invalid_argument("extract_rhs_column: nrhs must be greater than 0");
    }

    if (rhs_index >= nrhs) {
        throw std::invalid_argument("extract_rhs_column: rhs_index out of range");
    }

    if (B.size() != n * nrhs) {
        throw std::invalid_argument("extract_rhs_column: B size mismatch");
    }

    std::vector<double> b(n);

    for(std::size_t i = 0; i < n; ++i){

        b[i] = B[n * rhs_index + i];
    }

    return b;
}

void store_solution_column(std::vector<double>& X_computed, const std::vector<double>& x_computed, std::size_t n, std::size_t nrhs, std::size_t rhs_index){


    if (n == 0) {
        throw std::invalid_argument("store_solution_column: n must be greater than 0");
    }

    if (nrhs == 0) {
        throw std::invalid_argument("store_solution_column: nrhs must be greater than 0");
    }

    if (rhs_index >= nrhs) {
        throw std::invalid_argument("store_solution_column: rhs_index out of range");
    }

    if (X_computed.size() != n * nrhs) {
        throw std::invalid_argument("store_solution_column: X_computed size mismatch");
    }

    if (x_computed.size() != n) {
        throw std::invalid_argument("store_solution_column: x_computed size mismatch");
    }

    for(std::size_t i = 0; i < n; ++i){

        X_computed[n * rhs_index + i] = x_computed[i];
    }
}



double residual_factorization_debug(
    const std::vector<double>& A0, std::vector<double>& LU_Prod, const std::vector<std::size_t>& piv, std::vector<double>& PA, 
    const std::vector<double>& LU_fac, std::vector<double>& diff, std::size_t n, NormType norm){

    if (n == 0) {
        throw std::invalid_argument("residual_factorization_debug: n must be greater than 0");
    }

    if(norm != NormType::Frobenius && norm != NormType::Infinity) {
        throw std::invalid_argument("Norm not selected");
    }
    
    apply_piv_rows(A0, piv, PA, n);                 // calculate P times A
    compute_L_times_U(LU_fac, LU_Prod, n);          // calculate L times U

    diff.resize(n*n);

    matsub(PA, LU_Prod, diff, n);       // PA - LU

    const double eps_abs = 1e-15;
    double denom = (norm== NormType::Infinity ? inf_norm(A0,n) : fro_norm(A0,n));
    double num = (norm== NormType::Infinity ? inf_norm(diff,n) : fro_norm(diff,n));

    if (denom <= eps_abs) {
        std::cout << "FAIL: ||A|| is zero or near zero; cannot compute meaningful relative residual.\n";
        return std::numeric_limits<double>::infinity();
    }

    return num/denom;
}

double residual_factorization_fast(
    const std::vector<double>& A0, const std::vector<double>& LU_fac, const std::vector<std::size_t>& piv,
    std::size_t n, NormType norm){

    if (n == 0) {
        throw std::invalid_argument("residual_factorization_fast: n must be greater than 0");
    }


    if(norm != NormType::Frobenius && norm != NormType::Infinity) {
        throw std::invalid_argument("Norm not selected");
    }
    
    double L = 0.0;
    double U = 0.0;

    double max_row_sum = 0.0;

    double sum_Total = 0.0;

    for(std::size_t i = 0; i < n; ++i){

        double row_sum = 0.0;

        for(std::size_t j = 0; j < n; ++j){

            double PA_ij = at(A0, n, piv[i], j);

            double diff = 0.0;

            double sum = 0.0;

            for(std::size_t k = 0; k <= std::min(i,j); ++k){

                if(k == i) L = 1.0;
                else if (k < i) L = at(LU_fac, n, i, k);
                else L = 0.0;

                if(k <= j) U = at(LU_fac, n, k, j);
                else U = 0.0; 

                sum += L*U;
            }

            diff = PA_ij - sum;

            sum_Total += diff*diff;

            row_sum += std::abs(diff);

        }

        if(row_sum > max_row_sum) max_row_sum = (row_sum);
    }

    const double eps_abs = 1e-15;
    double denom = (norm== NormType::Infinity ? inf_norm(A0,n) : fro_norm(A0,n));
    double num = (norm== NormType::Infinity ? max_row_sum : std::sqrt(sum_Total));

    if (denom <= eps_abs) {
        std::cout << "FAIL: ||A|| is zero or near zero; cannot compute meaningful relative residual.\n";
        return std::numeric_limits<double>::infinity();
    }

    return num/denom;
}

// Relative solve residual (backward-error style):
//     ||b - A*x|| / ( ||A||*||x|| + ||b|| )
//
// Uses the selected norm:
// - Infinity:  ||b - A*x||_inf / ( ||A||_inf * ||x||_inf + ||b||_inf )
// - Frobenius: ||b - A*x||_2   / ( ||A||_F   * ||x||_2   + ||b||_2   )
//
// Measures how well the computed solution x satisfies A*x = b.
double residual_solve(
    const std::vector<double>& A0, const std::vector<double>& x, const std::vector<double>& b, 
    std::size_t n, NormType norm) {

    if (n == 0) {
        throw std::invalid_argument("residual_solve: n must be greater than 0");
    }


    const double eps_abs = 1e-15;


    if(norm == NormType::Infinity){

        double max_x_entry = 0.0;
        double max_b_entry = 0.0;
        double max_num_entry = 0.0;
        double A_inf = 0.0;

        for(std::size_t i = 0; i < n; ++i){

            double Ax_i = 0.0;
            double A0_row_sum = 0.0;

            for(std::size_t j = 0; j < n; ++j){

                Ax_i += at(A0, n, i, j) * x[j];
                A0_row_sum += std::abs(at(A0, n, i, j));
            }

            if(std::abs(b[i] - Ax_i) > max_num_entry) max_num_entry = std::abs(b[i] - Ax_i);
            if(A0_row_sum > A_inf) A_inf = A0_row_sum;
            if(std::abs(x[i]) > max_x_entry) max_x_entry = std::abs(x[i]);
            if(std::abs(b[i]) > max_b_entry) max_b_entry = std::abs(b[i]);
        }

        double denominator = A_inf * max_x_entry + max_b_entry;

        if(denominator <= eps_abs) return std::numeric_limits<double>::infinity();

        return (max_num_entry / denominator);

    }

    else if(norm == NormType::Frobenius){

        double A_sum = 0.0;
        double x_sum = 0.0;
        double b_sum = 0.0;
        double numerator = 0.0;

        for(std::size_t i = 0; i < n; ++i){

            double Ax_i = 0.0;

            for(std::size_t j = 0; j < n; ++j){

                Ax_i += (at(A0, n, i, j) * x[j]);
                A_sum += at(A0, n, i, j) * at(A0, n, i, j);
            }

            x_sum += x[i] * x[i];
            b_sum += b[i] * b[i];
            numerator += (b[i] - Ax_i) * (b[i] - Ax_i);
        }


        double denominator = (std::sqrt(A_sum) * std::sqrt(x_sum) + std::sqrt(b_sum));

        if(denominator <= eps_abs) return std::numeric_limits<double>::infinity();

        return(std::sqrt(numerator) / denominator);
    }


    else{
        throw std::invalid_argument("Norm not selected");
    }
}


std::vector<double> multiple_rhs_solve_residuals(
    const std::vector<double>& A, const std::vector<double>& X_computed, const std::vector<double>& B, 
    std::size_t n, std::size_t nrhs, NormType norm){

      if (n == 0) {
        throw std::invalid_argument("multiple_rhs_solve_residual: n must be greater than 0");
    }

    if (nrhs == 0) {
        throw std::invalid_argument("multiple_rhs_solve_residual: nrhs must be greater than 0");
    }

    if (A.size() != n * n) {
        throw std::invalid_argument("multiple_rhs_solve_residual: A size mismatch");
    }

    if (X_computed.size() != n * nrhs) {
        throw std::invalid_argument("multiple_rhs_solve_residual: X_computed size mismatch");
    }

    if (B.size() != n * nrhs) {
        throw std::invalid_argument("multiple_rhs_solve_residual: B size mismatch");
    }

    std::vector<double> residual_values(nrhs);


    std::vector<double> x_computed(n);
    std::vector<double> b(n);

    for(std::size_t i = 0; i < nrhs; ++i){

        x_computed = extract_rhs_column(X_computed, n, nrhs, i);
        b = extract_rhs_column(B, n, nrhs, i);

        residual_values[i] = residual_solve(A, x_computed, b, n, norm);

    }

    return residual_values;

}