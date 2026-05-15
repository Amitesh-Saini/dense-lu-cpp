// File purpose:
//   Declares verification utilities for the LU project.
//   This header provides matrix/vector norms, simple dense matrix operations,
//   permutation helpers, LU extraction, and residual checks for factorization,
//   single-RHS solves, and multiple-RHS solves.


#pragma once
#include <cstddef>
#include <vector>


enum class NormType {
    Infinity,
    Frobenius
};

// Function:
//   Frobenius norm calculator.
// What it does:
//   Computes the Frobenius norm of an n by n flat row-major matrix.
double fro_norm(const std::vector<double>& A, std::size_t n);

// Function:
//   Infinity norm calculator.
// What it does:
//   Computes the matrix infinity norm, meaning the maximum absolute row sum of an n by n flat row-major matrix.
double inf_norm(const std::vector<double>& A, std::size_t n);

// Function:
//   Naive dense matrix multiplication helper.
// What it does:
//   Computes C = A * B for n by n flat row-major matrices, mainly for verification rather than optimized performance.
void matmul(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, std::size_t n);

// Function:
//   Naive dense matrix subtraction helper.
// What it does:
//   Computes C = A - B for n by n flat row-major matrices, mainly for residual and verification calculations.
void matsub(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, std::size_t n);

// Function:
//   Row-permutation application helper.
// What it does:
//   Applies the pivot vector to the rows of A and writes the permuted matrix PA.
void apply_piv_rows(const std::vector<double>& A, const std::vector<std::size_t>& piv, std::vector<double>& PA, std::size_t n);

// Function:
//   Explicit LU-product builder.
// What it does:
//   Reconstructs the product L*U from the packed in-place LU storage for verification.
void compute_L_times_U(const std::vector<double>& A, std::vector<double>& LU, std::size_t n);

// Function:
//   Multiple-RHS column extraction helper.
// What it does:
//   Copies column rhs_index from the column-major multiple-RHS block B into a normal RHS vector b.
std::vector<double> extract_rhs_column(const std::vector<double>& B, std::size_t n, std::size_t nrhs, std::size_t rhs_index);

// Function:
//   Multiple-RHS solution storage helper.
// What it does:
//   Writes one computed solution vector into column rhs_index of the column-major solution block X_computed.
void store_solution_column(std::vector<double>& X_computed, const std::vector<double>& x_computed, std::size_t n, std::size_t nrhs, std::size_t rhs_index);

// Function:
//   Detailed factorization residual calculator.
// What it does:
//   Computes ||PA - LU|| / ||A|| while using caller-provided work buffers so intermediate matrices can be inspected.
double residual_factorization_debug(
    const std::vector<double>& A0, std::vector<double>& LU_Prod, const std::vector<std::size_t>& piv, std::vector<double>& PA, 
    const std::vector<double>& LU_fac, std::vector<double>& diff, std::size_t n, NormType norm = NormType::Infinity);

// Function:
//   Fast factorization residual calculator.
// What it does:
//   Computes ||PA - LU|| / ||A|| using internal temporary storage for compact verification code.
double residual_factorization_fast(
    const std::vector<double>& A0, const std::vector<double>& LU_fac, const std::vector<std::size_t>& piv, 
    std::size_t n, NormType norm = NormType::Infinity);

// Function:
//   Single-RHS solve residual calculator.
// What it does:
//   Computes a backward-error-style relative residual measuring how well the computed x satisfies A*x = b.
double residual_solve(
    const std::vector<double>& A0, const std::vector<double>& x, const std::vector<double>& b, 
    std::size_t n, NormType norm = NormType::Infinity);

// Function:
//   Multiple-RHS solve residual calculator.
// What it does:
//   Computes relative residuals for A*X_computed = B across all RHS columns using the selected norm.
std::vector<double> multiple_rhs_solve_residuals(
    const std::vector<double>& A, const std::vector<double>& X_computed, const std::vector<double>& B, 
    std::size_t n, std::size_t nrhs, NormType norm);