#pragma once
#include <cstddef>
#include <vector>

enum class NormType {
    Infinity,
    Frobenius
};

// Norms
double fro_norm(const std::vector<double>& A, std::size_t n);
double inf_norm(const std::vector<double>& A, std::size_t n);

// Naive helpers (for verification only)
void matmul(const std::vector<double>& A,
            const std::vector<double>& B,
            std::vector<double>& C,
            std::size_t n);

void matsub(const std::vector<double>& A,
            const std::vector<double>& B,
            std::vector<double>& C,
            std::size_t n);

// Apply permutation vector to rows of A (computes PA)
void apply_piv_rows(const std::vector<double>& A,
                    const std::vector<std::size_t>& piv,
                    std::vector<double>& PA,
                    std::size_t n);

void computeLU(const std::vector<double>& A, std::vector<double>& LU, std::size_t n);

// Extracts one RHS column from a column-major multiple-RHS block.
// What it does:
//   Copies column rhs_index from B into a normal std::vector<double> b.
// Why needed:
//   Your custom LU_solve currently solves one RHS vector at a time.

std::vector<double> extract_rhs_column(const std::vector<double>& B, std::size_t n, std::size_t nrhs, std::size_t rhs_index);

// Stores one computed solution vector into a column-major solution block.
// What it does:
//   Writes x_computed into column rhs_index of X_computed.
// Why needed:
//   Custom LU solves one RHS at a time, but accuracy checks need all computed
//   solution columns together.

void store_solution_column(std::vector<double>& X_computed, const std::vector<double>& x_computed, std::size_t n, std::size_t nrhs, std::size_t rhs_index);




// Factorization residual: ||PA - LU|| / ||A||
double residual_factorization_debug(const std::vector<double>& A0,
                                    std::vector<double>& LU_Prod,
                                    const std::vector<std::size_t>& piv,
                                    std::vector<double>& PA,
                                    const std::vector<double>& LU_fac,
                                    std::vector<double>& diff,
                                    std::size_t n,
                                    NormType norm = NormType::Infinity);

double residual_factorization_fast(const std::vector<double>& A0,
                                   const std::vector<double>& LU_fac,
                                   const std::vector<std::size_t>& piv,
                                   std::size_t n,
                                   NormType norm = NormType::Infinity);

// Relative solve residual (backward-error style):
//     ||b - A*x|| / ( ||A||*||x|| + ||b|| )
//
// Uses the selected norm:
// - Infinity:  ||b - A*x||_inf / ( ||A||_inf * ||x||_inf + ||b||_inf )
// - Frobenius: ||b - A*x||_2   / ( ||A||_F   * ||x||_2   + ||b||_2   )
//
// Measures how well the computed solution x satisfies A*x = b.
double residual_solve(const std::vector<double>& A0,
                      const std::vector<double>& x,
                      const std::vector<double>& b,
                      std::size_t n,
                      NormType norm = NormType::Infinity);


// Computes a multiple-RHS solve residual.
// What it does:
//   Computes a relative residual for A*X_computed = B over all RHS columns.
// Why needed:
//   Measures whether the computed multi-RHS solution satisfies the original
//   linear systems.

std::vector<double> multiple_rhs_solve_residuals(const std::vector<double>& A, const std::vector<double>& X_computed, const std::vector<double>& B,
 std::size_t n, std::size_t nrhs, NormType norm);