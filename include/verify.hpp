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