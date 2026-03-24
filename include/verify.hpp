#pragma once
#include <vector>

enum class NormType { Infinity, Frobenius };

// Norms
double fro_norm(const std::vector<double>& A, int n);
double inf_norm(const std::vector<double>& A, int n);

// Naive helpers (for verification only)
void matmul(const std::vector<double>& A,
            const std::vector<double>& B,
            std::vector<double>& C,
            int n);

void matsub(const std::vector<double>& A,
            const std::vector<double>& B,
            std::vector<double>& C,
            int n);

// Apply permutation vector to rows of A (computes PA)
void apply_piv_rows(const std::vector<double>& A,
                    const std::vector<int>& piv,
                    std::vector<double>& PA,
                    int n);


void computeLU(const std::vector<double>& A, std::vector<double>& LU, int n);

// Factorization residual: ||PA - LU|| / ||A||
double residual_factorization_debug(const std::vector<double>& A0, std::vector<double>& LU_Prod, 
const std::vector<int>& piv, std::vector<double>& PA, const std::vector<double>& LU_fac,
 std::vector<double>& diff, int n, NormType norm = NormType::Infinity);

double residual_factorization_fast(const std::vector<double>& A0,
                                   const std::vector<double>& LU_fac,
                                   const std::vector<int>& piv,
                                   int n,
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
                      int n,
                      NormType norm = NormType::Infinity);