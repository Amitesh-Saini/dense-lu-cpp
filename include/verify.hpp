#pragma once
#include <vector>

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

// Factorization residual: ||PA - L*U|| / ||A||
double residual_factorization(const std::vector<double>& A,
                              const std::vector<double>& L,
                              const std::vector<double>& U,
                              const std::vector<int>& piv,
                              int n);

// Solve residual: ||A*x - b|| / ||b||  (or a slightly better normalization)
double residual_solve(const std::vector<double>& A,
                      const std::vector<double>& x,
                      const std::vector<double>& b,
                      int n);