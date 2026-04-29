// lu.hpp
// Purpose:
//   Declare the public API for LU factorization and solving.
//
// Suggested API shape (you decide):
//   - bool lu_factorize_inplace(Matrix& A, std::vector<std::size_t>& piv, double tol)
//       Produces PA = LU stored in A (in-place LU) and records pivoting in 'piv'.
//   - void lu_solve(const Matrix& LU, const std::vector<std::size_t>& piv,
//                   const std::vector<double>& b, std::vector<double>& x)
//       Solves Ax=b using the factorization.
//
// Also consider exposing:
//   - helper routines for forward/back substitution
//   - residual computation utilities for testing

#pragma once
#include <cstddef>
#include <vector>

// Row-major element access helper (optional but recommended)
inline double& at(std::vector<double>& array, std::size_t n, std::size_t i, std::size_t j) {
    return array[i*n + j];
}

inline double at(const std::vector<double>& array, std::size_t n, std::size_t i, std::size_t j) {
    return array[i*n + j];
}

// Your existing helpers (generalized to nxn flat vector storage)
void Printarray(const std::vector<double>& array, std::size_t n);
void Printvec(const std::vector<double>& array, std::size_t n);
void Printpiv(const std::vector<std::size_t>& array, std::size_t n);
void rowswap(std::vector<double>& array, std::size_t rw_1, std::size_t rw_2, std::size_t n);

// Your existing LU step functions (names preserved)
bool pivot(std::vector<double>& array,
           std::vector<std::size_t>& piv,
           std::size_t rw,
           std::size_t n,
           double eps);

bool gaussian_eliminate(std::vector<double>& array, std::size_t rw, std::size_t n, double eps);

enum class LU_factorize_Status {
    Success,
    PivotFailure,
    EliminationFailure,
    GeneralFailure
};

// Top-level factorization wrapper (calls pivot + eliminate in a loop)
LU_factorize_Status LU_factorize(std::vector<double>& array,
                                  std::vector<std::size_t>& piv,
                                  std::size_t n,
                                  double eps);

bool forward_substitution(const std::vector<double>& LU,
                          const std::vector<std::size_t>& piv,
                          const std::vector<double>& b,
                          std::vector<double>& c,
                          std::size_t n);

bool back_substitution(const std::vector<double>& LU,
                       const std::vector<double>& c,
                       std::vector<double>& x,
                       std::size_t n,
                       double eps);

enum class LU_Solve_Status {
    Success,
    ForwardSubstitutionFailure,
    BackwardSubstitutionFailure,
    GeneralFailure
};

LU_Solve_Status LU_solve(const std::vector<double>& LU,
                         const std::vector<std::size_t>& piv,
                         const std::vector<double>& b,
                         std::vector<double>& x,
                         std::size_t n,
                         double eps);