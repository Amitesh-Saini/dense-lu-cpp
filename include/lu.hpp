// lu.hpp
// Purpose:
//   Declare the public API for LU factorization and solving.
//
// Suggested API shape (you decide):
//   - bool lu_factorize_inplace(Matrix& A, std::vector<int>& piv, double tol)
//       Produces PA = LU stored in A (in-place LU) and records pivoting in 'piv'.
//   - void lu_solve(const Matrix& LU, const std::vector<int>& piv,
//                   const std::vector<double>& b, std::vector<double>& x)
//       Solves Ax=b using the factorization.
//
// Also consider exposing:
//   - helper routines for forward/back substitution
//   - residual computation utilities for testing


#pragma once
#include <vector>

// Row-major element access helper (optional but recommended)
inline double& at(std::vector<double>& array, int n, int i, int j) {
    return array[i*n + j];
}

inline double at(const std::vector<double>& array, int n, int i, int j) {
    return array[i*n + j];
}

// Your existing helpers (generalized to nxn flat vector storage)
void Printarray(const std::vector<double>& array, int n);
void Printvec(const std::vector<double>& array, int n);
void Printpiv(const std::vector<int>& array, int n);
void rowswap(std::vector<double>& array, int rw_1, int rw_2, int n);

// Your existing LU step functions (names preserved)
bool pivot(std::vector<double>& array,        // U (or LU if you later go in-place)
           std::vector<int>& piv,             // permutation vector (recommended)
           int rw,
           int n,
           double eps);

bool gaussian_eliminate(std::vector<double>& array, int rw, int n, double eps);

enum class LU_factorize_Status {
    Success,
    PivotFailure,
    EliminationFailure,
    GeneralFailure
};

// Top-level factorization wrapper (calls pivot + eliminate in a loop)
LU_factorize_Status LU_factorize(std::vector<double>& array,     // can be initialized as a copy of A
                  std::vector<int>& piv,
                  int n,
                  double eps);


bool forward_substitution(const std::vector<double>& LU,
                          const std::vector<int>& piv,
                          const std::vector<double>& b,
                          std::vector<double>& c,
                          int n);

bool back_substitution(const std::vector<double>& LU,
                       const std::vector<double>& c,
                       std::vector<double>& x,
                       int n, double eps);


enum class LU_Solve_Status {
    Success,
    ForwardSubstitutionFailure,
    BackwardSubstitutionFailure,
    GeneralFailure
};


LU_Solve_Status LU_solve(const std::vector<double>& LU,
              const std::vector<int>& piv,
              const std::vector<double>& b,
              std::vector<double>& x,
              int n,
              double eps);

