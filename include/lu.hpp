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
