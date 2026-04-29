
#include <cmath>
#include <iostream>
#include <random>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>
#include <cstddef>
#include <vector>

#include "lu.hpp"
#include "verify.hpp"


#pragma once



// Multiplies a dense n x n matrix A by a vector x and stores the result in b.
// This is useful in the tests because we usually choose a known x_true first,
// then construct b = A*x_true so the solver has a known correct answer to recover.
void matrix_vector_mul(const std::vector<double>& A, const std::vector<double>& x, std::vector<double>& b, std::size_t n);

// Computes the relative L2 error between the known solution and the computed solution.
// This is useful because it measures the total accumulated solution error across all
// entries instead of only checking one component.
double relative_l2_error(const std::vector<double>& x_true, const std::vector<double>& x_computed, std::size_t n);

// Computes the relative infinity-norm error between the known solution and the computed solution.
// This is useful because it catches the worst individual entry error in the computed solution.
double relative_inf_error(const std::vector<double>& x_true, const std::vector<double>& x_computed, std::size_t n);

// Generates a random dense n x n matrix with entries sampled uniformly between lower_bound and upper_bound.
// This is useful for stress testing the LU factorization on general dense matrices without special structure.
std::vector<double> generate_random_dense_matrix(double lower_bound, double upper_bound, std::size_t n);

// Generates a random vector of length n with entries sampled uniformly between lower_bound and upper_bound.
// This is useful for creating random x_true vectors before forming b = A*x_true.
std::vector<double> generate_random_vector(double lower_bound, double upper_bound, std::size_t n);

// Generates a random diagonally dominant n x n matrix.
// This is useful because diagonal dominance usually gives more numerically stable systems,
// letting the tests check correctness over large sizes without unnecessary ill-conditioning.
std::vector<double> generate_random_diagonally_dominant_matrix(double lower_bound, double upper_bound, std::size_t n, double margin);

// Runs a full known-solution test using A and x_true.
// It forms b = A*x_true, factorizes A, solves Ax=b, then checks factorization residuals,
// solve residuals, relative L2 solution error, and relative infinity solution error.
// The default residual and solution tolerances are both 1e-10.
bool run_known_solution(const std::vector<double>& A, const std::vector<double>& x_true, std::size_t n, double pivot_tol, double residual_tol = 1e-10, double solution_tol = 1e-10);

// Runs a residual-only solve test.
// This is useful for ill-conditioned matrices such as Hilbert matrices, where the exact
// solution error may be large even if the backward residual is still acceptable.
bool run_residual_only(const std::vector<double>& A, const std::vector<double>& b, std::size_t n, double pivot_tol, double residual_tol);

// Runs a test where factorization is expected to fail.
// This is useful for singular matrices, zero-row matrices, zero-column matrices,
// and near-zero pivot cases where the solver should reject the input.
bool run_expected_factorization_failure(const std::vector<double>& A, std::size_t n, double pivot_tol);

// Builds a fixed dense 3 x 3 matrix with a known solution.
// This is useful as a small deterministic correctness test for the basic LU solve path.
std::vector<double> make_known_dense_3x3();

// Builds a 3 x 3 matrix that requires row pivoting because the first pivot is zero.
// This is useful for confirming that partial pivoting is actually being exercised.
std::vector<double> make_pivot_required_3x3();

// Builds a singular 3 x 3 matrix with a repeated row.
// This is useful for checking that the factorization detects exact linear dependence.
std::vector<double> make_singular_repeated_row_3x3();

// Builds an n x n identity matrix.
// This is useful as a simple sanity test because solving Ix=b should return x=b.
std::vector<double> make_identity_matrix(std::size_t n);

// Builds a fixed diagonal 4 x 4 matrix.
// This is useful for testing a simple non-identity system where the correct solve behavior is easy to verify.
std::vector<double> make_diagonal_4x4();

// Builds a fixed upper triangular 4 x 4 matrix.
// This is useful for exercising the backward-substitution part of the solve routine.
std::vector<double> make_upper_triangular_4x4();

// Builds a singular 3 x 3 matrix with an all-zero row.
// This is useful for checking that the solver rejects a matrix with no full-rank structure.
std::vector<double> make_zero_row_3x3();

// Builds a singular 3 x 3 matrix with an all-zero column.
// This is useful for checking that the pivot search fails when a pivot column is unusable.
std::vector<double> make_zero_column_3x3();

// Builds a singular 3 x 3 matrix where one row is a scalar multiple of another.
// This is useful for testing linear dependence that is less trivial than an exactly repeated row.
std::vector<double> make_singular_multiple_row_3x3();

// Builds a near-singular 3 x 3 matrix.
// This is useful for testing numerically difficult behavior where the tolerance may need to be relaxed.
std::vector<double> make_near_singular_3x3();

// Builds an n x n Hilbert matrix H(i,j) = 1/(i+j+1).
// This is useful because Hilbert matrices are famously ill-conditioned, so they test backward residual behavior.
std::vector<double> make_hilbert_matrix(std::size_t n);

// Builds a fixed lower triangular 4 x 4 matrix.
// This is useful for exercising the forward-substitution behavior of the solve routine.
std::vector<double> make_lower_triangular_4x4();

// Builds a fixed 4 x 4 permutation matrix.
// This is useful because it heavily exercises the pivoting path using a structured matrix.
std::vector<double> make_permutation_4x4();

// Builds a 1 x 1 matrix containing val.
// This is useful for testing the smallest valid matrix size and ensuring the solver does not crash.
std::vector<double> make_1x1(double val);

// Scales every entry of matrix A by scalar s and returns the scaled copy.
// This is useful for checking that relative residuals behave correctly under very large or very small scaling.
std::vector<double> scale_matrix(std::vector<double> A, double s);

// Runs an external reference comparison against Eigen's PartialPivLU solver.
// The test forms b = A*x_true, solves the system using your LU implementation,
// solves the same system using Eigen, then compares:
// your solution vs x_true, Eigen's solution vs x_true, your solution vs Eigen,
// your backward residual, and Eigen's backward residual.
// Returns false if any residual or solution comparison exceeds its tolerance.
bool run_eigen_comparison(const std::vector<double>& A, const std::vector<double>& x_true, std::size_t n, double pivot_tol, double residual_tol, double solution_tol, double eigen_compare_tol);