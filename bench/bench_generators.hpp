#pragma once

#include "bench_types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>


// Problem setup helpers


// Creates a matrix that forces partial pivoting.
// What it does:
//   Builds a simple matrix where the first pivot is tiny/invalid and a better
//   pivot exists below it.
// Why needed:
//   Tests that partial pivoting is actually being exercised.


std::vector<double> make_pivot_stress_matrix(std::size_t n);

// Creates a nearly singular matrix.
// What it does:
//   Builds a matrix with two nearly dependent rows.
// Why needed:
//   Stress-tests numerical tolerance and stability behavior.

std::vector<double> make_near_singular_matrix(std::size_t n);

// Creates a singular matrix.
// What it does:
//   Builds a matrix with two identical rows.
// Why needed:
//   Tests that LU factorization fails correctly on singular systems.

std::vector<double> make_singular_matrix(std::size_t n);


// creates random dense matrix using seed

std::vector<double> generate_seeded_random_dense_matrix(double lower_bound, double upper_bound, std::size_t n, std::uint32_t seed);


// creates random diagonally dominant dense matrix using seed

std::vector<double> generate_seeded_random_diagonally_dominant_matrix(double lower_bound, double upper_bound, std::size_t n,
 double margin, std::uint32_t seed);

// generate random seeded vector 

std::vector<double> generate_seeded_random_vector(double lower_bound, double upper_bound, std::size_t n, std::uint32_t seed);



// Creates a seeded matrix of true solutions X_true for multiple RHS.
// What it does:
//   Generates an n-by-nrhs matrix stored column-major, where column r is one
//   true solution vector x_true_r.
// Why needed:
//   Multiple-RHS benchmarks need known exact solutions so B = A*X_true can be
//   generated and solution error can be measured.

std::vector<double> make_multiple_x_true(std::size_t n, std::size_t nrhs, double lower, double upper, std::uint32_t seed);

// Computes B = A*X_true for multiple RHS.
// What it does:
//   Multiplies A, stored row-major as A[i*n + j], by X_true, stored column-major
//   as X_true[j + rhs*n], and writes B in the same column-major RHS layout.
// Why needed:
//   Builds known right-hand sides for A*X = B benchmarks.
// Note if nrhs = n this will be O(N^3) so keep n = 8 16 32 64 when n is large

void make_multiple_rhs(const std::vector<double>& A, const std::vector<double>& X_true, std::vector<double>& B, std::size_t n, std::size_t nrhs);


 // Computes relative infinity-norm solution error for multiple RHS.
// What it does:
//   Returns ||X_true - X_computed||_inf / ||X_true||_inf over the full RHS block.
// Why needed:
//   Measures worst-entry solution error across all RHS solves.

double multiple_rhs_solution_error_inf(const std::vector<double>& X_true, const std::vector<double>& X_computed, std::size_t n, std::size_t nrhs);

// Computes relative L2/Frobenius-style solution error for multiple RHS.
// What it does:
//   Returns ||X_true - X_computed||_F / ||X_true||_F.
// Why needed:
//   Measures total solution error across all RHS solves.

double multiple_rhs_solution_error_l2(const std::vector<double>& X_true, const std::vector<double>& X_computed, std::size_t n, std::size_t nrhs);



// Creates a benchmark matrix of the requested type.
// What it does:
//   Dispatches to your existing matrix generators based on MatrixType.
//   For example, RandomDense should call your existing random dense generator,
//   DiagonallyDominant should call your existing diagonally dominant generator,
//   and Hilbert should call your existing Hilbert generator.
// Why needed:
//   Keeps bench_lu.cpp independent of the details of matrix generation.

std::vector<double> make_benchmark_matrix(std::size_t n, MatrixType type, std::uint32_t seed);



// Creates a benchmark vector.
// What it does:
//   Dispatches to your existing random vector generator.
// Why needed:
//   Used to create x_true before forming b = A*x_true.

std::vector<double> make_benchmark_vector(std::size_t n, double lower, double upper, std::uint32_t seed);



// Creates the identity pivot vector [0, 1, 2, ..., n-1].
// What it does:
//   Allocates and initializes the pivot vector before LU factorization.
// Why needed:
//   Your LU factorization expects the pivot vector to already exist and have
//   valid initial row indices.

std::vector<std::size_t> make_identity_pivot(std::size_t n);



// Creates a full LU benchmark problem.
// What it does:
//   Builds A0, x_true, b = A0*x_true, pivot vector, size, tolerance, matrix
//   label, and seed.
// Why needed:
//   All benchmark phases need the same clean problem setup, and this avoids
//   repeating setup code in every benchmark driver.


LUProblem make_lu_problem(std::size_t n, MatrixType matrix_type, std::uint32_t seed, double eps);
