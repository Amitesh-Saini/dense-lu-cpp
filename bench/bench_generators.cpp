// bench_generators.cpp
// Purpose:
//   Implement benchmark problem-generation utilities.
//
// Implementation notes:
//   - Uses seeded random generation so benchmark cases are reproducible.
//   - Builds matrix families that test normal performance, pivoting behavior,
//     ill-conditioning, near-singularity, and expected factorization failure.
//   - Constructs known-solution systems by generating x_true and forming b = A*x_true.
//   - Provides multiple-RHS helpers using column-major RHS storage.

#include "bench_generators.hpp"
#include "lu.hpp"
#include "verify.hpp"
#include "test_lu_utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

// Problem setup helpers

std::vector<double> make_pivot_stress_matrix(std::size_t n){


    if (n == 0) {
        throw std::invalid_argument("make_pivot_stress_matrix: n must be greater than 0");
    }

    std::vector<double> A(n * n, 0.0);

    // Start with identity.
    for (std::size_t i = 0; i < n; ++i) {
        A[i * n + i] = 1.0;
    }

    if (n >= 2) {
        // Force the first pivot to be bad in row 0.
        A[0 * n + 0] = 1e-16;

        // Put a strong pivot below it, requiring a row swap.
        A[1 * n + 0] = 1.0;

        // Add some harmless off-diagonal structure.
        A[0 * n + 1] = 2.0;
        A[1 * n + 1] = 3.0;
    }

    return A;
}


std::vector<double> make_near_singular_matrix(std::size_t n){

    if (n == 0) {
        throw std::invalid_argument("make_near_singular_matrix: n must be greater than 0");
    }

    std::vector<double> A(n * n, 0.0);

    // Start with identity.
    for (std::size_t i = 0; i < n; ++i) {
        A[i * n + i] = 1.0;
    }

    if (n >= 2) {
        // Make row 1 almost equal to row 0, but not exactly.
        for (std::size_t j = 0; j < n; ++j) {
            A[1 * n + j] = A[0 * n + j];
        }

        A[1 * n + 1] += 1e-12;
    }

    return A;
}


std::vector<double> make_singular_matrix(std::size_t n){


    if (n == 0) {
        throw std::invalid_argument("make_singular_matrix: n must be greater than 0");
    }

    std::vector<double> A(n * n, 0.0);

    // Start with identity.
    for (std::size_t i = 0; i < n; ++i) {
        A[i * n + i] = 1.0;
    }

    if (n >= 2) {
        // Make row 1 exactly equal to row 0.
        for (std::size_t j = 0; j < n; ++j) {
            A[1 * n + j] = A[0 * n + j];
        }
    }

    return A;
}

std::vector<double> generate_seeded_random_dense_matrix(double lower_bound, double upper_bound, std::size_t n, std::uint32_t seed){

    if (n == 0) {
        throw std::invalid_argument("generate_seeded_random_dense_matrix: n must be greater than 0");
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);

    std::vector<double> A(n * n);

    for (std::size_t i = 0; i < n * n; ++i) {
        A[i] = dist(gen);
    }

    return A;
}

std::vector<double> generate_seeded_random_vector(double lower_bound, double upper_bound, std::size_t n, std::uint32_t seed){

    if (n == 0) {
        throw std::invalid_argument("generate_seeded_random_vector: n must be greater than 0");
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);

    std::vector<double> x(n);

    for (std::size_t i = 0; i < n; ++i) {
        x[i] = dist(gen);
    }

    return x;
}

std::vector<double> generate_seeded_random_diagonally_dominant_matrix(
    double lower_bound, double upper_bound, std::size_t n, double margin, std::uint32_t seed){

    if (n == 0) {
        throw std::invalid_argument("generate_seeded_random_diagonally_dominant_matrix: n must be greater than 0");
    }

    if (margin <= 0.0) {
        throw std::invalid_argument("generate_seeded_random_diagonally_dominant_matrix: margin must be positive");
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);

    std::vector<double> A(n * n);

    for (std::size_t i = 0; i < n; ++i) {
        double non_diagonal_sum = 0.0;

        for (std::size_t j = 0; j < n; ++j) {
            A[i * n + j] = dist(gen);

            if (i != j) {
                non_diagonal_sum += std::abs(A[i * n + j]);
            }
        }

        const double sign = (dist(gen) < 0.0) ? -1.0 : 1.0;
        A[i * n + i] = sign * (non_diagonal_sum + margin);
    }

    return A;
}


std::vector<double> make_multiple_x_true(std::size_t n, std::size_t nrhs, double lower, double upper, std::uint32_t seed){

    if (n == 0) {
        throw std::invalid_argument("make_multiple_x_true: n must be greater than 0");
    }

    if (nrhs == 0) {
        throw std::invalid_argument("make_multiple_x_true: nrhs must be greater than 0");
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(lower, upper);

    std::vector<double> X_true(n * nrhs);

    for(std::size_t i = 0; i < n * nrhs; ++i){

        X_true[i] = dist(gen);
    }

    return X_true;
}


void make_multiple_rhs(const std::vector<double>& A, const std::vector<double>& X_true, std::vector<double>& B, std::size_t n, std::size_t nrhs){

    if (n == 0) {
        throw std::invalid_argument("make_multiple_rhs: n must be greater than 0");
    }

    if (nrhs == 0) {
        throw std::invalid_argument("make_multiple_rhs: nrhs must be greater than 0");
    }

    if (A.size() != n * n) {
        throw std::invalid_argument("make_multiple_rhs: A size mismatch");
    }

    if (X_true.size() != n * nrhs) {
        throw std::invalid_argument("make_multiple_rhs: X_true size mismatch");
    }

    if (B.size() != n * nrhs) {
        throw std::invalid_argument("make_multiple_rhs: B size mismatch");
    }

    for(std::size_t i = 0; i < nrhs; ++i){

        for(std::size_t j = 0; j < n; ++j){

            double inner_prod = 0.0;

            for(std::size_t k = 0; k < n; ++k){

                inner_prod += at(A, n, j, k) * X_true[k + i*n];
            }

            B[j + i*n] = inner_prod;
        }
    }
}


double multiple_rhs_solution_error_inf(const std::vector<double>& X_true, const std::vector<double>& X_computed, std::size_t n, std::size_t nrhs){

    if (X_true.size() != n * nrhs || X_computed.size() != n * nrhs) {
        throw std::invalid_argument("multiple_rhs_solution_error_inf: size mismatch");
    }

    if (n == 0) {
        throw std::invalid_argument("multiple_rhs_solution_error_inf: n must be greater than 0");
    }

    if (nrhs == 0) {
        throw std::invalid_argument("multiple_rhs_solution_error_inf: nrhs must be greater than 0");
    }

    return relative_inf_error(X_true, X_computed, n * nrhs);

}




double multiple_rhs_solution_error_l2(const std::vector<double>& X_true, const std::vector<double>& X_computed, std::size_t n, std::size_t nrhs){

    if (n == 0) {
        throw std::invalid_argument("multiple_rhs_solution_error_l2: n must be greater than 0");
    }

    if (nrhs == 0) {
        throw std::invalid_argument("multiple_rhs_solution_error_l2: nrhs must be greater than 0");
    }

    if (X_true.size() != n * nrhs || X_computed.size() != n * nrhs) {
        throw std::invalid_argument("multiple_rhs_solution_error_l2: size mismatch");
    }

    return relative_l2_error(X_true, X_computed, n*nrhs);

}






std::vector<double> make_benchmark_matrix(std::size_t n, MatrixType type, std::uint32_t seed){


    if (n == 0) {
        throw std::invalid_argument("make_benchmark_matrix: n must be greater than 0");
    }

    // Fixed benchmark-generation defaults.
    // Random dense and diagonally dominant matrices use the same value range
    // so benchmark cases are controlled and comparable.
    constexpr double lower = -1.0;
    constexpr double upper = 1.0;
    constexpr double diagonal_dominance_margin = 1.0;

    switch (type) {
        case MatrixType::RandomDense:
            return generate_seeded_random_dense_matrix(lower, upper, n, seed);

        case MatrixType::DiagonallyDominant:
            return generate_seeded_random_diagonally_dominant_matrix(lower, upper, n, diagonal_dominance_margin, seed);

        case MatrixType::Hilbert:
            return make_hilbert_matrix(n);

        case MatrixType::PivotStress:
            return make_pivot_stress_matrix(n);

        case MatrixType::NearSingular:
            return make_near_singular_matrix(n);

        case MatrixType::Singular:
            return make_singular_matrix(n);
    }

    throw std::invalid_argument("make_benchmark_matrix: unknown MatrixType");
}


std::vector<double> make_benchmark_vector(std::size_t n, double lower, double upper, std::uint32_t seed){

    if(n == 0) {
        throw std::invalid_argument("make_benchmark_vector: n must be greater than 0");
    }

    return generate_seeded_random_vector(lower, upper, n, seed);
}


std::vector<std::size_t> make_identity_pivot(std::size_t n){

     if(n == 0) {
        throw std::invalid_argument("make_identity_pivot: n must be greater than 0");
    }

    std::vector<std::size_t> piv(n);

    std::iota(piv.begin(), piv.end(), 0);

    return piv;

}





LUProblem make_lu_problem(std::size_t n, MatrixType matrix_type, std::uint32_t seed, double eps){


    if (n == 0) {
        throw std::invalid_argument("make_lu_problem: n must be greater than 0");
    }

    if (eps <= 0.0) {
        throw std::invalid_argument("make_lu_problem: eps must be positive");
    }

    constexpr double lower = -1.0;
    constexpr double upper = 1.0;

    LUProblem problem;

    problem.A0 = make_benchmark_matrix(n, matrix_type, seed);

    // Use a different seed from the matrix so x_true is independently generated.
    problem.x_true = make_benchmark_vector(n, lower, upper, seed + 1);

    problem.b.resize(n);
    matrix_vector_mul(problem.A0, problem.x_true, problem.b, n);

    problem.piv = make_identity_pivot(n);

    problem.eps = eps;
    problem.matrix_type = matrix_type;
    problem.n = n;
    problem.seed = seed;

    return problem;
}