// solve_demo.cpp
// Purpose:
//   Provide a minimal usage example for the custom LU solver.
//
// Implementation notes:
//   - Demonstrates solving one dense system Ax = b.
//   - Demonstrates factorization reuse for multiple right-hand sides.
//   - Uses flat row-major matrix storage, matching the solver API.
//   - Computes residuals and solution errors to verify the result.

#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "lu.hpp"
#include "verify.hpp"


// Convert a regular vector-of-rows matrix into flat row-major storage.
std::vector<double> make_row_major_matrix(const std::vector<std::vector<double>>& rows)
{
    if (rows.empty()) {
        return {};
    }

    const std::size_t n = rows.size();

    for (const auto& row : rows) {
        if (row.size() != n) {
            throw std::invalid_argument("make_row_major_matrix: matrix must be square.");
        }
    }

    std::vector<double> A;
    A.reserve(n * n);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A.push_back(rows[i][j]);
        }
    }

    return A;
}


// Compute b = A*x for a flat row-major n x n matrix A.
std::vector<double> matrix_vector_product(const std::vector<double>& A, const std::vector<double>& x, std::size_t n)
{
    if (A.size() != n * n) {
        throw std::invalid_argument("matrix_vector_product: A size mismatch.");
    }

    if (x.size() != n) {
        throw std::invalid_argument("matrix_vector_product: x size mismatch.");
    }

    std::vector<double> b(n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            b[i] += A[i * n + j] * x[j];
        }
    }

    return b;
}


// Compute max_i |x_i - y_i| for two vectors.
double max_abs_difference(const std::vector<double>& x, const std::vector<double>& y)
{
    if (x.size() != y.size()) {
        throw std::invalid_argument("max_abs_difference: vector size mismatch.");
    }

    double max_diff = 0.0;

    for (std::size_t i = 0; i < x.size(); ++i) {
        const double diff = std::abs(x[i] - y[i]);

        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    return max_diff;
}


// Print a vector on one line.
void print_vector(const std::string& name, const std::vector<double>& x)
{
    std::cout << name << " = [ ";

    for (double value : x) {
        std::cout << value << " ";
    }

    std::cout << "]\n";
}


// Print one RHS vector stored in column-major multiple-RHS layout:
// X[n * rhs_index + i].
void print_rhs_column(const std::string& name, const std::vector<double>& X, std::size_t n, std::size_t rhs_index)
{
    std::cout << name << " = [ ";

    for (std::size_t i = 0; i < n; ++i) {
        std::cout << X[n * rhs_index + i] << " ";
    }

    std::cout << "]\n";
}


std::string to_string_factorization_status(LU_factorize_Status status)
{
    switch (status) {
        case LU_factorize_Status::Success:
            return "success";

        case LU_factorize_Status::PivotFailure:
            return "pivot_failure";

        case LU_factorize_Status::EliminationFailure:
            return "elimination_failure";

        case LU_factorize_Status::GeneralFailure:
            return "general_failure";
    }

    return "unknown_factorization_status";
}


std::string to_string_solve_status(LU_Solve_Status status)
{
    switch (status) {
        case LU_Solve_Status::Success:
            return "success";

        case LU_Solve_Status::ForwardSubstitutionFailure:
            return "forward_substitution_failure";

        case LU_Solve_Status::BackwardSubstitutionFailure:
            return "backward_substitution_failure";

        case LU_Solve_Status::GeneralFailure:
            return "general_failure";
    }

    return "unknown_solve_status";
}


int main()
{
    const double eps = 1e-12;

    // ============================================================
    // Ex 1: Simple case with one right-hand side
    // ============================================================

    std::cout << "============================================================\n";
    std::cout << "Ex 1: Single RHS solve\n";
    std::cout << "============================================================\n";

    // 1. Define matrix A in normal row format, then convert it to flat row-major storage.
    std::vector<std::vector<double>> A_rows = {
        { 2.0,  1.0, -1.0 },
        {-3.0, -1.0,  2.0 },
        {-2.0,  1.0,  2.0 }
    };

    const std::size_t n = A_rows.size();

    std::vector<double> A0 = make_row_major_matrix(A_rows);

    // 2. Define a known solution x_true and compute b = A*x_true.
    std::vector<double> x_true = {2.0, 3.0, -1.0};

    std::vector<double> b = matrix_vector_product(A0, x_true, n);

    // 3. Make a working copy of A because LU_factorize modifies the matrix in place.
    std::vector<double> LU = A0;

    // 4. Initialize pivot vector as the identity permutation.
    std::vector<std::size_t> piv(n);
    std::iota(piv.begin(), piv.end(), std::size_t{0});

    // 5. Factor A into LU with partial pivoting.
    LU_factorize_Status factorize_status = LU_factorize(LU, piv, n, eps);

    std::cout << "Factorization status = " << to_string_factorization_status(factorize_status) << "\n";

    if (factorize_status != LU_factorize_Status::Success) {
        std::cerr << "Ex 1 failed: factorization did not succeed.\n";
        return 1;
    }

    // 6. Solve Ax = b using the computed LU factors.
    std::vector<double> x_computed(n, 0.0);

    LU_Solve_Status solve_status = LU_solve(LU, piv, b, x_computed, n, eps);

    std::cout << "Solve status         = " << to_string_solve_status(solve_status) << "\n";

    if (solve_status != LU_Solve_Status::Success) {
        std::cerr << "Ex 1 failed: solve did not succeed.\n";
        return 1;
    }

    // 7. Compute factorization residuals and solve residual.
    std::vector<double> PA(n * n, 0.0);
    std::vector<double> L_mul_U(n * n, 0.0);
    std::vector<double> diff(n * n, 0.0);

    double factorization_residual_debug_inf = residual_factorization_debug(A0, L_mul_U, piv, PA, LU, diff, n, NormType::Infinity);
    double factorization_residual_fast_inf = residual_factorization_fast(A0, LU, piv, n, NormType::Infinity);
    double solve_residual_inf = residual_solve(A0, x_computed, b, n, NormType::Infinity);
    double max_solution_error = max_abs_difference(x_true, x_computed);

    // 8. Print results.
    print_vector("x_true    ", x_true);
    print_vector("x_computed", x_computed);

    std::cout << "max |x_true - x_computed|          = " << max_solution_error << "\n";
    std::cout << "factorization residual debug, inf = " << factorization_residual_debug_inf << "\n";
    std::cout << "factorization residual fast,  inf = " << factorization_residual_fast_inf << "\n";
    std::cout << "solve residual, inf               = " << solve_residual_inf << "\n";


    // ============================================================
    // Ex 2: Same matrix with multiple right-hand sides
    // ============================================================
    //
    // This example solves:
    //
    //     A*x_1 = b_1
    //     A*x_2 = b_2
    //     A*x_3 = b_3
    //
    // The matrix A is the same in all three systems, so we factor A once
    // and reuse the same LU factors for every RHS vector.
    //
    // Multiple-RHS storage convention:
    //   B stores each RHS vector contiguously:
    //
    //      B = [ b_1 entries, b_2 entries, b_3 entries ]
    //
    //   Entry i of RHS rhs_index is stored at:
    //
    //      B[n * rhs_index + i]

    std::cout << "\n============================================================\n";
    std::cout << "Ex 2: Multiple RHS solve\n";
    std::cout << "============================================================\n";

    const std::size_t nrhs = 3;

    // 1. Define several known solution vectors.
    //
    // Each block of n entries is one solution vector:
    //   x_true_1 = { 2,  3, -1}
    //   x_true_2 = { 1,  0,  2}
    //   x_true_3 = {-1,  4,  1}
    std::vector<double> X_true = {
         2.0,  3.0, -1.0,
         1.0,  0.0,  2.0,
        -1.0,  4.0,  1.0
    };

    // 2. Build B = A*X_true.
    //
    // Each RHS b_k is computed from b_k = A*x_true_k.
    std::vector<double> B(n * nrhs, 0.0);

    for (std::size_t rhs = 0; rhs < nrhs; ++rhs) {
        std::vector<double> x_rhs = extract_rhs_column(X_true, n, nrhs, rhs);
        std::vector<double> b_rhs = matrix_vector_product(A0, x_rhs, n);

        for (std::size_t i = 0; i < n; ++i) {
            B[n * rhs + i] = b_rhs[i];
        }
    }

    // 3. Make a fresh LU copy and pivot vector.
    std::vector<double> LU_multi = A0;

    std::vector<std::size_t> piv_multi(n);
    std::iota(piv_multi.begin(), piv_multi.end(), std::size_t{0});

    // 4. Factor A once.
    LU_factorize_Status factorize_status_multi = LU_factorize(LU_multi, piv_multi, n, eps);

    std::cout << "Factorization status = " << to_string_factorization_status(factorize_status_multi) << "\n";

    if (factorize_status_multi != LU_factorize_Status::Success) {
        std::cerr << "Ex 2 failed: factorization did not succeed.\n";
        return 1;
    }

    // 5. Solve each RHS using the same LU factors.
    std::vector<double> X_computed(n * nrhs, 0.0);

    for (std::size_t rhs = 0; rhs < nrhs; ++rhs) {
        std::vector<double> b_rhs = extract_rhs_column(B, n, nrhs, rhs);
        std::vector<double> x_rhs(n, 0.0);

        LU_Solve_Status solve_status_multi = LU_solve(LU_multi, piv_multi, b_rhs, x_rhs, n, eps);

        std::cout << "RHS " << rhs << " solve status = " << to_string_solve_status(solve_status_multi) << "\n";

        if (solve_status_multi != LU_Solve_Status::Success) {
            std::cerr << "Ex 2 failed: solve did not succeed for RHS " << rhs << ".\n";
            return 1;
        }

        store_solution_column(X_computed, x_rhs, n, nrhs, rhs);
    }

    // 6. Compute solve residual for each RHS.
    std::vector<double> multiple_rhs_residuals = multiple_rhs_solve_residuals(A0, X_computed, B, n, nrhs, NormType::Infinity);

    // 7. Print computed solutions and residuals.
    for (std::size_t rhs = 0; rhs < nrhs; ++rhs) {
        std::vector<double> x_true_rhs = extract_rhs_column(X_true, n, nrhs, rhs);
        std::vector<double> x_computed_rhs = extract_rhs_column(X_computed, n, nrhs, rhs);

        double rhs_solution_error = max_abs_difference(x_true_rhs, x_computed_rhs);

        std::cout << "\nRHS " << rhs << ":\n";
        print_rhs_column("x_true    ", X_true, n, rhs);
        print_rhs_column("x_computed", X_computed, n, rhs);

        std::cout << "max |x_true - x_computed| = " << rhs_solution_error << "\n";
        std::cout << "solve residual, inf      = " << multiple_rhs_residuals[rhs] << "\n";
    }

    std::cout << "\nDemo complete.\n";

    return 0;
}