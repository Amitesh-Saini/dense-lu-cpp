// steady_heat_1d.cpp
// Purpose:
//   Demonstrate LU factorization reuse on a finite-difference 1D steady heat problem.
//
// Implementation notes:
//   - Discretizes -u''(x) = f(x) on (0,1) with homogeneous Dirichlet boundaries.
//   - Builds a dense finite-difference matrix for the custom dense LU solver.
//   - Factors the matrix once and reuses the LU factors for multiple source terms.
//   - Notes that a production version of this problem would normally use a tridiagonal or sparse solver.

#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "lu.hpp"
#include "verify.hpp"

constexpr double PI = 3.14159265358979323846;


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
    // ============================================================
    // Problem setup
    // ============================================================

    const std::size_t n = 256;
    const std::size_t nrhs = 4;

    const double h = 1.0 / static_cast<double>(n + 1);
    const double inv_h2 = 1.0 / (h * h);
    const double eps = 1e-12;

    std::vector<double> A0(n * n, 0.0);
    std::vector<double> B(n * nrhs, 0.0);
    std::vector<double> X_computed(n * nrhs, 0.0);

    std::vector<std::size_t> piv(n);
    std::iota(piv.begin(), piv.end(), std::size_t{0});

    const std::vector<std::string> source_names = {
        "sin(pi*x)",
        "sin(2*pi*x)",
        "x*(1-x)",
        "exp(-100*(x - 0.5)^2)"
    };

    std::cout << "============================================================\n";
    std::cout << "Steady 1D heat equation demo\n";
    std::cout << "============================================================\n";
    std::cout << "Equation: -u''(x) = f(x), 0 < x < 1\n";
    std::cout << "Boundary conditions: u(0) = 0, u(1) = 0\n";
    std::cout << "Interior grid points n = " << n << "\n";
    std::cout << "Number of RHS/source terms = " << nrhs << "\n\n";

    // ============================================================
    // Build dense finite-difference matrix A and RHS matrix B
    // ============================================================
    //
    // Interior grid points:
    //     x_i = (i + 1)h,  i = 0, 1, ..., n-1
    //
    // Finite-difference approximation:
    //     -u''(x_i) ≈ (-u_{i-1} + 2u_i - u_{i+1}) / h^2
    //
    // Therefore:
    //     A(i,i)     =  2 / h^2
    //     A(i,i-1)   = -1 / h^2
    //     A(i,i+1)   = -1 / h^2
    //
    // RHS storage convention:
    //     B[n * rhs + i] stores entry i of RHS vector rhs.

    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i + 1) * h;

        // RHS 0: smooth single-mode heat source.
        B[0 * n + i] = std::sin(PI * x);

        // RHS 1: higher-frequency source with sign change.
        B[1 * n + i] = std::sin(2.0 * PI * x);

        // RHS 2: smooth polynomial source, strongest near the center.
        B[2 * n + i] = x * (1.0 - x);

        // RHS 3: localized Gaussian heat source near x = 0.5.
        B[3 * n + i] = std::exp(-100.0 * (x - 0.5) * (x - 0.5));

        at(A0, n, i, i) = 2.0 * inv_h2;

        if (i > 0) {
            at(A0, n, i, i - 1) = -1.0 * inv_h2;
        }

        if (i + 1 < n) {
            at(A0, n, i, i + 1) = -1.0 * inv_h2;
        }
    }

    // LU_factorize modifies the matrix in place, so use a working copy.
    std::vector<double> LU = A0;

    // ============================================================
    // Factor A once
    // ============================================================

    LU_factorize_Status factorize_status = LU_factorize(LU, piv, n, eps);

    std::cout << "Factorization status = "
              << to_string_factorization_status(factorize_status)
              << "\n";

    if (factorize_status != LU_factorize_Status::Success) {
        std::cerr << "steady_heat_1d failed: factorization did not succeed.\n";
        return 1;
    }

    // ============================================================
    // Solve A*u = f for each RHS using the same LU factors
    // ============================================================
    //
    // For each RHS:
    //   1. Extract f_rhs from B.
    //   2. Solve A*u_rhs = f_rhs using LU_solve().
    //   3. Store u_rhs into X_computed.
    //
    // This demonstrates factorization reuse:
    //   A is factored once, then reused for all source terms.

    for (std::size_t rhs = 0; rhs < nrhs; ++rhs) {
        std::vector<double> b_rhs = extract_rhs_column(B, n, nrhs, rhs);
        std::vector<double> x_rhs(n, 0.0);

        LU_Solve_Status solve_status = LU_solve(LU, piv, b_rhs, x_rhs, n, eps);

        std::cout << "RHS " << rhs << " [" << source_names[rhs] << "] solve status = "
                  << to_string_solve_status(solve_status)
                  << "\n";

        if (solve_status != LU_Solve_Status::Success) {
            std::cerr << "steady_heat_1d failed: solve did not succeed for RHS "
                      << rhs << ".\n";
            return 1;
        }

        store_solution_column(X_computed, x_rhs, n, nrhs, rhs);
    }

    // ============================================================
    // Verify factorization and solve residuals
    // ============================================================

    double inf_factorization_residual = residual_factorization_fast(A0, LU, piv, n, NormType::Infinity);
    double fro_factorization_residual = residual_factorization_fast(A0, LU, piv, n, NormType::Frobenius);

    std::cout << "\nFactorization residuals:\n";
    std::cout << "  fast infinity-norm residual  = " << inf_factorization_residual << "\n";
    std::cout << "  fast Frobenius-norm residual = " << fro_factorization_residual << "\n";

    std::vector<double> multiple_rhs_residuals = multiple_rhs_solve_residuals(A0, X_computed, B, n, nrhs, NormType::Infinity);

    std::cout << "\nSolve residuals:\n";

    for (std::size_t rhs = 0; rhs < nrhs; ++rhs) {
        std::cout << "  RHS " << rhs << " [" << source_names[rhs] << "] infinity-norm residual = "
                  << multiple_rhs_residuals[rhs]
                  << "\n";
    }

    std::cout << "\nDemo complete.\n";

    return 0;
}