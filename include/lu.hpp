// File purpose:
//   Declares the public LU factorization and solve API for dense row-major matrices.
//   This header provides element-access helpers, printing/debug helpers, row pivoting,
//   Gaussian elimination, LU factorization, and triangular solve routines.


#pragma once

#include <cstddef>
#include <vector>


// Function:
//   Mutable row-major matrix element accessor.
// What it does:
//   Returns a writable reference to the entry A(i, j) stored in a flat row-major vector.
inline double& at(std::vector<double>& array, std::size_t n, std::size_t i, std::size_t j) {
    return array[i*n + j];
}

// Function:
//   Const row-major matrix element accessor.
// What it does:
//   Returns the value of entry A(i, j) stored in a flat row-major vector without modifying it.
inline double at(const std::vector<double>& array, std::size_t n, std::size_t i, std::size_t j) {
    return array[i*n + j];
}

// Function:
//   Matrix printing helper.
// What it does:
//   Prints an n by n flat row-major matrix for debugging or manual inspection.
void Printarray(const std::vector<double>& array, std::size_t n);

// Function:
//   Vector printing helper.
// What it does:
//   Prints a vector of length n for debugging or manual inspection.
void Printvec(const std::vector<double>& array, std::size_t n);

// Function:
//   Pivot-vector printing helper.
// What it does:
//   Prints the permutation/pivot vector used by the LU factorization.
void Printpiv(const std::vector<std::size_t>& array, std::size_t n);

// Function:
//   Row-swap helper.
// What it does:
//   Swaps two rows of an n by n flat row-major matrix in place.
void rowswap(std::vector<double>& array, std::size_t rw_1, std::size_t rw_2, std::size_t n);

// Function:
//   Partial-pivoting step.
// What it does:
//   Finds and applies the pivot row for column rw, updates the pivot vector, and reports failure if the pivot is below eps.
bool pivot(std::vector<double>& array, std::vector<std::size_t>& piv, std::size_t rw, std::size_t n, double eps);

// Function:
//   Gaussian-elimination step.
// What it does:
//   Performs the elimination below pivot row rw and stores the multipliers in the lower-triangular part of the matrix.
bool gaussian_eliminate(std::vector<double>& array, std::size_t rw, std::size_t n, double eps);

enum class LU_factorize_Status {
    Success,
    PivotFailure,
    EliminationFailure,
    GeneralFailure
};

// Function:
//   LU factorization driver.
// What it does:
//   Computes the in-place PA = LU factorization with partial pivoting and returns a status code describing success or failure.
LU_factorize_Status LU_factorize(std::vector<double>& array, std::vector<std::size_t>& piv, std::size_t n, double eps);

// Function:
//   Forward-substitution solve.
// What it does:
//   Applies the pivot vector to b and solves Lc = Pb using the unit-lower-triangular part of the LU factorization.
bool forward_substitution(
    const std::vector<double>& LU, const std::vector<std::size_t>& piv, const std::vector<double>& b, std::vector<double>& c, std::size_t n);

// Function:
//   Back-substitution solve.
// What it does:
//   Solves Ux = c using the upper-triangular part of the LU factorization and checks for small pivots using eps.
bool back_substitution(const std::vector<double>& LU, const std::vector<double>& c, std::vector<double>& x, std::size_t n, double eps);

enum class LU_Solve_Status {
    Success,
    ForwardSubstitutionFailure,
    BackwardSubstitutionFailure,
    GeneralFailure
};

// Function:
//   LU-based linear-system solve.
// What it does:
//   Solves Ax = b from an existing LU factorization by running forward substitution followed by back substitution.
LU_Solve_Status LU_solve(
    const std::vector<double>& LU, const std::vector<std::size_t>& piv, const std::vector<double>& b, std::vector<double>& x, std::size_t n, double eps);