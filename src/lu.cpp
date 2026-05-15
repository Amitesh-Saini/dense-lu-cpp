// lu.cpp
// Purpose:
//   Implement LU factorization (PA = LU) and the linear solver.
//
// Implementation notes:
//   - Partial pivoting (choose max abs pivot in column).
//   - Store L and U in a single matrix (below diag = L multipliers, diag+above = U).
//   - Use a permutation vector instead of forming an explicit P matrix.
//   - Detect singular/near-singular pivots and return failure.

#include "lu.hpp"

#include <cmath>
#include <algorithm>
#include <iostream>


// Helpers

void Printarray(const std::vector<double>& array, std::size_t n){

    for(std::size_t i = 0; i < n*n; i++){

        std::cout << array[i] << ", ";
        if((i+1)%n == 0){
            std::cout << "\n";
        }
    }
    std::cout << std::endl;
}

void Printvec(const std::vector<double>& array, std::size_t n){

    for(std::size_t i = 0; i < n; i++){

        std::cout << array[i] << "\n";
    }
    std::cout << std::endl;
}

void Printpiv(const std::vector<std::size_t>& array, std::size_t n){

    for(std::size_t i = 0; i < n; i++){

        std::cout << array[i] << ",";
    }
    std::cout << std::endl; 
}

void rowswap(std::vector<double>& array, std::size_t rw_1, std::size_t rw_2, std::size_t n){
        
    for(std::size_t i = 0; i < n; i++) std::swap(at(array, n, rw_1, i), at(array, n, rw_2, i));
}

bool pivot(std::vector<double>& array, std::vector<std::size_t>& piv, std::size_t rw, std::size_t n, double eps){

    std::size_t max_row = rw;

    double max = std::abs(at(array, n, rw, rw));  // set max = |A[rw][rw]|

    for(std::size_t i = rw+1; i < n; i++){

        if(std::abs(at(array, n, i, rw)) > max){

            max = std::abs(at(array, n, i, rw));

            max_row = i;
        }
    }

    if(max <= eps) return false;  // check if pivot can cause instability
    if(max_row != rw){

        rowswap(array, rw, max_row, n);
        std::swap(piv[rw], piv[max_row]);
    }
    return true;
}

bool gaussian_eliminate(std::vector<double>& array, std::size_t rw, std::size_t n, double eps){

    if(std::abs(at(array, n, rw, rw)) <= eps){                    // check if pivot can cause instability
        return false;
    }

    for(std::size_t i = rw+1; i < n; i++){

        double mul = at(array, n, i, rw)/ at(array, n, rw, rw);   // Multiplier used to eliminate the entry below the pivot in column rw.
        at(array, n, i, rw) = mul;                                // store mul at A[rw + 1][rw]

        for(std::size_t j = rw+1; j < n; j++){

            at(array, n, i, j) -= mul*at(array, n, rw, j);
        }
    }

    return true;
}

LU_factorize_Status LU_factorize(std::vector<double>& array, std::vector<std::size_t>& piv, std::size_t n, double eps){

    if (n == 0) return LU_factorize_Status::GeneralFailure;
    if (array.size() != n * n) return LU_factorize_Status::GeneralFailure;
    if (piv.size() != n) return LU_factorize_Status::GeneralFailure;

    if(eps <= 0) return LU_factorize_Status::GeneralFailure;

    for(std::size_t i = 0; i < n-1; ++i){
        
        if(!pivot(array, piv, i, n, eps)) return LU_factorize_Status::PivotFailure;

        if(!gaussian_eliminate(array, i, n, eps)) return LU_factorize_Status::EliminationFailure;
    }

    // Final pivot check: no elimination needed, but U(n-1,n-1) must still be valid.
    if (!pivot(array, piv, n-1, n, eps)) return LU_factorize_Status::PivotFailure;

    return LU_factorize_Status::Success;
}

bool forward_substitution(
    const std::vector<double>& LU, const std::vector<std::size_t>& piv, const std::vector<double>& b, std::vector<double>& c, std::size_t n){

    if (n == 0) return false;
    if (LU.size() != n * n) return false;
    if (piv.size() != n) return false;
    if (b.size() != n) return  false;

    c.resize(n);

    for(std::size_t i = 0; i < n; ++i){
        if (piv[i] >= n) return false;            

        double sum = 0.0;

        for(std::size_t j = 0; j < i; ++j){
            sum += c[j]*LU[i*n+j];  // sum = L[i][0] * c[0] + L[i][1] * c[1] + ... + L[i][i-1] * c[i-1] 
        }

        c[i] = b[piv[i]] - sum;    // Apply the pivoted RHS and subtract the known L-row contribution.
    }

    return true;
}

bool back_substitution(const std::vector<double>& LU, const std::vector<double>& c, std::vector<double>& x, std::size_t n, double eps){

    if (n == 0) return false;
    if (LU.size() != n * n) return false;
    if (c.size() != n) return false;

    x.resize(n);

    for(std::size_t i = n; i-- > 0; ){   

        double sum = 0.0;
        double diag = LU[i*n+i]; // LU[i][i] = coefficient of x[i]

        if(std::abs(diag) <= eps) return false;   // check instability 

        for(std::size_t j = i+1; j < n; ++j){

            sum += LU[i*n+j] * x[j]; // U[i][i+1] * x[j] + U[i][i+2] * x[j+2] + ... + U[i][n-1] * x[n-1]
        }

        x[i] = (c[i] - sum)/diag;
    }

    return true;
}

LU_Solve_Status LU_solve(
    const std::vector<double>& LU, const std::vector<std::size_t>& piv, const std::vector<double>& b, std::vector<double>& x, std::size_t n, double eps){

    if(eps <= 0) return LU_Solve_Status::GeneralFailure;
    
    std::vector<double> c(n);

    if(!forward_substitution(LU, piv, b, c, n)) return LU_Solve_Status::ForwardSubstitutionFailure;

    if(!back_substitution(LU, c, x, n, eps)) return LU_Solve_Status::BackwardSubstitutionFailure;

    return LU_Solve_Status::Success;
}