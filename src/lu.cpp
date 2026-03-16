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




void Printarray(const std::vector<double>& array, int n){

    for(int i = 0; i<n*n; i++){

        std::cout << array[i] << ", ";
        if((i+1)%n == 0){
            std::cout << "\n";
        }
    }
    std::cout << std::endl;
}


void Printvec(const std::vector<double>& array, int n){

    for(int i = 0; i<n; i++){

        std::cout << array[i] << "\n";
    }
    std::cout << std::endl;
}




void Printpiv(const std::vector<int>& array, int n){

    for(int i = 0; i<n; i++){

        std::cout << array[i] << ",";
    }
    std::cout << std::endl;
}


void rowswap(std::vector<double>& array, int rw_1, int rw_2, int n){
        
    for(int i = 0; i<n; i++) std::swap(at(array, n, rw_1, i), at(array, n, rw_2, i));
}




void identity_mat(std::vector<double>& array, int n){

    for(int i = 0; i<n*n; i++){

        if((i)%(n+1) == 0){
            array[i] = 1;
        }

        else{
            array[i] = 0;
        }
    }
}

bool pivot(std::vector<double>& array, std::vector<int>& piv, int rw, int n, double eps){

    int max_row = rw;

    double max = std::abs(at(array, n, rw, rw));

   // std::cout << max << "\n";

    for(int i = rw+1; i<n; i++){

        if(std::abs(at(array, n, i, rw)) > max){

            max = std::abs(at(array, n, i, rw));

            max_row = i;
        }
    }

   // std::cout << max << "\n";
    //std::cout << max_row_entry << "\n";

    if(max <= eps) return false;
    if(max_row != rw){

        rowswap(array, rw, max_row, n);
        std::swap(piv[rw], piv[max_row]);
    }
    return true;
}



bool gaussian_eliminate(std::vector<double>& array, int rw, int n, double eps){

    if(std::abs(at(array, n, rw, rw)) <= eps){             // check if pivot can cause instability
        return false;
    }

    for(int i = rw+1; i <n; i++){

        double mul = at(array, n, i, rw)/ at(array, n, rw, rw);
        at(array, n, i, rw) = mul;

        for(int j = rw+1; j<n; j++){

            at(array, n, i, j) -= mul*at(array, n, rw, j);
        }
    }

    return true;
}

LU_factorize_Status LU_factorize(std::vector<double>& array, std::vector<int>& piv, int n, double eps){

    if (n <= 0) return LU_factorize_Status::GeneralFailure;
    if ((int)array.size() != n * n) return LU_factorize_Status::GeneralFailure;
    if ((int)piv.size() != n) return LU_factorize_Status::GeneralFailure;

    if(eps <= 0) return LU_factorize_Status::GeneralFailure;

    for(int i = 0; i<n-1; ++i){
        
        if(!pivot(array, piv, i, n, eps)) return LU_factorize_Status::PivotFailure;

        if(!gaussian_eliminate(array, i, n, eps)) return LU_factorize_Status::EliminationFailure;
    }
    // Final pivot check: no elimination needed, but U(n-1,n-1) must still be valid.
    if (!pivot(array, piv, n-1, n, eps)) return LU_factorize_Status::PivotFailure;

    return LU_factorize_Status::Success;
}

bool forward_substitution(const std::vector<double>& LU, const std::vector<int>& piv, const std::vector<double>& b,
std::vector<double>& c, int n){


    if (n <= 0) return false;
    if ((int)LU.size() != n * n) return false;
    if ((int)piv.size() != n) return false;
    if ((int)b.size() != n) return  false;

    c.resize(n);

    for(int i = 0; i<n; ++i){
        if (piv[i] < 0 || piv[i] >= n) return false;

        double sum = 0.0;

        for(int j = 0; j<i; ++j){

            sum += c[j]*LU[i*n+j];
        }

        c[i] = b[piv[i]] - sum;
    }

    return true;
}

bool back_substitution(const std::vector<double>& LU, const std::vector<double>& c, std::vector<double>& x, int n, double eps){

    if (n <= 0) return false;
    if ((int)LU.size() != n * n) return false;
    if ((int)c.size() != n) return false;

    x.resize(n);

    for(int i = n-1; i >= 0; --i){

        double sum = 0.0;
        double diag = LU[i*n+i];

        if(std::abs(diag) <= eps) return false;

        for(int j = n-1; j>i; --j){

            sum += LU[i*n+j] * x[j]; 
        }

        x[i] = (c[i] - sum)/diag;
    }

    return true;
}


LU_Solve_Status LU_solve(const std::vector<double>& LU, const std::vector<int>& piv, const std::vector<double>& b, std::vector<double>& x, int n, double eps){

    if(eps <= 0) return LU_Solve_Status::GeneralFailure;
    
    std::vector<double> c(n);

    if(!forward_substitution(LU, piv, b, c, n)) return LU_Solve_Status::ForwardSubstitutionFailure;

    if(!back_substitution(LU, c, x, n, eps)) return LU_Solve_Status::BackwardSubstitutionFailure;

    return LU_Solve_Status::Success;

}