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

    int max_row_entry = rw;

    double max = std::abs(at(array, n, rw, rw));

   // std::cout << max << "\n";

    for(int i = rw+1; i<n; i++){

        if(std::abs(at(array, n, i, rw)) > max){

            max = std::abs(at(array, n, i, rw));

            max_row_entry = i;
        }
    }

   // std::cout << max << "\n";
    //std::cout << max_row_entry << "\n";

    if(max <= eps){

        std::cout << "Pivot too small at column " << rw << " (max |at(array, n, i, rw)| <= eps)\n";
        return false;
    }

    if(max_row_entry != rw){

        rowswap(array, rw, max_row_entry, n);
        std::swap(piv[rw], piv[max_row_entry]);
    }
    return true;
}



bool gaussian_eliminate(std::vector<double>& array, int rw, int n, double eps){

    if(std::abs(at(array, n, rw, rw)) <= eps){             // check if pivot can cause instability
        return false;
    }

    for(int i = rw+1; i <n; i++){

        if(std::abs(at(array, n, i, rw)) <= eps){   // if any entry to be elimated is smaller than eps set to 0 - removes chance for instabilty 

            at(array, n, i, rw) = 0.0;
            continue;
        }

        double mul = at(array, n, i, rw)/ at(array, n, rw, rw);
        at(array, n, i, rw) = mul;

        for(int j = rw+1; j<n; j++){

            at(array, n, i, j) -= mul*at(array, n, rw, j);
            if(std::abs(at(array, n, i, j)) <= eps) at(array, n, i, j) = 0.0;
        }
    }

    return true;
}


