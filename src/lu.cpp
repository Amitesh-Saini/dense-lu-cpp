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

        std::cout << array[i];
        if((i+1)%n == 0){
            std::cout << "\n";
        }
    }
    std::cout << std::endl;
}



void rowswap(std::vector<double>& array, int rw_1, int rw_2, int n){
    for(int i = 0; i<n; i++){
        double temp = at(array, n, rw_1, i);
        at(array, n, rw_1, i) = at(array, n, rw_2, i);
        at(array, n, rw_2, i) = temp;
    }
}

void copy(const std::vector<double>& Farray, std::vector<double>& Tarray, int n){

    for(int i = 0; i<n*n; i++){

        Tarray[i] = Farray[i];
    }
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



