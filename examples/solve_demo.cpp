// solve_demo.cpp
// Purpose:
//   Small example program showing how to call your LU factorization + solve API.
//
// Suggested demonstration:
//   - Build a small matrix A and vector b
//   - Factorize (PA=LU)
//   - Solve Ax=b
//   - Print x and residual


//cmake --build --preset debug
//./out/build/debug/lu_example


#include <vector>
#include <iostream>
#include "lu.hpp"


int main(){

     int n = 3;

    std::vector<double> array = {1,2,3,4,5,6,7,8,9};

    std::vector<double> A0(n*n);

    A0 = array;

    std::vector<int> piv(n);
    for(int i = 0; i<n; i++){piv[i] = i;}

    std::vector<double> id(5*5);

    pivot(A0, piv, 0, n, 1e-12);

    gaussian_eliminate(A0, 0, n, 1e-12);

    pivot(A0, piv, 1, n, 1e-12);

   // gaussian_eliminate(A0, 1, n, 1e-12);

    

    Printarray(A0, n);

    return 0;
}