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

    std::vector<double> rray(n*n);

    std::vector<double> id(5*5);


    
    at(array, n, 2, 2) = 0;

    at(array, n, 2, 2) = 9;

    double x = at(array, n, 0, 0);

    std::cout << x << "\n";


    Printarray(array, n);

    rowswap(array, 0, 1, 3);

    Printarray(array, n);

    copy(array, rray, n);

    Printarray(rray, n);

    identity_mat(id, 5);

    Printarray(id, 5);

    return 0;
}