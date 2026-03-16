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
#include "verify.hpp"
#include <numeric>
#include <chrono>


// Average time for residual_factorization_debug
double benchmark_residual_debug_avg(
    const std::vector<double>& A0,
    const std::vector<int>& piv,
    const std::vector<double>& LU_fac,
    int n,
    int num_runs
) {
    using clock = std::chrono::high_resolution_clock;

    std::vector<double> times;
    times.reserve(num_runs);

    for (int run = 0; run < num_runs; ++run) {
        std::vector<double> LU_Prod;
        std::vector<double> PA;
        std::vector<double> diff;

        auto start = clock::now();

        volatile double res = residual_factorization_debug(
            A0, LU_Prod, piv, PA, LU_fac, diff, n, NormType::Infinity
        );

        auto end = clock::now();

        std::chrono::duration<double> elapsed = end - start;
        times.push_back(elapsed.count());

        (void)res;
    }

    double total = std::accumulate(times.begin(), times.end(), 0.0);
    return total / static_cast<double>(num_runs);
}

// Average time for residual_factorization_fast
double benchmark_residual_fast_avg(
    const std::vector<double>& A0,
    const std::vector<double>& LU_fac,
    const std::vector<int>& piv,
    int n,
    int num_runs
) {
    using clock = std::chrono::high_resolution_clock;

    std::vector<double> times;
    times.reserve(num_runs);

    for (int run = 0; run < num_runs; ++run) {
        auto start = clock::now();

        volatile double res = residual_factorization_fast(
            A0, LU_fac, piv, n, NormType::Infinity
        );

        auto end = clock::now();

        std::chrono::duration<double> elapsed = end - start;
        times.push_back(elapsed.count());

        (void)res;
    }

    double total = std::accumulate(times.begin(), times.end(), 0.0);
    return total / static_cast<double>(num_runs);
}


int main(){

     int n = 3;

    std::vector<double> array = {1,3,5,3,15,7,5,9,6};

    std::vector<double> b = {10, 9, 7};

    std::vector<double> A0(n*n);

    std::vector<double> LU(n*n);

    std::vector<double> PA(n*n);

    std::vector<double> diff(n*n);

    std::vector<double> x(n);

    A0 = array;

    double A_Scale = inf_norm(A0, n);

    double pivot_tol = 1e-12 * std::max(1.0, A_Scale); 


    std::vector<int> piv(n);
    for(int i = 0; i<n; i++){piv[i] = i;}

    std::vector<double> id(5*5);

    LU_factorize_Status fstatus = LU_factorize(A0, piv, n, pivot_tol);

    if (fstatus == LU_factorize_Status::PivotFailure) {
    std::cout << "FAIL: singular/unstable pivot encountered.\n";
    return 1;
    }

    if (fstatus == LU_factorize_Status::EliminationFailure) {
    std::cout << "FAIL: elimination failed.\n";
    return 1;
    }

    std::cout << "LU factorization succeeded.\n";
    Printarray(A0, n);
    std::cout << "\n";


    LU_Solve_Status status = LU_solve(A0, piv, b, x, n, pivot_tol);

    if (status == LU_Solve_Status::ForwardSubstitutionFailure) {
    std::cout << "FAIL: Forward sub error.\n";
    return 1;
    }

    if (status == LU_Solve_Status::BackwardSubstitutionFailure) {
    std::cout << "FAIL: Bakcward sub error.\n";
    return 1;
    }

    std::cout << "LU Solve succeeded.\n";


    Printarray(array, n);
    Printvec(x, n);
    std::cout << "=" << "\n";
    Printvec(b, n);





    return 0;
}
