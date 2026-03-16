#include "verify.hpp"
#include "lu.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>




void matmul(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n){

    C.resize(n*n);

    for(int i = 0; i<n; ++i){
        for(int j = 0; j<n; ++j){

            at(C, n, i, j) = 0.0;

            for(int k = 0; k<n; ++k){

                at(C, n, i, j) += at(A, n, i, k)*at(B, n, k, j);
            }
        }    
    }
}

void matsub(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n){

    C.resize(n*n);

    for(int i = 0; i<n; ++i){
        for(int j = 0; j<n; ++j) at(C, n, i, j) = at(A, n, i, j) - at(B, n, i, j);
    }
}

void apply_piv_rows(const std::vector<double>& A, const std::vector<int>& piv, std::vector<double>& PA, int n){

    PA.resize(n*n);

    for(int i = 0; i<n; i++){
        for(int j = 0; j<n; j++){

            at(PA, n, i, j) = at(A, n, piv[i], j);
        }
    }
}

double fro_norm(const std::vector<double>& A, int n){

    double sum = 0.0;

    for(int i = 0; i<n*n; i++) sum += A[i]*A[i];

    return std::sqrt(sum);
}



double inf_norm(const std::vector<double>& A, int n){

    double max = 0.0;

    for(int i = 0; i<n; i++){

        double sum = 0; 

        for(int j = 0; j<n; j++) sum += std::abs(A[i*n + j]);
        
        if(sum > max) max = sum;
    }

    return max;
}




void computeLU(const std::vector<double>& A, std::vector<double>& LU, int n){

     double L = 0.0;
     double U = 0.0;

     LU.resize(n*n);

   
    for(int i = 0; i<n; ++i){
        for(int j = 0; j<n; ++j){

            double sum = 0.0;

            for(int k = 0; k <= std::min(i,j); ++k){

                if(k == i) L = 1.0;
                else if (k < i) L = at(A, n, i, k);
                else L = 0.0;

                if(k <= j) U = at(A, n, k, j);
                else U = 0.0;

                sum += L*U;
            }

            LU[i*n+j] = sum;
        }
    }
}




double residual_factorization_debug(const std::vector<double>& A0, std::vector<double>& LU_Prod, 
const std::vector<int>& piv, std::vector<double>& PA, const std::vector<double>& LU_fac,
 std::vector<double>& diff, int n, NormType norm){
    
    apply_piv_rows(A0, piv, PA, n);
    computeLU(LU_fac, LU_Prod, n);

    diff.resize(n*n);

    matsub(PA, LU_Prod, diff, n);

    const double eps_abs = 1e-15;
    double denom = (norm== NormType::Infinity ? inf_norm(A0,n) : fro_norm(A0,n));
    double num = (norm== NormType::Infinity ? inf_norm(diff,n) : fro_norm(diff,n));

    if (denom <= eps_abs) {
        std::cout << "FAIL: ||A|| is zero or near zero; cannot compute meaningful relative residual.\n";
        return std::numeric_limits<double>::infinity();
        }

    return num/denom;

}


double residual_factorization_fast(const std::vector<double>& A0, const std::vector<double>& LU_fac,
const std::vector<int>& piv, int n, NormType norm){


    double L = 0.0;
    double U = 0.0;

    double max_row_sum = 0.0;

    double sum_Total = 0.0;


    for(int i = 0; i<n; ++i){

        double row_sum = 0.0;

        for(int j = 0; j<n; ++j){

            double PA_ij = at(A0, n, piv[i], j);

            double diff = 0.0;

            double sum = 0.0;

            for(int k = 0; k <= std::min(i,j); ++k){

                if(k == i) L = 1.0;
                else if (k < i) L = at(LU_fac, n, i, k);
                else L = 0.0;

                if(k <= j) U = at(LU_fac, n, k, j);
                else U = 0.0; 

                sum += L*U;
            }

            diff = PA_ij - sum;

            sum_Total += diff*diff;

            row_sum += std::abs(diff);

        }

        if(row_sum > max_row_sum) max_row_sum = (row_sum);
    }

    const double eps_abs = 1e-15;
    double denom = (norm== NormType::Infinity ? inf_norm(A0,n) : fro_norm(A0,n));
    double num = (norm== NormType::Infinity ? max_row_sum : std::sqrt(sum_Total));

    if (denom <= eps_abs) {
        std::cout << "FAIL: ||A|| is zero or near zero; cannot compute meaningful relative residual.\n";
        return std::numeric_limits<double>::infinity();
        }

    return num/denom;
}

