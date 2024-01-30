#include "system.h"
// #include <systemc.h>
#include <string>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include <queue>
#include <vector>
#include <cstdlib>
#include <random>
#include <iostream>

//stuff 
using namespace std;

void inverse(float** matrix, int n){
    float** lower = new float*[n]; 
    float** upper = new float*[n];
    float** trans = new float*[n]; 

    for(int i = 0; i < n; i++){
        lower[i] = new float[n];
        upper[i] = new float[n];
        trans[i] = new float[n];

        for(int j = 0; j < n; j++){
            lower[i][j] = 0; 
            upper[i][j] = 0;
            trans[i][j] = 0;
        }
    }

    // conduct LU decomposition producing L^-1 and U^T
    for(int i = 0; i < n; i++){
        for(int k = i; k < n; k++){
            float sum = 0; 
            for(int j = 0; j < i; j++){
                sum+= (lower[i][j]*upper[k][j]);
            }
            upper[k][i] = matrix[i][k] - sum; 
        }

        for(int k = i; k < n; k++){
            if(i == k)lower[i][i] = 1;
            else{
                int sum = 0;
                for(int j = 0; j < i; j++){
                    sum+= lower[k][j] * upper[i][j];
                }
                lower[k][i] = (matrix[k][i] - sum) / upper[i][i];
            }
        }
    }

    for(int i = 0; i < n; i++){
        for(int k = i + n - 1; k > i; k--){
            if(k < n) lower[k][i] = (k - i == 2) ? (lower[k - 1][i]*lower[k][i + 1] - lower[k][i]) : -lower[k][i];
        }
    }
    
    //obtain U^-1
    transform_lower(upper, trans, n); 

    //calculate A^-1 = U^-1 * L^-1
    mult_mat(trans, lower, n);

    for (int i = 0; i< n; i++){
        for (int j = 0; j < n; j++){
            matrix[i][j] = lower[i][j];
        }
    }

    return;
}

void transform_lower(float** lower, float**trans, int n){
    for(int i = 0; i < n; i++){
        trans[i][i] = 1/lower[i][i];
        for (int j = i + 1; j < n; j++){
            trans[i][j] = (j - i == 1) ? -(lower[j][i]/(lower[j - 1][i]*lower[j][i+1])) : (lower[j - 1][i]*lower[j][i + 1] - lower[j - 1][i + 1]*lower[j][i])/diag(lower, n); 
        }
    }

    return; 
}

float diag(float** lower, int n){
    float prod  = 1; 
    for (int i = 0; i < n; i++){
        prod*=lower[i][i];
    }
    return prod; 
}


void mult_mat(float** matrix1, float** matrix2, int n){
    float inter[n][n]; 
    uint8_t i = 0;
    uint8_t j = 0;
    uint8_t k = 0;
    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            inter[i][j] = matrix2[i][j];
        }
    }

    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            matrix2[i][j] = 0;
            for (k = 0; k < n; k++){
                matrix2[i][j] += matrix1[i][k]*inter[k][j];
            }
        }
    }
}

int main(){
    float** test_matrix = new float*[2];

    float** test_matrix_3 = new float*[3]; 

    for (int i = 0; i < 3; i++){
        if(i < 3) test_matrix[i] = new float[2];
        test_matrix_3[i] = new float[3]; 
    }

    test_matrix_3[0][0] = 1; 
    test_matrix_3[0][1] = 0; 
    test_matrix_3[0][2] = 4; 
    test_matrix_3[1][0] = 0; 
    test_matrix_3[1][1] = 1; 
    test_matrix_3[1][2] = 2;  
    test_matrix_3[2][0] = 4; 
    test_matrix_3[2][1] = 2; 
    test_matrix_3[2][2] = 35;  

    test_matrix[0][0] = 1; 
    test_matrix[0][1] = 2; 
    test_matrix[1][0] = 3; 
    test_matrix[1][1] = 4;       

    inverse(test_matrix_3, 3); 

}