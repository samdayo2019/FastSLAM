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
    float temp_val; 

    for (int i = 0; i < n; i++){
        for (int j = 0; j <2 * n; j++){
            if(j == (i + n))
                matrix[i][j] = 1;
        }
    }

    // perform row swapping
    for (int i = n - 1; i > 0; i--){
        if(matrix[i - 1][0] < matrix[i][0]){
            float* temp_val = matrix[i];
            matrix[i] = matrix[i - 1];
            matrix[i - 1] = temp_val;
        }
    }

    //replace row by sum of itself and constant multiple of another row
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if(j != i){
                temp_val = matrix[j][i] / matrix[i][i];
                for (int k = 0; k < 2 * n; k++){
                    matrix[j][k] -= matrix[i][k] * temp_val;
                }
            }
        }
    }

    for (int i = 0; i < n; i++){
        temp_val = matrix[i][i]; 
        for (int j = 0; j < 2 * n; j++){
            matrix[i][j] = matrix[i][j] / temp_val;
        }
    }

    return;
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