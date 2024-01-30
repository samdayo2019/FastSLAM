#ifndef SYSTEM_H
#define SYSTEM_H

#include <cmath>
#include <stdint.h>
#include <vector>


void mult_mat(float** matrix1, float** matrix2, int n);
void inverse(float** matrix, int n);
float diag(float** lower, int n);
void transform_lower(float** lower, float**trans, int n);





#endif