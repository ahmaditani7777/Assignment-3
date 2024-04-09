#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>

void matrixMultiplyBasicOpenACC(float* A, float* B, float* C, int numRowsA, int numColsA, int numColsB) {
    #pragma acc data copyin(A[0:numRowsA*numColsA], B[0:numColsA*numColsB]) copyout(C[0:numRowsA*numColsB])
    {
        #pragma acc parallel loop
        for (int row = 0; row < numRowsA; row++) {
            #pragma acc loop
            for (int col = 0; col < numColsB; col++) {
                float sum = 0;
                for (int k = 0; k < numColsA; k++) {
                    sum += A[row * numColsA + k] * B[k * numColsB + col];
                }
                C[row * numColsB + col] = sum;
            }
        }
    }
}
int main() {
    int numARows = 1024, numACols = 1024, numBCols = 1024;
    float *A, *B, *C;
    A = (float *)malloc(sizeof(float) * numARows * numACols);
    B = (float *)malloc(sizeof(float) * numACols * numBCols);
    C = (float *)malloc(sizeof(float) * numARows * numBCols);

   for(int i = 0; i < numARows * numACols; i++) {
        A[i] = rand() / (float)RAND_MAX;
    }
    
    for(int i = 0; i < numACols * numBCols; i++) {
        B[i] = rand() / (float)RAND_MAX;
    }

    matrixMultiplyBasicOpenACC(A, B, C, numARows, numACols, numBCols);
    free(A);
    free(B);
    free(C);

    return 0;
}