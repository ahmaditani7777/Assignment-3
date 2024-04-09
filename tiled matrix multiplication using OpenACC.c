#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>

#define TILE_SIZE 16 

void matrixMultiplyTiledOpenACC(float* A, float* B, float* C, int numRowsA, int numColsA, int numColsB) {
    #pragma acc data copyin(A[0:numRowsA*numColsA], B[0:numColsA*numColsB]) copyout(C[0:numRowsA*numColsB])
    {
        for (int ii = 0; ii < numRowsA; ii += TILE_SIZE){
            for (int jj = 0; jj < numColsB; jj += TILE_SIZE){
                #pragma acc parallel loop collapse(2)
                for(int i = ii; i < min(ii + TILE_SIZE, numRowsA); i++){
                    for(int j = jj; j < min(jj + TILE_SIZE, numColsB); j++) {
                        float sum= 0;
                        for (int k= 0; k < numColsA; k++) { 
                            sum += A[i * numColsA + k]* B[k * numColsB + j];
                        }
                        C[i* numColsB + j]= sum;
                    }
                }
            }
        }
    }
}

int main() {
    int numRowsA = 1024, numColsA = 1024, numColsB = 1024;
    float *A, *B, *C;

    A = (float*)malloc(numRowsA * numColsA * sizeof(float));
    B = (float*)malloc(numColsA * numColsB * sizeof(float));
    C = (float*)malloc(numRowsA * numColsB * sizeof(float));
    for (int i = 0; i < numRowsA * numColsA; i++) {
        A[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < numColsA * numColsB; i++) {
        B[i] = rand() / (float)RAND_MAX;
    }
    matrixMultiplyTiledOpenACC(A, B, C, numRowsA, numColsA, numColsB);

    free(A);
    free(B);
    free(C);
}