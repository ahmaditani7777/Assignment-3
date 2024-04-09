#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixMultiplyBasic(float* A, float* B, float* C, int numARows, int numACols, int numBCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < numARows && col < numBCols) {
        for (int k = 0; k < numACols; ++k) {
            sum += A[row * numACols + k] * B[k * numBCols + col];
        }
        C[row * numBCols + col] = sum;
    }
}

int main() {
   
    int numARows = 1024; 
    int numACols = 1024; 
    int numBCols = 1024; 

    float *A, *B, *C;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeof(float) * numARows * numACols);
    cudaMalloc((void **)&d_B, sizeof(float) * numACols * numBCols);
    cudaMalloc((void **)&d_C, sizeof(float) * numARows * numBCols);
    cudaMemcpy(d_A, A, sizeof(float) * numARows * numACols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * numACols * numBCols, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((numBCols + dimBlock.x - 1) / dimBlock.x, (numARows + dimBlock.y - 1) / dimBlock.y);
    matrixMultiplyBasic<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, numARows, numACols, numBCols);

    cudaMemcpy(C, d_C, sizeof(float) * numARows * numBCols, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}