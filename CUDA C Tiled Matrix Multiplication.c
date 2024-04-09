#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int numARows, int numAColumns, int numBColumns){
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float sum = 0.0f;
    for (int m = 0;m< (numAColumns - 1)/TILE_WIDTH + 1; ++m){
        if(m * TILE_WIDTH + tx < numAColumns && row < numARows){
            tileA[ty][tx] = A[row * numAColumns + m * TILE_WIDTH + tx];
        }else 
        {tileA[ty][tx] = 0.0;
        }
        if (m * TILE_WIDTH + ty < numAColumns && col < numBColumns){
            tileB[ty][tx]= B[(m * TILE_WIDTH + ty) * numBColumns + col];
        } else{tileB[ty][tx] = 0.0;}
        __syncthreads();
        for(int k = 0; k < TILE_WIDTH; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
         }
        __syncthreads();
    }
    if (row < numARows && col < numBColumns) {
        C[row * numBColumns + col] = sum;
    }
}
int main() {
    const int numARows = 1024, numAColumns = 1024, numBColumns = 1024;
    size_t sizeA= numARows * numAColumns * sizeof(float);
    size_t sizeB= numAColumns * numBColumns * sizeof(float);
    size_t sizeC= numARows * numBColumns * sizeof(float);

    float *h_A= (float *)malloc(sizeA);
    float *h_B= (float *)malloc(sizeB);
    float *h_C= (float *)malloc(sizeC);
    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A,sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((numBColumns + TILE_WIDTH - 1) / TILE_WIDTH, (numARows + TILE_WIDTH - 1) / TILE_WIDTH);
    matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, numARows, numAColumns, numBColumns);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}