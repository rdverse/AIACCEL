


#include <kernels.cuh>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <data_gen.h>
#include <unistd.h>
#include <math.h>

__global__ void matmul_kernel(int* A, int* B, int* C, int WIDTH) {
    // consider M == N == K
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < WIDTH && col < WIDTH) {
        int sum = 0;
        for (int i = 0; i < WIDTH; i++) {
            sum += A[row*WIDTH + i] * B[i*WIDTH + col];
        }
        C[row*WIDTH + col] = sum;
    }
}

__global__ void matmul_kernel_tiled(int* A, int* B, int* C, int WIDTH, int TILE_WIDTH){
    // consider M == N == K
    const int TILE = 32;
    __shared__ int A_tile[TILE][TILE];
    __shared__ int B_tile[TILE][TILE];
    
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by*TILE + ty;
    int col = bx*TILE + tx;
    
    int sum = 0;
    
    // Loop over tiles
    for(int m = 0; m < WIDTH/TILE; m++) {
        // Load tiles into shared memory
        if(row < WIDTH && m*TILE + tx < WIDTH)
            A_tile[ty][tx] = A[row*WIDTH + m*TILE + tx];
        else
            A_tile[ty][tx] = 0;
            
        if(col < WIDTH && m*TILE + ty < WIDTH)
            B_tile[ty][tx] = B[(m*TILE + ty)*WIDTH + col];
        else
            B_tile[ty][tx] = 0;
            
        __syncthreads();
        
        // Compute partial sum for this tile
        for(int i = 0; i < TILE; i++) {
            sum += A_tile[ty][i] * B_tile[i][tx];
        }
        __syncthreads();
    }
    
    // Write result
    if(row < WIDTH && col < WIDTH)
        C[row*WIDTH + col] = sum;
}