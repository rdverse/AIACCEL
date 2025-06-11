#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include "kernels.cuh"   
#include "data_gen.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)




//const int TILE = 16;
void matmul_benchmark(){
    const int WIDTH = 128;//state.get_int64("WIDTH");
    //const int TILE = state.get_int64("TILE");
    
    int* A_d;
    int* B_d;
    int* C_d;
    
    printf("Allocating device memory...\n");
    CUDA_CHECK(cudaMalloc(&A_d, WIDTH*WIDTH*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&B_d, WIDTH*WIDTH*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&C_d, WIDTH*WIDTH*sizeof(int)));

    // This will fail and CUDA_CHECK will catch it
    // CUDA_CHECK(cudaMemcpy(A_d, A_h, WIDTH*WIDTH*sizeof(int), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(B_d, B_h, WIDTH*WIDTH*sizeof(int), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(C_d, C_h, WIDTH*WIDTH*sizeof(int), cudaMemcpyHostToDevice));

    int numThreads = 64;
    int numBlocks = (WIDTH*WIDTH + numThreads - 1) / numThreads;
    printf("Launching kernel with %d blocks and %d threads per block\n", numBlocks, numThreads);
    
    // Initialize A and B
    simple_1d_gen<<<numBlocks, numThreads>>>(A_d, B_d, WIDTH*WIDTH);
    CUDA_CHECK(cudaDeviceSynchronize());
    

    // copy data to host and print it
    int* A_h = new int[WIDTH*WIDTH];
    int* B_h = new int[WIDTH*WIDTH];
    int* C_h = new int[WIDTH*WIDTH];
    
    printf("Copying data back to host...\n");
    CUDA_CHECK(cudaMemcpy(A_h, A_d, WIDTH*WIDTH*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(B_h, B_d, WIDTH*WIDTH*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C_h, C_d, WIDTH*WIDTH*sizeof(int), cudaMemcpyDeviceToHost));

    printf("Verifying first 10 elements before matmul:\n");
    for(int i=0; i<10; i++){
    printf("A_h[%d] = %d, B_h[%d] = %d, C_h[%d] = %d\n", i, A_h[i], i, B_h[i], i, C_h[i]);
    }
    fflush(stdout);

    const int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((WIDTH + TILE - 1) / TILE, (WIDTH + TILE - 1) / TILE);
    printf("Launching matrix multiplication kernel...\n");
    matmul_kernel_tiled<<<grid, block>>>(A_d, B_d, C_d, WIDTH);
     
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("Copying results back to host...\n");
    CUDA_CHECK(cudaMemcpy(C_h, C_d, WIDTH*WIDTH*sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("Verifying first 10 elements after matmul:\n");
    for(int i=0; i<10; i++){
        printf("A_h[%d] = %d, B_h[%d] = %d, C_h[%d] = %d\n", i, A_h[i], i, B_h[i], i, C_h[i]);
    }
    fflush(stdout);
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    delete[] A_h;
    delete[] B_h;
    delete[] C_h;
}

// int main(){
//     for(int i=0;i<1000;i++){
//         //printf("i: %d\n", i);
//     matmul_benchmark();
//     }
//     return 0;
// }


int main() {
    matmul_benchmark();
    return 0;
}
