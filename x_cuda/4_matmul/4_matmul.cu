#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
//#include <nvbench/nvbench.cuh>
#include <data_gen.h>
/*
NVBench is a library for benchmarking CUDA kernels.
Collecting the time run, memory usage, and other metrics is requiring large amount of boilerplate code.
One of the main concerns is are caches being flushed or not - see example 3 with image processing.
*/


__global__ void matmul_kernel(int* A, int* B, int* C, int M, int N, int K) {
    // Calculate the row and column index of the element to be computed
    // A dims M*K
    // B dims K*N
    // C dims M*N
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int sum = 0.0f;
        for (int i=0;i<K;i++){
            sum += A[row*M + i] * B[i*K + col];
        }
        C[row*N + col] = sum;
    }

}


int main(){

    int M = 2;//4096*4;
    int N = 2;//4096*4;
    int K = 2;//4096*4;

    const int TILE_SIZE = 16;

    int* MAT_A_h = new int[M * K]; //size M*K
    int* MAT_B_h = new int[K * N]; //size K*N
    int* MAT_C_h = new int[M * N]; //size M*N

    int* MAT_A_d;
    int* MAT_B_d;
    int* MAT_C_d;

    cudaMalloc((void**)&MAT_A_d, M * K * sizeof(int));
    cudaMalloc((void**)&MAT_B_d, K * N * sizeof(int));
    cudaMalloc((void**)&MAT_C_d, M * N * sizeof(int));

    cudaMemcpy(MAT_A_d, MAT_A_h, M*K*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(MAT_B_d, MAT_B_h, K*N*sizeof(int), cudaMemcpyHostToDevice);

    // load data into A_d and B_d
    int numThreads = 4;
    int numBlocks = (M*N + numThreads - 1) / numThreads;
    simple_1d_gen<<<numBlocks, numThreads>>>(MAT_A_d, MAT_B_d, M*N);
    cudaDeviceSynchronize();
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_kernel<<<gridDim, blockDim>>>(MAT_A_d, MAT_B_d, MAT_C_d, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %f ms\n", milliseconds);

    // print out C_h
    cudaMemcpy(MAT_C_h, MAT_C_d, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(MAT_A_h, MAT_A_d, M*K*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(MAT_B_h, MAT_B_d, K*N*sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "MAT_A_h: " << std::endl;
    for (int i=0;i<M;i++){
        for (int j=0;j<K;j++){
            printf("%d ", MAT_A_h[i*K + j]);
        }
        printf("\n");
    }
    std::cout<< "MAT_B_h: " << std::endl;
    for (int i=0;i<K;i++){
        for (int j=0;j<N;j++){
            printf("%d ", MAT_B_h[i*N + j]);
        }
        printf("\n");
    }
    std::cout<< "MAT_C_h: " << std::endl;
    for (int i=0;i<M;i++){
        for (int j=0;j<N;j++){
            printf("%d ", MAT_C_h[i*N + j]);
        }
        printf("\n");
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(MAT_A_d);
    cudaFree(MAT_B_d);
    cudaFree(MAT_C_d);
    delete[] MAT_A_h;
    delete[] MAT_B_h;
    delete[] MAT_C_h;
    return 0;
}