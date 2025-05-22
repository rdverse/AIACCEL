#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <nvbench/nvbench.cuh>
#include <data_gen.h>

/*
NVBench-based matrix multiplication benchmark
This benchmark measures the performance of a simple CUDA matrix multiplication implementation
across different matrix sizes. The matrices are initialized with sequential values for verification.
*/

__global__ void matmul_kernel(int* A, int* B, int* C, int M, int N, int K) {
    // Calculate the row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int sum = 0;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];  // Fixed indexing
        }
        C[row * N + col] = sum;
    }
}

void matmul_benchmark(nvbench::state& state) {
    // Get matrix dimensions from state
    const int M = state.get_int64("M");
    const int N = state.get_int64("N");
    const int K = state.get_int64("K");

    // Allocate device memory
    int* MAT_A_d;
    int* MAT_B_d;
    int* MAT_C_d;
    cudaMalloc(&MAT_A_d, M * K * sizeof(int));
    cudaMalloc(&MAT_B_d, K * N * sizeof(int));
    cudaMalloc(&MAT_C_d, M * N * sizeof(int));

    // Initialize data directly on GPU
    int numThreads = 256;
    int numBlocks = (M * K + numThreads - 1) / numThreads;
    simple_1d_gen<<<numBlocks, numThreads>>>(MAT_A_d, nullptr, M * K);
    
    numBlocks = (K * N + numThreads - 1) / numThreads;
    simple_1d_gen<<<numBlocks, numThreads>>>(nullptr, MAT_B_d, K * N);
    
    cudaDeviceSynchronize();

    // Configure kernel launch parameters
    const int TILE_SIZE = 16;
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Run benchmark
    state.exec([&](nvbench::launch& launch) {
        matmul_kernel<<<gridDim, blockDim, 0, launch.get_stream()>>>(
            MAT_A_d, MAT_B_d, MAT_C_d, M, N, K);
    });

    // Cleanup
    cudaFree(MAT_A_d);
    cudaFree(MAT_B_d);
    cudaFree(MAT_C_d);
}

// Register benchmark
NVBENCH_BENCH(matmul_benchmark)
    .add_int64_axis("M", {1024, 2048, 4096}) // Much larger matrices
    .add_int64_axis("N", {1024, 2048, 4096})
    .add_int64_axis("K", {1024, 2048, 4096})
    .set_timeout(300); // 5 minute timeout per measurement