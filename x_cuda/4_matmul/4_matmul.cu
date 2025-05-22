#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <data_gen.h>
#include <unistd.h>
#include <math.h>

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
            sum += A[row*K + i] * B[i*N + col];
        }
        C[row*N + col] = sum;
    }
}



int main(){
    // Array of matrix sizes to test
    int sizes[] = {4096, 2048, 1024};
    const int NUM_SIZES = 3;
    const int WARMUP_RUNS[] = {100, 500, 1000}; // Discard first 100 runs as warmup
    const int NUM_RUNS[] = {500,1000,1500};
    const int TILE_SIZE = 16;

    for (int s = 0; s < NUM_SIZES; s++) {
        int M = sizes[s];
        int N = sizes[s];
        int K = sizes[s];
        
        std::cout << "\nMatrix size: " << M << " x " << N << " x " << K << std::endl;
        float run_times[NUM_RUNS[s]];

        int* MAT_A_h = new int[M * K];
        int* MAT_B_h = new int[K * N];
        int* MAT_C_h = new int[M * N];

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

        usleep(5000000);
        for (int run = 0; run < NUM_RUNS[s]; run++) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            cudaEventRecord(start);
            matmul_kernel<<<gridDim, blockDim>>>(MAT_A_d, MAT_B_d, MAT_C_d, M, N, K);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            run_times[run] = milliseconds;
             
            if (run % 100 == 0) {
                std::cout << "Run " << run + 1 << ": " << milliseconds << " ms" << std::endl;
            }
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            usleep(10000);
        }
        
        // Calculate mean and standard deviation for noise calculation (excluding warmup)
        float sum = 0.0f;
        for (int i = WARMUP_RUNS[s]; i < NUM_RUNS[s]; i++) {
            sum += run_times[i];
        }
        float mean = sum / (NUM_RUNS[s] - WARMUP_RUNS[s]);
        
        float sq_sum = 0.0f;
        for (int i = WARMUP_RUNS[s]; i < NUM_RUNS[s]; i++) {
            float diff = run_times[i] - mean;
            sq_sum += diff * diff;
        }
        float std_dev = sqrt(sq_sum / (NUM_RUNS[s] - WARMUP_RUNS[s]));
        float noise_percentage = (std_dev / mean) * 100.0f;
        
        // Calculate average of last 3 runs
        float avg_time = (run_times[NUM_RUNS[s]-3] + run_times[NUM_RUNS[s]-2] + run_times[NUM_RUNS[s]-1]) / 3.0f;

        std::cout << "Total runs: " << NUM_RUNS[s] 
                  << ", Average time (last 3 runs): " << avg_time 
                  << " ms\nNoise: " << noise_percentage 
                  << "% (calculated over " << NUM_RUNS[s] - WARMUP_RUNS[s] 
                  << " stable runs)" << std::endl;


        cudaFree(MAT_A_d);
        cudaFree(MAT_B_d);
        cudaFree(MAT_C_d);
        delete[] MAT_A_h;
        delete[] MAT_B_h;
        delete[] MAT_C_h;
    }
    return 0;
}