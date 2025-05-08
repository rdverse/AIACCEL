// Compare shared memory vs global memory access performance

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(err); \
    } \
}

const int N = 1024; //  matrix
const int TILE_SIZE = 64; // Tile size for shared memory

//-----------------------------------------------------------
// Kernel 1: Global memory only (no shared memory)
//-----------------------------------------------------------
__global__ void no_shared_memory_kernel(float* input, float* output, int width) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx < width && ty < width) {
        float val = input[ty * width + tx];
        output[ty * width + tx] = val * 2.0f; // Simple computation
    }
}

//-----------------------------------------------------------
// Kernel 2: Using shared memory (manual tiling)
//-----------------------------------------------------------
__global__ void shared_memory_kernel(float* input, float* output, int width) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    if (gx < width && gy < width) {
        // Load from global memory to shared memory
        tile[ty][tx] = input[gy * width + gx];
    }
    __syncthreads(); // Ensure tile is fully loaded

    if (gx < width && gy < width) {
        // Read from shared memory
        output[gy * width + gx] = tile[ty][tx] * 2.0f;
    }
}

//-----------------------------------------------------------
// Main Function
//-----------------------------------------------------------
int main() {
    size_t bytes = N * N * sizeof(float);

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    float* h_input = new float[N * N];
    for (int i = 0; i < N * N; i++) h_input[i] = static_cast<float>(i % 100);

    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Timers
    cudaEvent_t start, stop;
    float time_no_shared = 0.0f, time_shared = 0.0f;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    //-----------------------------------------------------------
    // Run no-shared-memory kernel
    //-----------------------------------------------------------
    CHECK_CUDA(cudaEventRecord(start));
    no_shared_memory_kernel<<<blocks, threads>>>(d_input, d_output, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_no_shared, start, stop));

    //-----------------------------------------------------------
    // Run shared-memory kernel
    //-----------------------------------------------------------
    CHECK_CUDA(cudaEventRecord(start));
    shared_memory_kernel<<<blocks, threads>>>(d_input, d_output, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&time_shared, start, stop));

    //-----------------------------------------------------------
    // Print results
    //-----------------------------------------------------------
    std::cout << "Results on A100:" << std::endl;
    std::cout << "  [NO Shared Memory]  Time = " << time_no_shared << " ms" << std::endl;
    std::cout << "  [WITH Shared Memory] Time = " << time_shared << " ms" << std::endl;
    std::cout << "  Speedup = " << (time_no_shared / time_shared) << "x" << std::endl;

    //-----------------------------------------------------------
    // Clean up
    //-----------------------------------------------------------
    delete[] h_input;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
