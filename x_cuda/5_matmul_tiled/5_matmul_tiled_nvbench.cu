#include <cuda_runtime.h>
#include <data_gen.h>
#include <iostream>
#include <nvbench/nvbench.cuh>
#include <nvbench/main.cuh>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


// __global__ void matmul_kernel(int* A, int* B, int* C, int WIDTH) {
//     // consider M == N == K
//     int row = blockIdx.y*blockDim.y + threadIdx.y;
//     int col = blockIdx.x*blockDim.x + threadIdx.x;

//     if (row < WIDTH && col < WIDTH) {
//         int sum = 0;
//         for (int i = 0; i < WIDTH; i++) {
//             sum += A[row*WIDTH + i] * B[i*WIDTH + col];
//         }
//         C[row*WIDTH + col] = sum;
//     }
// }


// kernel with 2D TILES - compiler takes care of striding in memory
__global__ void matmul_kernel_tiled(int* A, int* B, int* C, int WIDTH, int TILE_WIDTH){
    // consider M == N == K
    const int TILE = 8;
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

// TODO: 1d TIle - we specify the tile size during runtime
// use this for dynamically specifying tile size 
// the shared memory needs this information during compile time
// therefore we need these extra steps
// other option is to use template
// __global__ void matmul_kernel_tiled_T(int* A, int* B, int* C, int WIDTH, int TILE_WIDTH){
//     extern __shared__ int mem[];

// }


void matmul_benchmark(nvbench::state& state){
    const int WIDTH = state.get_int64("WIDTH");
    const int TILE = 8;//state.get_int64("TILE");

    int* A_d;
    int* B_d;
    int* C_d;

    cudaMalloc(&A_d, WIDTH*WIDTH*sizeof(int));
    cudaMalloc(&B_d, WIDTH*WIDTH*sizeof(int));
    cudaMalloc(&C_d, WIDTH*WIDTH*sizeof(int));

    // Initialize data directly on GPU
    int numThreads = 256;
    int numBlocks = (WIDTH*WIDTH + numThreads - 1) / numThreads;
    simple_1d_gen<<<numBlocks, numThreads>>>(A_d, nullptr, WIDTH*WIDTH);
    
    numBlocks = (WIDTH*WIDTH + numThreads - 1) / numThreads;
    simple_1d_gen<<<numBlocks, numThreads>>>(nullptr, B_d, WIDTH*WIDTH);
    
    cudaDeviceSynchronize();
    
    // Use 16x16 block size for non-tiled kernel
    dim3 block(TILE, TILE);
    dim3 grid((WIDTH + TILE - 1) / TILE, (WIDTH + TILE - 1) / TILE);
    int a_size = TILE*TILE*sizeof(int);
    state.exec([&](nvbench::launch& launch) {
        //matmul_kernel_tiled<<<grid, block, a_size, launch.get_stream()>>>(A_d, B_d, C_d, WIDTH, TILE);
        matmul_kernel_tiled<<<grid, block, 0, launch.get_stream()>>>(A_d, B_d, C_d, WIDTH, TILE);

    
    });
     
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


NVBENCH_BENCH(matmul_benchmark)
    .add_int64_axis("WIDTH", {2048})
    //.add_int64_axis("TILE", {8,16,32,64,128}) TODO , currently TILE is hardcoded
    .set_timeout(300);