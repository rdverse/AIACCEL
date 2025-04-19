#include <stdio.h>

// threadIdx.x: Thread index within its block

// blockIdx.x: Block index within the grid

// blockDim.x: Number of threads per block

// gridDim.x: Number of blocks in the grid

__global__ void main_thread_info(){
    if(threadIdx.x==10){
        printf("Hello from block (%d) and thread (%d), (Block dim: %d and grid dim : %d \n)", blockIdx.x, 
            threadIdx.x,
            blockDim.x,
            gridDim.x);
        }
    }

int main(){

    int threadsPerBlock = 1024*5;
    int numbBlocks = 16;

    // Launch kernel with specified grid/block
    main_thread_info<<<numbBlocks, threadsPerBlock>>>();

    // Wait for GPUs to finish execution
    cudaDeviceSynchronize();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads Per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Max Block Size: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    return 0;

}