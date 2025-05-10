
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
} /

const int N = 256; // 1D array size
const int nblocks =8; //  matrix
const int nthreads = N/nblocks; // Tile size for shared memory//

//-----------------------------------------------------------
// Kernel 1: Global memory only (no shared memory)
//-----------------------------------------------------------
__global__ void no_shared_memory_kernel(float* input, float* output, int width) {
    
    int bdim = blockDim.x;
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int prev_tx, next_tx;
    int prev_bx, next_bx;
    int prev_data, next_data; 
    if (tx - 1 >= 0) {
        prev_tx = tx - 1;
        prev_bx = bx;
        prev_data = input[bdim*bx + prev_tx];

    } else { //tx<0
        prev_tx = nthreads - 1;
        prev_bx = bx > 0 ? bx - 1 : nblocks - 1;
        prev_data = input[bdim*prev_bx + prev_tx]; // updated to use prev_bx
    }

    if (tx + 1 < nthreads) {
        next_tx = tx + 1;
        next_bx = bx;
        next_data = input[bdim*bx + next_tx];
    } else {
        next_tx = 0;
        next_bx = bx + 1 < nblocks ? bx + 1 : 0; // wrap around to 0
        next_data = input[bdim*next_bx + next_tx]; // updated to use next_bx
    }

    output[bdim*bx + tx] = input[bdim*bx + tx] * prev_data * next_data; 

}

//-----------------------------------------------------------
// Kernel 2: Using shared memory (manual tiling)
//-----------------------------------------------------------
__global__ void shared_memory_kernel(float* input, float* output, int width) {

    // okay thre are a few alternatives here: 
    // 1. load the previous and next data into shared memory 
    // caveat is that we need more complex data structure to store the data with metadata
    // 2. only load current, previous and next data into shared memory
    // Will cause a lot of trashing in shared memory
    // maybe good for extremely small tiles
    // 3. load the whole tile into shared memory
    // bad for small tiles - less data reuse for shared memory (more edge cases)
    // 4. Is there some circular buffer that we can use to store the data? 

    __shared__ float tile[nthreads]; // shared mem is allocated per block for 2070 super - 48kb
    int bdim = blockDim.x;
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    // load data into shared memory
    tile[tx] = input[bx * bdim + tx];
     
    __syncthreads(); // Ensure data is fully loaded

    // get local data, next, previous data
    int prev_tx, next_tx;
    int prev_bx, next_bx;
    int prev_data, next_data; 

    if (tx - 1 >= 0) {
        prev_tx = tx - 1;
        prev_data = tile[prev_tx];
    } else { //tx<0
        // need data from last block
        prev_tx = nthreads - 1;
        prev_bx = bx > 0 ? bx - 1 : nblocks - 1; // wrap around to nblocks-1
        prev_data = input[(prev_bx) * bdim + prev_tx]; // updated to use prev_bx
    }
    if (tx + 1 < bdim) {
        next_tx = tx + 1;
        next_bx = bx;
        next_data = tile[tx + 1];
    } else { //tx is the last thread
        next_tx = 0;
        next_bx = bx + 1 < nblocks ? bx + 1 : 0; // wrap around to 0
        next_data = input[bdim * next_bx + next_tx]; // updated to use next_bx
    }
    // Read from shared memory
    output[bdim * bx + tx] =  prev_data * tile[tx] * next_data;
}

//-----------------------------------------------------------
// Main Function
//-----------------------------------------------------------
int main() {

    size_t bytes = N * sizeof(float);
    float *d_input_shared;
    float *d_input_notshared;
    float *d_output_notshared;
    float *d_output_shared = new float[N];
    float* h_inputshared = new float[N];
    float* h_inputnotshared = new float[N];
    float* h_outputnotshared = new float[N];
    float* h_outputshared = new float[N];

    std::cout << "Total threads: " << nthreads*nblocks << std::endl;
    std::cout << "Threads per block: " << nthreads << std::endl;
    std::cout << "Number of blocks: " << nblocks << std::endl;
    std::cout << "data length: " << N << std::endl;
    std::cout << "Size per block: " << bytes / nblocks / 1024 << "Kb" << std::endl;
    std::cout << "Total size: " << bytes / 1024 << "Kb" << std::endl;

    cudaDeviceProp deviseProp;
    int devise;
    cudaGetDevice(&devise);
    cudaGetDeviceProperties(&deviseProp, devise);
    std::cout << "Device: " << deviseProp.name << std::endl;
    std::cout << "Compute capability: " << deviseProp.major << "." << deviseProp.minor << std::endl;

    cudaMalloc((void **)&d_input_shared, bytes);
    cudaMalloc((void **)&d_output_shared, bytes);  

    //compile time cast for implicit conversion
    for (int i = 0; i < N; i++) h_inputshared[i] = static_cast<float>(float(i));
    for (int i = 0; i < N; i++) h_inputnotshared[i] = static_cast<float>(float(i));

    // Initialize input data - this will copy it to global or l2
    cudaMemcpy(d_input_shared, h_inputshared, bytes, cudaMemcpyHostToDevice);

    // Timers
    float time_no_shared = 0.0f, time_shared = 0.0f;
    std::cout << "Results on :" << deviseProp.name << std::endl;
    
    //-----------------------------------------------------------
    // Run shared-memory kernel
    //-----------------------------------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    shared_memory_kernel<<<nblocks, nthreads>>>(d_input_shared, d_output_shared, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_shared, start, stop);
    cudaMemcpy(h_outputshared, d_output_shared, bytes, cudaMemcpyDeviceToHost); 
    // print output not shared
    // std::cout << "Output for shared memory kernel: ";
    // for (int i = 0; i < N; i++) {
    //     std::cout << h_outputshared[i] << " ";
    // }
    // std::cout << std::endl;
    
    auto sum = [](float *output) {
        float total = 0.0f;
        for (int i = 0; i < N; i++) {
            total += output[i];
        }
        return total;
    };

    auto output_sum_shared = sum(h_outputshared);

    delete[] h_inputshared;
    delete[] h_outputnotshared;
    cudaFree(d_input_shared);
    cudaFree(d_output_shared);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    //-----------------------------------------------------------
    // Run no-shared-memory kernel
    //-----------------------------------------------------------

    cudaMalloc((void **)&d_input_notshared, bytes);
    cudaMalloc((void **)&d_output_notshared, bytes);
    cudaMemcpy(d_input_notshared, h_inputnotshared, bytes, cudaMemcpyHostToDevice);
    cudaEvent_t start_notshared, stop_notshared;
    cudaEventCreate(&start_notshared);
    cudaEventCreate(&stop_notshared);
    cudaEventRecord(start_notshared);

    no_shared_memory_kernel<<<nblocks, nthreads>>>(d_input_notshared, d_output_notshared, N);
    
    cudaEventRecord(stop_notshared);
    cudaEventSynchronize(stop_notshared);
    cudaEventElapsedTime(&time_no_shared, start_notshared, stop_notshared);
    cudaMemcpy(h_outputnotshared, d_output_notshared, bytes, cudaMemcpyDeviceToHost);
    auto output_sum_notshared = sum(h_outputnotshared); 
    
    // std::cout << "Output for not shared memory kernel: ";
    // for (int i = 0; i < N; i++) {
    //     std::cout << h_outputnotshared[i] << " ";
    // }
    // std::cout << std::endl;
    
    delete[] h_outputshared;
    delete[] h_outputnotshared;
    cudaFree(d_input_notshared);
    cudaFree(d_output_notshared);
    cudaEventDestroy(start_notshared);
    cudaEventDestroy(stop_notshared);
    //-----------------------------------------------------------
    // Print results
    //-----------------------------------------------------------
    std::cout << " output value for not shared: " << output_sum_notshared << std::endl;
    std::cout << " output value for shared: " << output_sum_shared << std::endl;

    std::cout << "  [NO Shared Memory]  Time = " << time_no_shared << " ms" << std::endl;
    std::cout << "  [WITH Shared Memory] Time = " << time_shared << " ms" << std::endl;
    std::cout << "  Speedup for shared = " << (time_no_shared / time_shared) << "x" << std::endl;

    return 0;
}



// Total threads: 256
// Threads per block: 64
// Number of blocks: 4
// data length: 256
// Size per block: 0Kb
// Total size: 1Kb
// Device: NVIDIA GeForce RTX 2070 SUPER
// Compute capability: 7.5
// Results on :NVIDIA GeForce RTX 2070 SUPER
//  output value for not shared: 1.04876e+09
//  output value for shared: 1.04876e+09
//   [NO Shared Memory]  Time = 0.017152 ms
//   [WITH Shared Memory] Time = 13.6808 ms
//   Speedup for shared = 0.00125372x


// Total threads: 256
// Threads per block: 256
// Number of blocks: 1
// data length: 256
// Size per block: 1Kb
// Total size: 1Kb
// Device: NVIDIA GeForce RTX 2070 SUPER
// Compute capability: 7.5
// Results on :NVIDIA GeForce RTX 2070 SUPER
//  output value for not shared: 1.04876e+09
//  output value for shared: 1.04876e+09
//   [NO Shared Memory]  Time = 0.015968 ms
//   [WITH Shared Memory] Time = 0.157696 ms
// //   Speedup for shared = 0.101258x


// Total threads: 256
// Threads per block: 32
// Number of blocks: 8
// data length: 256
// Size per block: 0Kb
// Total size: 1Kb
// Device: NVIDIA GeForce RTX 2070 SUPER
// Compute capability: 7.5
// Results on :NVIDIA GeForce RTX 2070 SUPER
//  output value for not shared: 1.04876e+09
//  output value for shared: 1.04876e+09
//   [NO Shared Memory]  Time = 0.007168 ms
//   [WITH Shared Memory] Time = 14.2863 ms
//   Speedup for shared = 0.000501739x