#include "simple_funcs.h"

__host__ void print_gpuspecs(int device_id) {
    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, device_id);

    if (err != cudaSuccess) {
        std::cerr << "Error: Unable to get device properties for device " << device_id
                  << ". CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "GPU Specifications for Device " << device_id << ":" << std::endl;
    std::cout << "  Name: " << deviceProp.name << std::endl;
    std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Registers per Block: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
    std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Threads per SM: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Number of SMs: " << deviceProp.multiProcessorCount << std::endl;
}