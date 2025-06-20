#ifndef KERNELS_CUH
#define KERNELS_CUH


__global__ void conv2d_kernel(const unsigned char* input,
    unsigned char* output,
    const int* filter,
    int width, 
    int height, 
    int filterIdx);

__global__ void conv2d_kernel_tiled(const unsigned char* input,
    unsigned char* output,
    const int* filter,
    int width, 
    int height, 
    int filterIdx);

#endif
