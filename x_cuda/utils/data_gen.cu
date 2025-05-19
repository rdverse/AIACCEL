#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <data_gen.h>
#include <stdio.h>


__global__ void simple_1d_gen(int *a, int *b, int n)
{
    //a bit of an overkill to do this on the device
    // but the name of directory is cuda
    //  :)
    // For debugging, printing, and other purposes better to do it on host rather
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<int>(idx);
        b[idx] = static_cast<int>(idx+1);}
    printf("simpledatagen: thread %d, block %d\n", threadIdx.x, blockIdx.x);
}

__global__ void simple_1d_gen(float *a, float *b, int n)
{
    //a bit of an overkill to do this on the device
    // but the name of directory is cuda
    //  :)
    // For debugging, printing, and other purposes better to do it on host rather
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = static_cast<float>(idx);
        b[idx] = static_cast<float>(idx+1);}
    printf("simpledatagen: thread %d, block %d\n", threadIdx.x, blockIdx.x);
}