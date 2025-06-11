#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <math.h>


__global__ void matmul_kernel_tiled(int* A, int* B, int* C, int WIDTH, int TILE_WIDTH=0);

__global__ void matmul_kernel(int* A, int* B, int* C, int WIDTH, int TILE_WIDTH=0);

#endif