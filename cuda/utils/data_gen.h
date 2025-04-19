#ifndef data_gen
#define data_gen


// a bunch of data gen kernels

__global__ void rand_1d_gen(float *data, int n, int seed);

__global__ void rand_2d_gen(float *data, int n, int m, int seed);

__global__ void simple_1d_gen(float *a, float *b, int n);

#endif