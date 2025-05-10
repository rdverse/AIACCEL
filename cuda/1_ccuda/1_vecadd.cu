#include <stdio.h>
#include <cuda.h>
#include <data_gen.h>

// define cuda kernel here
// __device__ vecAdd(*A_d, *B_d){

//     C_d = A_d + B_d;

//     return C_d
// }

// compute C=A+B - elt

__global__ void vecAddKernel(float* a,float* b,float* c,int n){

    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<n){
        c[i]=a[i]+b[i];
    }
    printf("%d thread and %d block\n" , threadIdx.x, blockIdx.x);
}


void vecAdd(float* A_h, float* B_h, float* C_h, int n){

    float *A_d, *B_d, *C_d;  // declared on the host
    int size = n*sizeof(float);

    // part1: allocate device memory for A,B,C
    cudaError_t err = cudaMalloc((void**)&A_d, size);
    // how to control which sm to use?
    // no direct control over sm 
    // does cuda malloc store data in l1 / l2 ? 
    // cudaFuncSetCacheConfig - provides some priority specification
    // allocation on l1/l2 only when kernel called. 
    
    if (err!=cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    }

    cudaMalloc((void**)&C_d, size);
    cudaMalloc((void**)&B_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // cudamallomanaged is another option to coalesce all
    // allocate , copy , and free functions
    int nblocks = 4;
    int nthreads = n/nblocks;

    // 2. Launch kernels to do vector addition
    printf("%d",nthreads);
    // baiscally one block and n threads
    vecAddKernel<<<nblocks,nthreads>>>(A_d, B_d, C_d, n);
    cudaDeviceSynchronize(); // Ensures kernel execution finishes before copying data

    // 3. Free data on memory and free decvice vectors
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    float total = 0;
    for (int i = 0; i < n; i++) {
       total += C_h[i];
   }
    printf("Total: %f\n", total);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


int main(){
// 1. Allocate host memory for A, B, C
int size=128; 
float A[size];// = {1.0f,2.0f,3.0f,4.0f,5.0f,18.0f};
float B[size];// = {1.0f,2.0f,3.0f,4.0f,6.0f,10.0f};
float C[size];  // declared on the host
int n_blocks = 4; // for data gen
int nthreads = size/n_blocks; // for data gen
float *A_d, *B_d;

// Allocation on device
cudaMalloc(&A_d, size*sizeof(float));
cudaMalloc(&B_d, size*sizeof(float));
// now we have allocation on device
// threads and blocks specification for gen kernel
simple_1d_gen<<< n_blocks,nthreads>>>(A_d, B_d, size);

// generate data on device and then copy to host
// this will be moved to device again for the add kernel
// this is not optimal but just for practice
cudaDeviceSynchronize(); 
// data is generated, now copy device to host
cudaMemcpy(A, A_d, size*sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(B, B_d, size*sizeof(float), cudaMemcpyDeviceToHost);
//return 0;
// for(int i = 0; i < size; i++){
//     printf("%f ", A[i]);
//     printf("%f ", B[i]);
//     printf("%f ", C[i]);
//     }
    // if the size is increased just for C then expect some 0s and garbade values
    vecAdd(A, B, C, size);
return 0;

}

// simpledatagen: thread 0, block 1
// simpledatagen: thread 1, block 1
// simpledatagen: thread 2, block 1
// simpledatagen: thread 3, block 1
// simpledatagen: thread 4, block 1
// simpledatagen: thread 5, block 1
// simpledatagen: thread 6, block 1
// simpledatagen: thread 7, block 1
// simpledatagen: thread 8, block 1
// simpledatagen: thread 9, block 1
// simpledatagen: thread 10, block 1
// simpledatagen: thread 11, block 1
// simpledatagen: thread 12, block 1
// simpledatagen: thread 13, block 1
// simpledatagen: thread 14, block 1
// simpledatagen: thread 15, block 1
// simpledatagen: thread 16, block 1
// simpledatagen: thread 17, block 1
// simpledatagen: thread 18, block 1
// simpledatagen: thread 19, block 1
// simpledatagen: thread 20, block 1
// simpledatagen: thread 21, block 1
// simpledatagen: thread 22, block 1
// simpledatagen: thread 23, block 1
// simpledatagen: thread 24, block 1
// simpledatagen: thread 25, block 1
// simpledatagen: thread 26, block 1
// simpledatagen: thread 27, block 1
// simpledatagen: thread 28, block 1
// simpledatagen: thread 29, block 1
// simpledatagen: thread 30, block 1
// simpledatagen: thread 31, block 1
// simpledatagen: thread 0, block 2
// simpledatagen: thread 1, block 2
// simpledatagen: thread 2, block 2
// simpledatagen: thread 3, block 2
// simpledatagen: thread 4, block 2
// simpledatagen: thread 5, block 2
// simpledatagen: thread 6, block 2
// simpledatagen: thread 7, block 2
// simpledatagen: thread 8, block 2
// simpledatagen: thread 9, block 2
// simpledatagen: thread 10, block 2
// simpledatagen: thread 11, block 2
// simpledatagen: thread 12, block 2
// simpledatagen: thread 13, block 2
// simpledatagen: thread 14, block 2
// simpledatagen: thread 15, block 2
// simpledatagen: thread 16, block 2
// simpledatagen: thread 17, block 2
// simpledatagen: thread 18, block 2
// simpledatagen: thread 19, block 2
// simpledatagen: thread 20, block 2
// simpledatagen: thread 21, block 2
// simpledatagen: thread 22, block 2
// simpledatagen: thread 23, block 2
// simpledatagen: thread 24, block 2
// simpledatagen: thread 25, block 2
// simpledatagen: thread 26, block 2
// simpledatagen: thread 27, block 2
// simpledatagen: thread 28, block 2
// simpledatagen: thread 29, block 2
// simpledatagen: thread 30, block 2
// simpledatagen: thread 31, block 2
// simpledatagen: thread 0, block 3
// simpledatagen: thread 1, block 3
// simpledatagen: thread 2, block 3
// simpledatagen: thread 3, block 3
// simpledatagen: thread 4, block 3
// simpledatagen: thread 5, block 3
// simpledatagen: thread 6, block 3
// simpledatagen: thread 7, block 3
// simpledatagen: thread 8, block 3
// simpledatagen: thread 9, block 3
// simpledatagen: thread 10, block 3
// simpledatagen: thread 11, block 3
// simpledatagen: thread 12, block 3
// simpledatagen: thread 13, block 3
// simpledatagen: thread 14, block 3
// simpledatagen: thread 15, block 3
// simpledatagen: thread 16, block 3
// simpledatagen: thread 17, block 3
// simpledatagen: thread 18, block 3
// simpledatagen: thread 19, block 3
// simpledatagen: thread 20, block 3
// simpledatagen: thread 21, block 3
// simpledatagen: thread 22, block 3
// simpledatagen: thread 23, block 3
// simpledatagen: thread 24, block 3
// simpledatagen: thread 25, block 3
// simpledatagen: thread 26, block 3
// simpledatagen: thread 27, block 3
// simpledatagen: thread 28, block 3
// simpledatagen: thread 29, block 3
// simpledatagen: thread 30, block 3
// simpledatagen: thread 31, block 3
// simpledatagen: thread 0, block 0
// simpledatagen: thread 1, block 0
// simpledatagen: thread 2, block 0
// simpledatagen: thread 3, block 0
// simpledatagen: thread 4, block 0
// simpledatagen: thread 5, block 0
// simpledatagen: thread 6, block 0
// simpledatagen: thread 7, block 0
// simpledatagen: thread 8, block 0
// simpledatagen: thread 9, block 0
// simpledatagen: thread 10, block 0
// simpledatagen: thread 11, block 0
// simpledatagen: thread 12, block 0
// simpledatagen: thread 13, block 0
// simpledatagen: thread 14, block 0
// simpledatagen: thread 15, block 0
// simpledatagen: thread 16, block 0
// simpledatagen: thread 17, block 0
// simpledatagen: thread 18, block 0
// simpledatagen: thread 19, block 0
// simpledatagen: thread 20, block 0
// simpledatagen: thread 21, block 0
// simpledatagen: thread 22, block 0
// simpledatagen: thread 23, block 0
// simpledatagen: thread 24, block 0
// simpledatagen: thread 25, block 0
// simpledatagen: thread 26, block 0
// simpledatagen: thread 27, block 0
// simpledatagen: thread 28, block 0
// simpledatagen: thread 29, block 0
// simpledatagen: thread 30, block 0
// simpledatagen: thread 31, block 0
// 320 thread and 1 block
// 1 thread and 1 block
// 2 thread and 1 block
// 3 thread and 1 block
// 4 thread and 1 block
// 5 thread and 1 block
// 6 thread and 1 block
// 7 thread and 1 block
// 8 thread and 1 block
// 9 thread and 1 block
// 10 thread and 1 block
// 11 thread and 1 block
// 12 thread and 1 block
// 13 thread and 1 block
// 14 thread and 1 block
// 15 thread and 1 block
// 16 thread and 1 block
// 17 thread and 1 block
// 18 thread and 1 block
// 19 thread and 1 block
// 20 thread and 1 block
// 21 thread and 1 block
// 22 thread and 1 block
// 23 thread and 1 block
// 24 thread and 1 block
// 25 thread and 1 block
// 26 thread and 1 block
// 27 thread and 1 block
// 28 thread and 1 block
// 29 thread and 1 block
// 30 thread and 1 block
// 31 thread and 1 block
// 0 thread and 2 block
// 1 thread and 2 block
// 2 thread and 2 block
// 3 thread and 2 block
// 4 thread and 2 block
// 5 thread and 2 block
// 6 thread and 2 block
// 7 thread and 2 block
// 8 thread and 2 block
// 9 thread and 2 block
// 10 thread and 2 block
// 11 thread and 2 block
// 12 thread and 2 block
// 13 thread and 2 block
// 14 thread and 2 block
// 15 thread and 2 block
// 16 thread and 2 block
// 17 thread and 2 block
// 18 thread and 2 block
// 19 thread and 2 block
// 20 thread and 2 block
// 21 thread and 2 block
// 22 thread and 2 block
// 23 thread and 2 block
// 24 thread and 2 block
// 25 thread and 2 block
// 26 thread and 2 block
// 27 thread and 2 block
// 28 thread and 2 block
// 29 thread and 2 block
// 30 thread and 2 block
// 31 thread and 2 block
// 0 thread and 3 block
// 1 thread and 3 block
// 2 thread and 3 block
// 3 thread and 3 block
// 4 thread and 3 block
// 5 thread and 3 block
// 6 thread and 3 block
// 7 thread and 3 block
// 8 thread and 3 block
// 9 thread and 3 block
// 10 thread and 3 block
// 11 thread and 3 block
// 12 thread and 3 block
// 13 thread and 3 block
// 14 thread and 3 block
// 15 thread and 3 block
// 16 thread and 3 block
// 17 thread and 3 block
// 18 thread and 3 block
// 19 thread and 3 block
// 20 thread and 3 block
// 21 thread and 3 block
// 22 thread and 3 block
// 23 thread and 3 block
// 24 thread and 3 block
// 25 thread and 3 block
// 26 thread and 3 block
// 27 thread and 3 block
// 28 thread and 3 block
// 29 thread and 3 block
// 30 thread and 3 block
// 31 thread and 3 block
// 0 thread and 0 block
// 1 thread and 0 block
// 2 thread and 0 block
// 3 thread and 0 block
// 4 thread and 0 block
// 5 thread and 0 block
// 6 thread and 0 block
// 7 thread and 0 block
// 8 thread and 0 block
// 9 thread and 0 block
// 10 thread and 0 block
// 11 thread and 0 block
// 12 thread and 0 block
// 13 thread and 0 block
// 14 thread and 0 block
// 15 thread and 0 block
// 16 thread and 0 block
// 17 thread and 0 block
// 18 thread and 0 block
// 19 thread and 0 block
// 20 thread and 0 block
// 21 thread and 0 block
// 22 thread and 0 block
// 23 thread and 0 block
// 24 thread and 0 block
// 25 thread and 0 block
// 26 thread and 0 block
// 27 thread and 0 block
// 28 thread and 0 block
// 29 thread and 0 block
// 30 thread and 0 block
// 31 thread and 0 block
// Total: 16384.000000