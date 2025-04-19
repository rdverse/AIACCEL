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
    printf("%d thread and %d block\n", blockIdx.x , threadIdx.x);
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

    int threads = n;
    // 2. Launch kernels to do vector addition
    printf("%d",threads);
    // baiscally one block and n threads
    vecAddKernel<<<ceil(n/threads), threads>>>(A_d, B_d, C_d, n);

    cudaDeviceSynchronize(); // Ensures kernel execution finishes before copying data

    // 3. Free data on memory and free decvice vectors
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
 
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}


int main(){

// 1. Allocate host memory for A, B, C
int size=60; 
float A[size];// = {1.0f,2.0f,3.0f,4.0f,5.0f,18.0f};
float B[size];// = {1.0f,2.0f,3.0f,4.0f,6.0f,10.0f};
float C[size];  // declared on the host
int nthreads = size/2; // for data gen
float *A_d, *B_d;

// Allocation on device
cudaMalloc(&A_d, size*sizeof(float));
cudaMalloc(&B_d, size*sizeof(float));
// now we have allocation on device
// threads and blocks specification for gen kernel
simple_1d_gen<<<ceil(size/nthreads), nthreads>>>(A_d, B_d, size);

// generate data on device and then copy to host
// this will be moved to device again for the add kernel
// this is not optimal but just for practice
cudaDeviceSynchronize(); 
// data is generated, now copy device to host
cudaMemcpy(A, A_d, size*sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(B, B_d, size*sizeof(float), cudaMemcpyDeviceToHost);
return 0;
for(int i = 0; i < size; i++){
    printf("%f ", A[i]);
    printf("%f ", B[i]);
    printf("%f ", C[i]);
    }
    // if the size is increased just for C then expect some 0s and garbade values
    vecAdd(A, B, C, size);

    //for(int i=0;i<10;i++){
    //vecAdd(A, B, C, size);

    for (int i = 0; i < size; i++) {
        printf("%d : %f \n",i, C[i]);
    }

return 0;

}
