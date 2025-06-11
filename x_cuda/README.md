<img src="assets/lokiandthor_cuda.png" width="250" alt="Description">

sudo apt update
sudo apt install libopencv-dev

# CUDA LEARNING AND EXPERIMENTS 

| Index | Directory                          | Concepts                          | Explorations                                                                 |
|-------|------------------------------------|------------------------------------|------------------------------------------------------------------------------|
| 0     | `x_cuda/0_helloworld`             | CUDA Basics, Thread Hierarchy     | Understanding `threadIdx`, `blockIdx`, `blockDim`, and `gridDim`.            |
| 1     | `x_cuda/1_vectoradd_make`         | Vector Addition, Memory Transfers | Host-to-Device and Device-to-Host memory transfers, kernel execution.        |
| 2     | `x_cuda/2_slidingwindowmul3_sharedmem` | Shared vs Global Memory Access    | Performance comparison of shared vs global memory for sliding window kernels.|
| 3     | `x_cuda/3_imageedit_grids_bw`     | Image Processing, OpenCV, Grids, CUDA    | Compare performance of fused kernel with non-fused kernel for grayscale and blur kernels.               |
| 4     | `x_cuda/4_matmul`                 | Simple matmul                     | Using nvbench for benchmarking kernels |   
| 5     | `x_cuda/5_matmul_tiled`           | Shared memory, tiling             | Tiled matrix multiplication, thread/block limits |