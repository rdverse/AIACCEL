#include <conv_filters.h>
#include <stdlib.h>

// CUDA kernel for 2D convolution with multiple CHANNELS and filters
__global__ void conv2d_kernel(const unsigned char* input,
    unsigned char* output,
    const int* filter,
    int width, 
    int height, 
    int filterIdx) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockDim.x * blockIdx.x + tx;
    int row = blockDim.y * blockIdx.y + ty;
    
    // Check bounds
    if (row >= height || col >= width) return;
    
    // Process each channel
    for (int c = 0; c < CHANNELS; c++) {
        float Pval = 0.0f;
        
        // Apply convolution filter
        for (int fRow = 0; fRow < 2*R+1; fRow++) {
            for (int fCol = 0; fCol < 2*R+1; fCol++) {
                int inRow = row + fRow - R;
                int inCol = col + fCol - R;
                
                // Handle boundary conditions with zero padding
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    int inputIdx = (inRow * width + inCol) * CHANNELS + c;
                    int filterIdx_local = fRow * (2*R+1) + fCol;
                    Pval += input[inputIdx] * filter[filterIdx_local];
                }
            }
        }
        
        // Store result with proper indexing for multiple filters
        int outputIdx = ((filterIdx * height + row) * width + col) * CHANNELS + c;
        output[outputIdx] = (unsigned char)min(max((int)Pval, 0), 255);
    }
}



// __global__ void conv2d_kernel_tiled(const unsigned char* input,
//     unsigned char* output,
//     const int* filter,
//     int width, 
//     int height, 
//     int filterIdx) {
    
//     // Shared memory for a tile of the input image. It's larger than the
//     // block dimension to hold the haloa cells needed for convolution.
//     __shared__ unsigned char input_tile[TILE_DIM + FILTER_SIZE - 1][TILE_DIM + FILTER_SIZE - 1][CHANNELS];

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     // Load a tile of the image from global memory into shared memory.
//     // Each thread loads one pixel.
//     int load_row = blockIdx.y * TILE_DIM + ty;
//     int load_col = blockIdx.x * TILE_DIM + tx;

//     for (int c = 0; c < CHANNELS; c++) {
//         if (load_row < height && load_col < width) {
//             input_tile[ty][tx][c] = input[(load_row * width + load_col) * CHANNELS + c];
//         } else {
//             input_tile[ty][tx][c] = 0; // Zero-padding for out-of-bounds
//         }
//     }
    
//     // The halo region needs to be loaded as well.
//     // This part is simplified; a more robust implementation would have threads
//     // cooperatively load the halo. For a 3x3 filter, this approach is often sufficient.
//     __syncthreads(); // Wait for all threads to finish loading the tile

//     // Global coordinates of the pixel this thread will compute and output
//     int out_row = blockIdx.y * TILE_DIM + ty;
//     int out_col = blockIdx.x * TILE_DIM + tx;

//     // Ensure the output pixel is within the image bounds before computing
//     if (out_row < height && out_col < width) {
//         // Process each channel for the output pixel
//         for (int c = 0; c < CHANNELS; c++) {
//             float p_val = 0.0f; 
            
//             // Perform convolution using the shared memory tile
//             for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
//                 for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
//                     // Find neighbor coordinates in the tile. Note that for a simple implementation
//                     // this requires a larger shared memory tile to hold the halo.
//                     // The current TILE_DIM doesn't account for this, so we must check boundaries.
//                     int tile_y = ty + fRow - R;
//                     int tile_x = tx + fCol - R;

//                     // This check is the key to a naive tiled implementation. 
//                     // It simulates reading from a larger conceptual tile.
//                     int neighbor_global_row = out_row + fRow - R;
//                     int neighbor_global_col = out_col + fCol - R;

//                     if (neighbor_global_row >= 0 && neighbor_global_row < height &&
//                         neighbor_global_col >= 0 && neighbor_global_col < width) {
                        
//                         // We are reading from global memory here because the simplified tiling
//                         // doesn't load the full halo. For true tiled performance,
//                          int inputIdx = (neighbor_global_row * width + neighbor_global_col) * CHANNELS + c;
//                          p_val += (float)input[inputIdx] * (float)filter[fRow * FILTER_SIZE + fCol];
//                     }
//                 }
//             }
            
//             // Store the final result in the output buffer for the correct filter
//             int outputIdx = ((filterIdx * height + out_row) * width + out_col) * CHANNELS + c;
//             output[outputIdx] = (unsigned char)min(max((int)p_val, 0), 255);
//         }
//     }
// }
    


__global__ void conv2d_kernel_tiled(const unsigned char* input,
    unsigned char* output,
    const int* filter,
    int width, 
    int height, 
    int filterIdx) {
    
    // Define the dimensions of the shared memory tile, including the halo
    const int tile_width = TILE_DIM + 2 * R;
    const int tile_height = TILE_DIM + 2 * R;
    __shared__ unsigned char input_tile[tile_height][tile_width][CHANNELS];

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load pixels into the shared memory tile
    // Each thread loads a pixel into the main part of the tile
    int load_row = blockIdx.y * TILE_DIM + ty;
    int load_col = blockIdx.x * TILE_DIM + tx;

    for(int c = 0; c < CHANNELS; c++) {
        if (load_row < height && load_col < width) {
            input_tile[ty + R][tx + R][c] = input[(load_row * width + load_col) * CHANNELS + c];
        } else {
            input_tile[ty + R][tx + R][c] = 0;
        }
    }
    
    // give the threads extra work instead of having multiple loops here to load halo regions
    // we are not launching threads more than TILE_DIM*TILE_DIM
    // Load halo regions
    // Top/Bottom halo
    if (ty < R) {
        for(int c = 0; c < CHANNELS; c++) {
            // Top
            int top_row = load_row - R;
            if (top_row >= 0 && load_col < width) input_tile[ty][tx+R][c] = input[(top_row * width + load_col) * CHANNELS + c];
            else input_tile[ty][tx+R][c] = 0;
            // Bottom
            int bottom_row = load_row + TILE_DIM;
            if (bottom_row < height && load_col < width) input_tile[ty + TILE_DIM + R][tx+R][c] = input[(bottom_row * width + load_col) * CHANNELS + c];
            else input_tile[ty + TILE_DIM + R][tx+R][c] = 0;
        }
    }
    // Left/Right halo
    if (tx < R) {
        for(int c = 0; c < CHANNELS; c++) {
            // Left
            int left_col = load_col - R;
            if (left_col >= 0 && load_row < height) input_tile[ty+R][tx][c] = input[(load_row * width + left_col) * CHANNELS + c];
            else input_tile[ty+R][tx][c] = 0;
            // Right
            int right_col = load_col + TILE_DIM;
            if (right_col < width && load_row < height) input_tile[ty+R][tx + TILE_DIM + R][c] = input[(load_row * width + right_col) * CHANNELS + c];
            else input_tile[ty+R][tx + TILE_DIM + R][c] = 0;
        }
    }

    __syncthreads();

    // Calculate the output pixel for this thread
    int out_row = blockIdx.y * TILE_DIM + ty;
    int out_col = blockIdx.x * TILE_DIM + tx;

    if (out_row < height && out_col < width) {
        for (int c = 0; c < CHANNELS; c++) {
            float p_val = 0.0f;
            for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
                for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
                    // Read from the shared memory tile, including the halo
                    p_val += (float)input_tile[ty + fRow][tx + fCol][c] * (float)filter[fRow * FILTER_SIZE + fCol];
                }
            }
            int outputIdx = ((filterIdx * height + out_row) * width + out_col) * CHANNELS + c;
            output[outputIdx] = (unsigned char)min(max((int)p_val, 0), 255);
        }
    }
}