#include <iostream>
#include "../utils/image_data_gen.h"
#include <conv_filters.h>

// CUDA kernel for 2D convolution with multiple channels and filters
__global__ void conv2d_kernel(const unsigned char* input,
    unsigned char* output,
    const int* filter,
    int width, 
    int height, 
    int channels,
    int filterIdx) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockDim.x * blockIdx.x + tx;
    int row = blockDim.y * blockIdx.y + ty;
    
    // Check bounds
    if (row >= height || col >= width) return;
    
    // Process each channel
    for (int c = 0; c < channels; c++) {
        float Pval = 0.0f;
        
        // Apply convolution filter
        for (int fRow = 0; fRow < 2*R+1; fRow++) {
            for (int fCol = 0; fCol < 2*R+1; fCol++) {
                int inRow = row + fRow - R;
                int inCol = col + fCol - R;
                
                // Handle boundary conditions with zero padding
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                    int inputIdx = (inRow * width + inCol) * channels + c;
                    int filterIdx_local = fRow * (2*R+1) + fCol;
                    Pval += input[inputIdx] * filter[filterIdx_local];
                }
            }
        }
        
        // Store result with proper indexing for multiple filters
        int outputIdx = ((filterIdx * height + row) * width + col) * channels + c;
        output[outputIdx] = (unsigned char)min(max((int)Pval, 0), 255);
    }
}




int main() {
    try {
        // Load image
        ImageHandler imgHandler("../assets/lokiandthor.png");
        int width = imgHandler.getWidth();
        int height = imgHandler.getHeight();
        int channels = imgHandler.getImage().channels();
        unsigned char* imgData = imgHandler.getImageData();

        std::cout << "Image loaded: " << width << "x" << height << " channels: " << channels << std::endl;

        
        print_filters();

        // Allocate device memory
        unsigned char *d_input, *d_output;
        int *d_filters;

        size_t imgSize = width * height * channels * sizeof(unsigned char);
        size_t outputSize = NUM_FILTERS * width * height * channels * sizeof(unsigned char);
        size_t filterSize = FILTER_SIZE * FILTER_SIZE * sizeof(int);
        
        cudaMalloc(&d_input, imgSize);
        cudaMalloc(&d_output, outputSize);
        cudaMalloc(&d_filters, filterSize);
        // Copy input image to device
        cudaMemcpy(d_input, imgData, imgSize, cudaMemcpyHostToDevice);

        // Set up grid and block dimensions
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                      (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Process each filter
        for (int filterIdx = 0; filterIdx < NUM_FILTERS; filterIdx++) {
            std::cout << "Processing filter " << filterIdx << std::endl;
            
            // Get current filter and flatten it
            const int (*currentFilter)[FILTER_SIZE] = FILTERS[filterIdx];
            int flatFilter[FILTER_SIZE * FILTER_SIZE];
            
            for (int i = 0; i < FILTER_SIZE; ++i) {
                for (int j = 0; j < FILTER_SIZE; ++j) {
                    flatFilter[i * FILTER_SIZE + j] = currentFilter[i][j];
                }
            }
            
            // Copy filter to device
            cudaMemcpy(d_filters, flatFilter, filterSize, cudaMemcpyHostToDevice);

           for(int i=0;i<1000000;i++){

            // Launch kernel
            conv2d_kernel<<<numBlocks, threadsPerBlock>>>(
                d_input, d_output, d_filters, width, height, channels, filterIdx);
            
           } 
            // Check for kernel errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Kernel launch failed for filter " << filterIdx 
                         << ": " << cudaGetErrorString(err) << std::endl;
                continue;
            }
            cudaDeviceSynchronize();
        }

        // Copy results back to host
        unsigned char* allOutputs = new unsigned char[outputSize];
        cudaMemcpy(allOutputs, d_output, outputSize, cudaMemcpyDeviceToHost);

        // Save each filter result as a separate image
        for (int filterIdx = 0; filterIdx < NUM_FILTERS; filterIdx++) {
            unsigned char* currentOutput = &allOutputs[filterIdx * width * height * channels];
            
            // Create filename
            std::string filename = "conv_output_filter_" + std::to_string(filterIdx) + ".png";
            
            // If input is grayscale but we need RGB output
            if (channels == 1) {
                unsigned char* rgbOutput = new unsigned char[width * height * 3];
                for (int i = 0; i < width * height; ++i) {
                    rgbOutput[3*i + 0] = currentOutput[i]; 
                    rgbOutput[3*i + 1] = currentOutput[i]; 
                    rgbOutput[3*i + 2] = currentOutput[i]; 
                }
                imgHandler.saveImage(rgbOutput, filename, 3);
                delete[] rgbOutput;
            } else {
                imgHandler.saveImage(currentOutput, filename, channels);
            }
            
            std::cout << "Saved: " << filename << std::endl;
        }

        // Print first few values for debugging
        std::cout << "First 20 values of filter 0 output: ";
        for (int i = 0; i < 20 && i < width * height * channels; ++i) {
            std::cout << static_cast<int>(allOutputs[i]) << " ";
        }
        std::cout << std::endl;

        // Cleanup
        delete[] allOutputs;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_filters);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}