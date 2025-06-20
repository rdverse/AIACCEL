#include <iostream>
#include "../utils/image_data_gen.h"
#include <conv_filters.h>
#include <kernels.cuh>
#include <cuda_runtime.h>


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


int main() {
    try {
        // Load image
        ImageHandler imgHandler("../assets/lokiandthor.png");
        int width = imgHandler.getWidth();
        int height = imgHandler.getHeight();
        //int CHANNELS = imgHandler.getImage().channels();
        unsigned char* imgData = imgHandler.getImageData();

        std::cout << "Image loaded: " << width << "x" << height << " channels: " << CHANNELS << std::endl;

        print_filters();

        // Allocate device memory
        unsigned char *d_input, *d_output;
        int *d_filters;

        size_t imgSize = width * height * CHANNELS * sizeof(unsigned char);
        size_t outputSize = NUM_FILTERS * width * height * CHANNELS * sizeof(unsigned char);
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

           for(int i=0;i<1;i++){

            // Launch the simple, non-tiled kernel which is guaranteed to be correct
            conv2d_kernel<<<numBlocks, threadsPerBlock>>>(
                d_input, d_output, d_filters, width, height, filterIdx);

            
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
            unsigned char* currentOutput = &allOutputs[filterIdx * width * height * CHANNELS];
            
            // Create filename
            std::string filename = "conv_output_filter_" + std::to_string(filterIdx) + ".png";
            
            // If input is grayscale but we need RGB output
            if (CHANNELS == 1) {
                unsigned char* rgbOutput = new unsigned char[width * height * 3];
                for (int i = 0; i < width * height; ++i) {
                    rgbOutput[3*i + 0] = currentOutput[i]; 
                    rgbOutput[3*i + 1] = currentOutput[i]; 
                    rgbOutput[3*i + 2] = currentOutput[i]; 
                    #pragma message ("Warning : The output is not in rgb format")
                }
                imgHandler.saveImage(rgbOutput, filename, 3);
                delete[] rgbOutput;
            } else {
                imgHandler.saveImage(currentOutput, filename, CHANNELS);
            }
            
            std::cout << "Saved: " << filename << std::endl;
        }

        // Print first few values for debugging
        std::cout << "First 20 values of filter 0 output: ";
        for (int i = 0; i < 20 && i < width * height * CHANNELS; ++i) {
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