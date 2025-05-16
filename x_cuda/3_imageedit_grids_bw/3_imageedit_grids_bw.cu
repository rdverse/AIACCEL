#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <simple_funcs.h>
#include <image_data_gen.h>


__global__ void grayscale_kernel(unsigned char* image, unsigned char* gray_image, int width, int height){
    /*
    This kernel converts an image to grayscale.
    Each thread processes one pixel of the image (assumed to have rgb channels).
    */
    int x_coord = 0;//blockIdx.x * blockDim.x + threadIdx.x;
    int y_coord = 0;//blockIdx.y * blockDim.y + threadIdx.y;

    if (x_coord < width && y_coord < height) {
        int pixel_index = (y_coord*width + x_coord)*3;
        unsigned char r = image[pixel_index];
        unsigned char g = image[pixel_index + 1];
        unsigned char b = image[pixel_index + 2];
        // fewer ops - 2 add and 1 divide
        unsigned char gray = (r + g + b) / 3;
        // more ops - 2 add and 3 multiply
        //unsigned char gray = (0.3 * r + 0.59 * g + 0.11 * b); 
        gray_image[pixel_index] = gray;
        gray_image[pixel_index + 1] = gray;
        gray_image[pixel_index + 2] = gray;     

        // printf("Grayscale conversion: thread %d, block %d, pixel (%d, %d), gray value %d\n", 
        //     threadIdx.x, blockIdx.x, x_coord, y_coord, gray);
        // // print rgb values
        // printf("Grayscale conversion: thread %d, block %d, pixel (%d, %d), rgb values (%d, %d, %d)\n", 
        //     threadIdx.x, blockIdx.x, x_coord, y_coord, r, g, b);
    }
}


int main() {
    // print gpu specs
    print_gpuspecs();
    // Initialize the ImageHandler with file paths
    ImageHandler handler;
    // Access raw image data
    
    int width = handler.getWidth();
    int height = handler.getHeight();

    unsigned char* d_image;
    unsigned char* d_gray_image;
    unsigned char* d_blurred_image;
    
    //unsigned char* h_image = new unsigned char[width * height * 3];
    unsigned char* h_image = handler.getImageData();
    unsigned char* h_gray_image = new unsigned char[width * height * 3];
    unsigned char* h_blurred_image= new unsigned char[width * height * 3];



    cudaMalloc((void **)&d_image, width * height * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&d_gray_image, width * height * 3 * sizeof(unsigned char));

    cudaMemcpy(d_image, h_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // for (int i = 0; i < 100; i++) {
    //     std::cout << "d_image[" << i << "] = " << (int)h_image[i] << std::endl;
    // }
    // return 0; 

    dim3 grid(ceil(width / 16.0), ceil(height / 16.0), 1);
    dim3 block(16, 16, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // check contents of d_image
   for (int i = 0; i < 10000; i++) {
    //printf("[%d] \n", i);
    grayscale_kernel<<<grid, block>>>(d_image, d_gray_image, width, height);
    } 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Grayscale conversion took " << milliseconds << " milliseconds." << std::endl;
    
    // Copy the result back to the host
    cudaMemcpy(h_gray_image, d_gray_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // check contents of h_gray_image
    // for (int i = 0; i < 10; i++) {
    //     std::cout << "h_gray_image[" << i << "] = " << (int)h_gray_image[i] << std::endl;
    // }
    
    handler.saveImage(h_gray_image, "gray_image.png");
    handler.saveImage(h_image, "color_image.png");

    cudaFree(d_image);
    cudaFree(d_gray_image);
    cudaFree(h_image);
    cudaFree(h_gray_image);
    cudaFree(h_blurred_image);

    // read image
    // host and device vars
    return 0;

}