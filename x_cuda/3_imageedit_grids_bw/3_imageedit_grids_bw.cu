#include <cuda_runtime.h>
#include <iostream>
#include <simple_funcs.h>
#include <image_data_gen.h>


__global__ void grayscale_kernel(unsigned char* image, unsigned char* gray_image, int width, int height){
    /*
    This kernel converts an image to grayscale.
    Each thread processes one pixel of the image (assumed to have rgb channels).
    */
    int x_coord = blockIdx.x * blockDim.x + threadIdx.x; //=0
    int y_coord = blockIdx.y * blockDim.y + threadIdx.y; //=0

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

__global__ void blur_kernel(unsigned char* image,
    unsigned char* blurred_image, int width, int height,
    int filter_size){
    int x_coord = blockIdx.x*blockDim.x + threadIdx.x; 
    int y_coord = blockIdx.y*blockDim.y + threadIdx.y;
    // printf("Blur kernel: thread %d, block %d, pixel (%d, %d)\n", 
    //     threadIdx.x, blockIdx.x, x_coord, y_coord);
    if (x_coord<width && y_coord<height){
        //printf("Blur kernel: thread %d, block %d, pixel (%d, %d)\n",
        // threadIdx.x, blockIdx.x, x_coord, y_coord);
        int r_sum = 0;
        int g_sum = 0;
        int b_sum = 0;
        int count_pxs = 0;
        for (int i=-filter_size; i<=filter_size; i++){ 
            for (int j=-filter_size; j<=filter_size; j++){ 
                
                int x = x_coord + i;
                int y = y_coord + j;
                
                if ((x>=0 && x<width) && (y>=0 && y<height)){
                //printf("Blur kernel: thread %d, block %d, pixel (%d, %d)\n",
                //threadIdx.x, blockIdx.x, x, y);
                int pixel_index = (y*width + x)*3; 
                r_sum += image[pixel_index];
                g_sum += image[pixel_index + 1];
                b_sum += image[pixel_index + 2];
                //printf("rsum: %d, g_sum: %d, b_sum: %d\n", r_sum, g_sum, b_sum);
                count_pxs++;
                // rgb values
                //printf("Blur kernel: thread %d, block %d, pixel (%d, %d), rgb values (%d, %d, %d)\n",
                //threadIdx.x, blockIdx.x, x, y, image[pixel_index], image[pixel_index + 1], image[pixel_index + 2]);
                }
            }
        }
    int pixel_index = (y_coord*width + x_coord)*3;
    unsigned char r_avg = r_sum/count_pxs;
    unsigned char g_avg = g_sum/count_pxs;
    unsigned char b_avg = b_sum/count_pxs; 
    blurred_image[pixel_index] = r_avg;
    blurred_image[pixel_index + 1] = g_avg;
    blurred_image[pixel_index + 2] = b_avg;
    // printf("Total pixels: %d, pixel index: %d, blurred pixel (%d, %d), rgb values blur(%d, %d, %d) rgb values original (%d, %d, %d)\n", 
    //     count_pxs, pixel_index, x_coord, y_coord, r_avg, g_avg, b_avg, 
    //     image[pixel_index], image[pixel_index + 1], image[pixel_index + 2]);
    }
}

__global__ void blur_grayscale_fused_kernel(unsigned char* image,
    unsigned char* blurred_image, int width, int height,
    int filter_size){
    int x_coord = blockIdx.x*blockDim.x + threadIdx.x; 
    int y_coord = blockIdx.y*blockDim.y + threadIdx.y;
    // printf("Blur kernel: thread %d, block %d, pixel (%d, %d)\n", 
    //     threadIdx.x, blockIdx.x, x_coord, y_coord);
    // convert to grayscale
    if (x_coord < width && y_coord < height) {
        image[y_coord*width + x_coord] = (image[y_coord*width + x_coord] + image[y_coord*width + x_coord + 1] + image[y_coord*width + x_coord + 2]) / 3;
    }
    
    if (x_coord<width && y_coord<height){
        //printf("Blur kernel: thread %d, block %d, pixel (%d, %d)\n",
        // threadIdx.x, blockIdx.x, x_coord, y_coord);
        int r_sum = 0;
        int g_sum = 0;
        int b_sum = 0;
        int count_pxs = 0;
        for (int i=-filter_size; i<=filter_size; i++){ 
            for (int j=-filter_size; j<=filter_size; j++){ 
                
                int x = x_coord + i;
                int y = y_coord + j;
                
                if ((x>=0 && x<width) && (y>=0 && y<height)){
                //printf("Blur kernel: thread %d, block %d, pixel (%d, %d)\n",
                //threadIdx.x, blockIdx.x, x, y);
                int pixel_index = (y*width + x)*3; 
                r_sum += image[pixel_index];
                g_sum += image[pixel_index + 1];
                b_sum += image[pixel_index + 2];
                //printf("rsum: %d, g_sum: %d, b_sum: %d\n", r_sum, g_sum, b_sum);
                count_pxs++;
                // rgb values
                //printf("Blur kernel: thread %d, block %d, pixel (%d, %d), rgb values (%d, %d, %d)\n",
                //threadIdx.x, blockIdx.x, x, y, image[pixel_index], image[pixel_index + 1], image[pixel_index + 2]);
                }
            }
        }
    int pixel_index = (y_coord*width + x_coord)*3;
    unsigned char r_avg = r_sum/count_pxs;
    unsigned char g_avg = g_sum/count_pxs;
    unsigned char b_avg = b_sum/count_pxs;
    blurred_image[pixel_index] = r_avg;
    blurred_image[pixel_index + 1] = g_avg;
    blurred_image[pixel_index + 2] = b_avg;
    }
}

int main() {
    // print gpu specs
    print_gpuspecs();
    // Initialize the ImageHandler with file paths
    ImageHandler handler;
    // Access raw image data
    int blur_filter_size = 18;
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
    cudaMalloc((void **)&d_blurred_image, width * height * 3 * sizeof(unsigned char));

    cudaMemcpy(d_image, h_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // for (int i = 0; i < 100; i++) {
    //     std::cout << "d_image[" << i << "] = " << (int)h_image[i] << std::endl;
    // }
    // return 0; 

    dim3 grid(ceil(width / 16.0), ceil(height / 16.0), 1);
    dim3 block(16, 16, 1);

    cudaEvent_t start_gray, stop_gray;
    cudaEventCreate(&start_gray);
    cudaEventCreate(&stop_gray);
    cudaEventRecord(start_gray, 0);
    // check contents of d_image
   //for (int i = 0; i < 10000000; i++) {
    //printf("[%d] \n", i);
    grayscale_kernel<<<grid, block>>>(d_image, d_gray_image, width, height);
    //} 
    cudaEventRecord(stop_gray, 0);
    cudaEventSynchronize(stop_gray);
    float grayscalems = 0;
    cudaEventElapsedTime(&grayscalems, start_gray, stop_gray);
    std::cout << "Grayscale conversion took " << grayscalems << " ms." << std::endl;
    
    cudaMemcpy(h_gray_image, d_gray_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
   
    // sweep for blur only kernel
    cudaEvent_t start_blur, stop_blur;
    float blurms = 0;
    for (int i = 20; i > 3; i--) {
    blur_filter_size = i;
    cudaEventCreate(&start_blur);
    cudaEventCreate(&stop_blur);
    cudaEventRecord(start_blur, 0);
    blur_kernel<<<grid, block>>>(d_gray_image, d_blurred_image, width, height, blur_filter_size);
    cudaEventRecord(stop_blur, 0);
    cudaEventSynchronize(stop_blur);
    cudaEventElapsedTime(&blurms, start_blur, stop_blur);
    std::cout << "Blur kernel time: " << blurms << " ms. " << "Total time (blur + grayscale) : " << blurms + grayscalems << " ms. " << "Filter size: " << blur_filter_size*2+1 << std::endl;
    }
    
    // sweep for fused kernel 
    for (int i = 20; i > 3; i--) {
    blur_filter_size = i;
    cudaEventCreate(&start_blur);
    cudaEventCreate(&stop_blur);
    cudaEventRecord(start_blur, 0);
    blur_grayscale_fused_kernel<<<grid, block>>>(d_image, d_blurred_image, width, height, blur_filter_size);
    cudaEventRecord(stop_blur, 0);
    cudaEventSynchronize(stop_blur);
    cudaEventElapsedTime(&blurms, start_blur, stop_blur);
    std::cout << "Fused kernel time: " << blurms << " ms. " << "Filter size: " << blur_filter_size*2+1 << std::endl;
    }

    cudaMemcpy(h_blurred_image, d_blurred_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    handler.saveImage(h_blurred_image, "blurred_image.png");
    handler.saveImage(h_gray_image, "gray_image.png");
    handler.saveImage(h_image, "color_image.png");

    cudaFree(d_image);
    cudaFree(d_gray_image);
    cudaFree(d_blurred_image);
    cudaFree(h_image);
    cudaFree(h_gray_image);
    cudaFree(h_blurred_image);

    return 0;

}