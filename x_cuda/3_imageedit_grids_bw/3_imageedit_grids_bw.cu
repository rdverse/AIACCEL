#include <cuda_runtime.h>
#include <iostream>
#include <simple_funcs.h>
#include <image_data_gen.h>

#include <map> 


/*

This code is comparing the performance of fused kernels vs separate kernels for grayscale conversion and blurring of an image.

Overall, there is a small fractional improvement in the fused kernel. Some filter shapes give larger improvement .

The fused kernel can be made more efficient but to make an apple to apple comparison, the only change is merging the grayscale and blurring into one kernel.

*/

__global__ void grayscale_kernel(unsigned char* image, unsigned char* gray_image, int width, int height){
    /*
    This kernel converts an image to grayscale.
    Each thread processes one pixel of the image (assumed to have rgb channels).
    */
    int x_coord = blockIdx.x * blockDim.x + threadIdx.x; //=0
    int y_coord = blockIdx.y * blockDim.y + threadIdx.y; //=0

    if (x_coord < width && y_coord < height) {
        int pixel_index = (y_coord*width + x_coord)*3;
        unsigned int r = image[pixel_index];
        unsigned int g = image[pixel_index + 1];
        unsigned int b = image[pixel_index + 2];
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
        int pixel_index = (y_coord*width + x_coord)*3;
        unsigned int r = image[pixel_index];
        unsigned int g = image[pixel_index + 1];
        unsigned int b = image[pixel_index + 2];
        unsigned char gray = (r + g + b) / 3;
        image[pixel_index] = gray;
        image[pixel_index + 1] = gray;
        image[pixel_index + 2] = gray;
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

struct KernelTiming {
    float grayscale_time;
    float blur_time;
    float blur_grayscale_time;
    float fused_time;
    float filter_size;
    float deltatime;
    float percent_improvement;
};


void print_results(std::map<int, KernelTiming> results) {
    // Print the table header
    std::cout << std::left
              << std::setw(15) << "Filter Size"
              << std::setw(20) << "Grayscale Time"
              << std::setw(20) << "Blur Time"
              << std::setw(25) << "Blur + Grayscale Time"
              << std::setw(15) << "Fused Time"
              << std::setw(20) << "nonfuse-fuse time"
              << std::setw(25) << "(nonfuse-fuse)/non-fuse%"
              << std::endl;

    // Print a separator line
    std::cout << std::string(130, '-') << std::endl;

    // Print each result in the map
    for (const auto& result : results) {
        const auto& timing = result.second;
        std::cout << std::left
                  << std::setw(15) << timing.filter_size 
                  << std::setw(20) << std::fixed << std::setprecision(6) << timing.grayscale_time
                  << std::setw(20) << std::fixed << std::setprecision(6) << timing.blur_time
                  << std::setw(25) << std::fixed << std::setprecision(6) << timing.blur_grayscale_time
                  << std::setw(15) << std::fixed << std::setprecision(6) << timing.fused_time
                  << std::setw(20) << std::fixed << std::setprecision(6) << timing.deltatime
                  << std::setw(6) << std::fixed << std::setprecision(2) << timing.percent_improvement
                  << "%"
                  << std::endl;
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

    std::map<int, KernelTiming> results;

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
    float grayscalems = 0;
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
    cudaEventElapsedTime(&grayscalems, start_gray, stop_gray);
    //std::cout << "Grayscale conversion took " << grayscalems << " ms." << std::endl;
    
    cudaMemcpy(h_gray_image, d_gray_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
   
    // sweep for blur only kernel
    cudaEvent_t start_blur, stop_blur;
    float blurms = 0;
    for (int i = 40; i >= 0; i--) {
    blur_filter_size = i;
    cudaEventCreate(&start_blur);
    cudaEventCreate(&stop_blur);
    cudaEventRecord(start_blur, 0);
    blur_kernel<<<grid, block>>>(d_gray_image, d_blurred_image, width, height, blur_filter_size);
    cudaEventRecord(stop_blur, 0);
    cudaEventSynchronize(stop_blur);
    cudaEventElapsedTime(&blurms, start_blur, stop_blur);
    //std::cout << "Blur kernel time: " << blurms << " ms. " << "Total time (blur + grayscale) : " << blurms + grayscalems << " ms. " << "Filter size: " << blur_filter_size*2+1 << std::endl;
    results[blur_filter_size] = {grayscalems, 
                                blurms,
                                grayscalems + blurms, 
                                0, // set this in fused kernel
                                (float)(blur_filter_size*2+1), 
                                0,
                                0}; // set this in fused kernel
    }

    // sweep for blur kernel 
    // cache locality seems to favoring fused kernel , so rerun the blur kernel
    for (int i = 40; i >= 0; i--) {
    blurms=0;
    blur_filter_size = i;
    cudaEventCreate(&start_blur);
    cudaEventCreate(&stop_blur);
    cudaEventRecord(start_blur, 0);
    blur_grayscale_fused_kernel<<<grid, block>>>(d_image, d_blurred_image, width, height, blur_filter_size);
    cudaEventRecord(stop_blur, 0);
    cudaEventSynchronize(stop_blur);
    cudaEventElapsedTime(&blurms, start_blur, stop_blur);
    //std::cout << "Fused kernel time: " << blurms << " ms. " << "Filter size: " << blur_filter_size*2+1 << std::endl;
    
    results[blur_filter_size].fused_time = blurms;
    results[blur_filter_size].deltatime = results[blur_filter_size].fused_time - results[blur_filter_size].blur_grayscale_time;
    results[blur_filter_size].percent_improvement = results[blur_filter_size].deltatime / results[blur_filter_size].blur_grayscale_time * 100.0f;
    if (i==10){
    cudaMemcpy(h_blurred_image, d_blurred_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    handler.saveImage(h_blurred_image, "blurred_image.png");
    }
    }


    for (int i = 40; i >= 0; i--) {
    blurms=0;
    blur_filter_size = i;
    cudaEventCreate(&start_blur);
    cudaEventCreate(&stop_blur);
    cudaEventRecord(start_blur, 0);
    blur_kernel<<<grid, block>>>(d_gray_image, d_blurred_image, width, height, blur_filter_size);
    cudaEventRecord(stop_blur, 0);
    cudaEventSynchronize(stop_blur);
    cudaEventElapsedTime(&blurms, start_blur, stop_blur);
    //std::cout << "Blur kernel time: " << blurms << " ms. " << "Total time (blur + grayscale) : " << blurms + grayscalems << " ms. " << "Filter size: " << blur_filter_size*2+1 << std::endl;
    results[blur_filter_size].blur_time = blurms;
    results[blur_filter_size].blur_grayscale_time = grayscalems + blurms;
    results[blur_filter_size].deltatime = blurms - results[blur_filter_size].fused_time; 
    results[blur_filter_size].percent_improvement = results[blur_filter_size].deltatime / results[blur_filter_size].blur_grayscale_time * 100.0f;
    }



    print_results(results);
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