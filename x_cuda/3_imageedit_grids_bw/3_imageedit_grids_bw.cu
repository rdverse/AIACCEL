#include <cuda_runtime.h>
#include <iostream>
#include <simple_funcs.h>
#include <image_data_gen.h>
#include <map> 
#include <unistd.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cub/cub.cuh>

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
    int filter_size, bool grayscale = true){
    int x_coord = blockIdx.x*blockDim.x + threadIdx.x; 
    int y_coord = blockIdx.y*blockDim.y + threadIdx.y;
    // printf("Blur kernel: thread %d, block %d, pixel (%d, %d)\n", 
    //     threadIdx.x, blockIdx.x, x_coord, y_coord);
    // convert to grayscale 
    // use below code for grayscle conversion if only comparing 
    if (grayscale && x_coord < width && y_coord < height) {
        int pixel_index = (y_coord*width + x_coord)*3;
        unsigned int r = image[pixel_index];
        unsigned int g = image[pixel_index + 1];
        unsigned int b = image[pixel_index + 2];
        unsigned char gray = (r + g + b) / 3;
        image[pixel_index] = gray;
        image[pixel_index + 1] = gray;
        image[pixel_index + 2] = gray;
        // synchronize threads to ensure all threads have completed grayscale conversion
        __syncthreads();
    }

    // blur the image 
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
                // keeping this consistent with the blur kernel, at this point r=g=b=gray
    
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
    unsigned char gray_blur = (r_sum + g_sum + b_sum) / (3*count_pxs);
    unsigned char r_avg = r_sum/count_pxs;
    unsigned char g_avg = g_sum/count_pxs;
    unsigned char b_avg = b_sum/count_pxs;
    blurred_image[pixel_index] = r_avg;
    blurred_image[pixel_index + 1] = g_avg;
    blurred_image[pixel_index + 2] = b_avg;
    }
}

struct KernelTiming {
    int filter_size;
    float non_fused_time;
    float fused_time;
    float deltatime;
    float percent_improvement;
};


void print_results(std::map<int, KernelTiming> results) {
    // Print the table header
    std::cout << std::left
              << std::setw(15) << "Filter Size"
              << std::setw(25) << "Non-Fused Time"
              << std::setw(15) << "Fused Time"
              << std::setw(20) << "nonfused-fused time"
              << std::setw(25) << "(nonfused-fused)/non-fused%"
              << std::endl;

    // Print a separator line
    std::cout << std::string(130, '-') << std::endl;

    // Print each result in the map
    for (const auto& result : results) {
        const auto& timing = result.second;
        std::cout << std::left
                  << std::setw(15) << timing.filter_size 
                  << std::setw(25) << std::fixed << std::setprecision(6) << timing.non_fused_time
                  << std::setw(15) << std::fixed << std::setprecision(6) << timing.fused_time
                  << std::setw(20) << std::fixed << std::setprecision(6) << timing.deltatime
                  << std::setw(6) << std::fixed << std::setprecision(2) << timing.percent_improvement
                  << "%"
                  << std::endl;
    }
    
}


float kernel_launcher(const ImageHandler& handler, int filter_size,
                    bool fused = false) {
    
    unsigned char* d_image;
    unsigned char* h_image = handler.getImageData();
    int width = handler.getWidth();
    int height = handler.getHeight();
    
    dim3 grid(ceil(width / 16.0), ceil(height / 16.0), 1);
    dim3 block(16, 16, 1);

    cudaMalloc((void **)&d_image, width * height * 3 * sizeof(unsigned char));
    cudaMemcpy(d_image, h_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    unsigned char* d_blurred_image;
    unsigned char* h_blurred_image= new unsigned char[width * height * 3];
    cudaMalloc((void **)&d_blurred_image, width * height * 3 * sizeof(unsigned char));
    
    unsigned char* d_gray_image;
    unsigned char* h_gray_image;// = new unsigned char[width * height * 3];
    if (!fused){
    h_gray_image = new unsigned char[width * height * 3];
    cudaMalloc((void **)&d_gray_image, width * height * 3 * sizeof(unsigned char));
    }
    cudaEvent_t start, stop;
    float blurms = 0;
    int blur_filter_size = filter_size;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    if (fused) {
        // Launch the fused kernel
        blur_grayscale_fused_kernel<<<grid, block>>>(d_image, d_blurred_image, width, height, filter_size);
    } else {
        // Launch the grayscale kernel
        grayscale_kernel<<<grid, block>>>(d_image,d_gray_image, width, height);
        // Launch the blur kernel
        blur_kernel<<<grid, block>>>(d_gray_image, d_blurred_image, width, height, filter_size);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&blurms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (!fused) {
        cudaFree(d_gray_image);
    }
    cudaFree(d_blurred_image);
    cudaFree(d_image);
    return blurms;
}


int main() {
    // print gpu specs
    print_gpuspecs();
    std::map<int, KernelTiming> results;
    ImageHandler image_handler;
    // Access raw image data
    for(int i=0;i<101;i++){
        float fused_time=0;
        float nonfused_time=0;
        for(int k=0;k<2;k++){
            for(int j=0;j<5;j++){
                
                size_t l2_evict_size = 4 * 1024 * 1024; // 4MB for 2070 super
                thrust::device_vector<unsigned char> l2_evict_vec(l2_evict_size);
                thrust::fill(l2_evict_vec.begin(), l2_evict_vec.end(), 1);
                thrust::transform(l2_evict_vec.begin(), l2_evict_vec.end(),
                                l2_evict_vec.begin(),
                                thrust::placeholders::_1 + 1);
                //cudaDeviceSynchronize();
                if(k==0){
                    sleep(0.5); // Sleep for 1 second
                    float time = kernel_launcher(image_handler,i, false);
                    //std::cout << time<< std::endl;
                    if (j>1) {nonfused_time+=time;}
                }else{
                    sleep(0.5); // Sleep for 1 second
                    float time = kernel_launcher(image_handler,i, true);
                    //std::cout << time<< std::endl;
                    if (j>1) {fused_time+=time;}
                    // add sleep 1 seconds
                }
            }
            sleep(2); // Sleep for 1 second
        }
        fused_time=fused_time/3;
        nonfused_time=nonfused_time/3;
        std::cout<<"Filter size :" << i << " fused time :"<<fused_time << " nonfused time : "<< nonfused_time << std::endl;
        float deltatime = nonfused_time-fused_time;
        float percent_improvement = (deltatime/nonfused_time)*100;
        results[i]={i,nonfused_time,fused_time,deltatime, percent_improvement};
    }

    print_results(results);
    return 0;

}