#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <simple_funcs.h>
#include <image_data_gen.h>

int main() {
    // print gpu specs
    print_gpuspecs();
    // Initialize the ImageHandler with file paths
    ImageHandler handler;
    // Access raw image data
    unsigned char* imageData = handler.getImageData();
    
    // read image
    // host and device vars
    return 0;

}