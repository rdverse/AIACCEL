
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp> // Include the main OpenCV header


#include "image_data_gen.h"
#include <iostream>
#include <opencv2/opencv.hpp>

// Constructor
ImageHandler::ImageHandler(const std::string& loadFileName, const std::string& saveFileName)
    : loadFileName(loadFileName), saveFileName(saveFileName), imageData(nullptr) {
    // Load the image in the constructor
    image = cv::imread(loadFileName, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        throw std::runtime_error("Could not open or find the image: " + loadFileName);
    }

    // Store the raw data pointer
    imageData = image.data;

    // Print image details
    std::cout << "Loaded image: " << loadFileName << std::endl;
    std::cout << "  Width: " << image.cols << std::endl;
    std::cout << "  Height: " << image.rows << std::endl;
    std::cout << "  Channels: " << image.channels() << std::endl;
}

// Destructor
ImageHandler::~ImageHandler() {
    // No need to delete imageData because cv::Mat manages its memory
}

// Save the image
void ImageHandler::saveImage() {
    if (image.empty()) {
        throw std::runtime_error("Cannot save an empty image.");
    }
    if (cv::imwrite(saveFileName, image)) {
        std::cout << "Saved image: " << saveFileName << std::endl;
    } else {
        throw std::runtime_error("Could not save the image to: " + saveFileName);
    }
}

// Accessor for raw image data
unsigned char* ImageHandler::getImageData() const {
    return imageData;
}

// Accessor for the cv::Mat image
cv::Mat ImageHandler::getImage() const {
    return image;
}



// class ImageHandler {
// private:
//     std::string loadFileName;
//     std::string saveFileName;
//     cv::Mat image;               
    
//     // Pointer to raw image data
//     // stored in row-major order
//     // dimensions: height x width x channels
//     unsigned char* imageData;     

// public:
//     // Constructor to initialize file paths and load the image
//     ImageHandler(const std::string& loadFileName = "../assets/lokiandthor.png", 
//                  const std::string& saveFileName = "saved_image.png")
//         : loadFileName(loadFileName), saveFileName(saveFileName), imageData(nullptr) {
//         // Load the image in the constructor
//         image = cv::imread(loadFileName, cv::IMREAD_UNCHANGED);
//         if (image.empty()) {
//             throw std::runtime_error("Could not open or find the image: " + loadFileName);
//         }

//         // Store the raw data pointer
//         imageData = image.data;

//         // Print image details
//         std::cout << "Loaded image: " << loadFileName << std::endl;
//         std::cout << "  Width: " << image.cols << std::endl;
//         std::cout << "  Height: " << image.rows << std::endl;
//         std::cout << "  Channels: " << image.channels() << std::endl;
//     }

//     // Destructor to clean up resources
//     ~ImageHandler() {
//         // No need to delete imageData because cv::Mat manages its memory
//     }

//     // Save the image stored in the class
//     void saveImage() {
//         if (image.empty()) {
//             throw std::runtime_error("Cannot save an empty image.");
//         }
//         if (cv::imwrite(saveFileName, image)) {
//             std::cout << "Saved image: " << saveFileName << std::endl;
//         } else {
//             throw std::runtime_error("Could not save the image to: " + saveFileName);
//         }
//     }

//     // Accessor for raw image data
//     unsigned char* getImageData() const {
//         return imageData;
//     }

//     // Accessor for the cv::Mat image
//     cv::Mat getImage() const {
//         return image;
//     }
// };

// int main() {
//     try {
//         // Initialize the ImageHandler with file paths
//         ImageHandler handler;

//         // Access raw image data
//         unsigned char* imageData = handler.getImageData();
//         std::cout << "Image data pointer: " << static_cast<void*>(imageData) << std::endl;

//         // Print the first few pixel values
//         std::cout << "First few pixel values: ";
//         for (int i = 0; i < 10 && i < handler.getImage().total() * handler.getImage().channels(); ++i) {
//             std::cout << static_cast<int>(imageData[i]) << " ";
//         }
//         std::cout << std::endl;

//         // Save the loaded image
//         handler.saveImage();

//     } catch (const std::runtime_error& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }

//     return 0;

//  }