
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
void ImageHandler::saveImage(unsigned char* imageData, const std::string& saveFileName) {
    if (image.empty()) {
        throw std::runtime_error("Cannot save an empty image.");
    }
    if (imageData != nullptr) {
        // If imageData is provided, use it to create a new cv::Mat
        cv::Mat tempImage(image.rows, image.cols, image.type(), imageData);
        if (cv::imwrite(saveFileName, tempImage)) {
            std::cout << "Saved image: " << saveFileName << std::endl;
        } else {
            throw std::runtime_error("Could not save the image to: " + saveFileName);
        }
    } 
    else if (cv::imwrite(saveFileName, image)) {
        std::cout << "Saved image: " << saveFileName << std::endl;
    } 
    else {
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

int ImageHandler::getHeight() const {
    return image.rows;
}

int ImageHandler::getWidth() const {
    return image.cols;
}


