#ifndef IMAGE_DATA_GEN_H
#define IMAGE_DATA_GEN_H

#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept> // For exceptions

// Class to handle image loading, processing, and saving using OpenCV
class ImageHandler {
private:
    std::string loadFileName;  // Path to load the image
    std::string saveFileName;  // Path to save the image
    cv::Mat image;             // OpenCV matrix to store the image
    unsigned char* imageData;  // Pointer to raw image data

public:
    // Constructor to initialize file paths and load the image
    ImageHandler(const std::string& loadFileName = "../assets/lokiandthor.png", 
                 const std::string& saveFileName = "saved_image.png");

    // Destructor to clean up resources
    ~ImageHandler();

    // Save the image stored in the class
    void saveImage();

    // Accessor for raw image data
    unsigned char* getImageData() const;

    // Accessor for the cv::Mat image
    cv::Mat getImage() const;

    // Delete copy constructor and assignment operator to prevent copying
    ImageHandler(const ImageHandler&) = delete;
    ImageHandler& operator=(const ImageHandler&) = delete;
};

#endif