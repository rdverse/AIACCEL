#include <stdio.h>
#include <iostream>

#ifndef CONV_FILTERS_H
#define CONV_FILTERS_H

constexpr int NUM_FILTERS = 9;
constexpr int FILTER_SIZE = 3;
constexpr int R=1;
constexpr int TILE_DIM = 32; 
constexpr int OUT_TILE_DIM = TILE_DIM-2*R;
constexpr int CHANNELS = 3;

// 9 filters: identity, edge, sharpen, box blur, gaussian blur, emboss, outline, left sobel, right sobel
constexpr const int FILTERS[NUM_FILTERS][FILTER_SIZE][FILTER_SIZE] = {
    // Identity
    { {0, 0, 0}, {0, 1, 0}, {0, 0, 0} },
    // Edge detection
    { {1, 0, -1}, {0, 0, 0}, {-1, 0, 1} },
    // Sharpen
    { {0, -1, 0}, {-1, 5, -1}, {0, -1, 0} },
    // Box blur
    { {1, 1, 1}, {1, 1, 1}, {1, 1, 1} },
    // Gaussian blur
    { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} },
    // Emboss
    { {-2, -1, 0}, {-1, 1, 1}, {0, 1, 2} },
    // Outline
    { {-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1} },
    // Left Sobel
    { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} },
    // Right Sobel
    { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} }
};


// inli ne is imp here - one def rule
inline void print_filters() {
    for (int f = 0; f < NUM_FILTERS; ++f) {
        std::cout << "--- Filter " << f << " ---\n";
        for (int i = 0; i < FILTER_SIZE; ++i) {
            for (int j = 0; j < FILTER_SIZE; ++j) {
                std::cout << FILTERS[f][i][j] << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}


#endif // CONV_FILTERS_H