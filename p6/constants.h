// constants.h
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

// True for all CUDA architectures so far. This is the maximum number of threads per block.
const int MAX_BLOCK_SIZE = 1024;
const std::string input_filename = "out/x_y.csv";
const std::string output_filename = "out/x_y_scan.csv";

// This struct is used to store the x and y values of a point in a 2D plane.
struct X_Y {
    float x;
    float y;
    int original_index;
};

#endif // CONSTANTS_H