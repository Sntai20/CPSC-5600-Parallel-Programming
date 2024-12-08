// constants.h
#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

// True for all CUDA architectures so far. This is the maximum number of threads per block.
const int MAX_BLOCK_SIZE = 1024;
const std::string input_filename = "out/x_y.csv";
const std::string output_filename = "out/x_y_scan.csv";

#endif // CONSTANTS_H