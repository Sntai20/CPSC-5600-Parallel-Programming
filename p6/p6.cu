
#include <iostream>
#include <string>

using std::string;

// Global Variables

// Constant all CUDA blocks have 1024 threads.
const int MAX_BLOCK_SIZE = 1024;
const string filename = "x_y.csv";
const string output_filename = "x_y_scan.csv";


int main() {
    std::cout << "Hello, World!" << std::endl;

    // Reads a large csv file named x_y.csv, containing up to one million (x,y) pairs of floating-point numbers.


    // Sorts this sequence o (x,y) pairs by their x-values, using a parallel bitonic sort in CUDA.


    // Scans the y-values in the sorted sequence using a parallel prefix scan Links to an external site. in CUDA.
    
    
    // Writes the sorted sequence to a new file named x_y_scan.csv with four fields per line, in the following order: x value, y value, cumulative y value, original row number.
    
    return 0;
}