
#include <iostream>
#include <random>
#include <string>

#include "constants.h"
#include "bitonic_naive.h"
#include "reduce_scan_1block.h"

using namespace std;

int main() {
    std::cout << "Hello, World!" << std::endl;

    // Reads a large csv file named x_y.csv, containing up to one million (x,y) pairs of floating-point numbers.
    // The file has two fields per line, separated by a comma: the x value and the y value.
    /* Example:
    x,y
    12.278249,1.152063
    13.043502,0.114110
    */

    // Sorts this sequence o (x,y) pairs by their x-values, using a parallel bitonic sort in CUDA.
    bitonic_naive_sort();

    // Scans the y-values in the sorted sequence using a parallel prefix scan in CUDA.
    reduce_scan_1block();
    
    // Writes the sorted sequence to a new file named x_y_scan.csv with four fields per line, in the following order: x value, y value, cumulative y value, original row number.
    
    return 0;
}