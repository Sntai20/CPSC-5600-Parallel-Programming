#include <iostream>
#include <random>
#include <string>

#include "constants.h"
#include "file_utils.h"
#include "bitonic_naive.h"
#include "reduce_scan_1block.h"
#include "p6_test.h"

using namespace std;

int main(int argc, char *argv[]) {
    
    // Determine if the program is running a test
    bool is_test = (argc > 1 && std::string(argv[1]) == "test");

    if (!is_test)
    {
        cout << "Running in normal mode" << endl;
        // Reads the input file named x_y.csv, which contains two fields per line: x value, y value.
        std::vector<std::pair<float, float>> data = read_csv(input_filename);

        // Sorts this sequence o (x,y) pairs by their x-values, using a parallel bitonic sort in CUDA.
        bitonic_naive_sort();

        // Scans the y-values in the sorted sequence using a parallel prefix scan in CUDA.
        reduce_scan_1block();
        
        // Writes the sorted sequence to a new file named x_y_scan.csv with four fields per line, in the following order: x value, y value, cumulative y value, original row number.    
    }
    else {
        p6_test();
    }

    
    return 0;
}