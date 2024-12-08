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
    
    // Determine if the program is running a test.
    bool is_test = (argc > 1 && std::string(argv[1]) == "test");

    if (!is_test)
    {
        cout << "Running in normal mode" << endl;
        // Reads the input file named x_y.csv, which contains two fields per line: x value, y value.
        std::vector<X_Y> data = read_csv(input_filename);

        // Convert data to X_Y array for CUDA.
        int n = data.size();
        X_Y *d_data;
        cudaMalloc(&d_data, n * sizeof(X_Y));
        cudaMemcpy(d_data, data.data(), n * sizeof(X_Y), cudaMemcpyHostToDevice);

        // Sorts this sequence of (x,y) pairs by their x-values, using a parallel bitonic sort in CUDA.
        bitonic_sort(d_data, n);

        // Copy sorted data back to host
        cudaMemcpy(data.data(), d_data, n * sizeof(X_Y), cudaMemcpyDeviceToHost);

        // Scans the y-values in the sorted sequence using a parallel prefix scan in CUDA.
        std::vector<float> y_values(n);
        for (int i = 0; i < n; ++i) {
            y_values[i] = data[i].y;
        }
        float *d_y_values;
        cudaMalloc(&d_y_values, n * sizeof(float));
        cudaMemcpy(d_y_values, y_values.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        reduce_scan_1block(d_y_values, n);
        cudaMemcpy(y_values.data(), d_y_values, n * sizeof(float), cudaMemcpyDeviceToHost);

        // Writes the sorted sequence to a new file named x_y_scan.csv with four fields per line,
        // in the following order: x value, y value, cumulative y value, original row number.
        write_csv(output_filename, data, y_values);

        cudaFree(d_data);
        cudaFree(d_y_values);
    }
    else {
        p6_test();
    }

    
    return 0;
}