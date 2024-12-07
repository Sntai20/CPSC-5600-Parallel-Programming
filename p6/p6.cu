
#include <iostream>
#include <random>
#include <string>

#include "constants.h"
#include "bitonic_naive.h"
#include "reduce_scan_1block.h"

using namespace std;

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <utility>

std::vector<std::pair<float, float>> read_csv(const std::string& filename) {
    std::vector<std::pair<float, float>> data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string x_str, y_str;
        if (std::getline(ss, x_str, ',') && std::getline(ss, y_str, ',')) {
            float x = std::stof(x_str);
            float y = std::stof(y_str);
            data.emplace_back(x, y);
        }
    }

    file.close();
    return data;
}

int main(int argc, char *argv[]) {
    std::cout << "Hello, World!" << std::endl;

    // Determine if the program is running a test
    bool is_test = (argc > 1 && std::string(argv[1]) == "test");
    std::string input_filename = is_test ? "x_y_16.csv" : "x_y.csv";

    // Reads a large csv file named x_y.csv, containing up to one million (x,y) pairs of floating-point numbers.
    // The file has two fields per line, separated by a comma: the x value and the y value.
    /* Example:
    x,y
    12.278249,1.152063
    13.043502,0.114110
    */
   std::vector<std::pair<float, float>> data = read_csv(input_filename);

    // Sorts this sequence o (x,y) pairs by their x-values, using a parallel bitonic sort in CUDA.
    bitonic_naive_sort();

    // Scans the y-values in the sorted sequence using a parallel prefix scan in CUDA.
    reduce_scan_1block();
    
    // Writes the sorted sequence to a new file named x_y_scan.csv with four fields per line, in the following order: x value, y value, cumulative y value, original row number.
    
    return 0;
}