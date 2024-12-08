#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <utility>
#include <filesystem>

#include "file_utils.h"
#include "constants.h"

// Read a CSV file with two columns of float values, two fields per line,
// separated by a comma: the x value and the y value.
/* Example:
x,y
12.278249,1.152063
13.043502,0.114110
*/
std::vector<X_Y> read_csv(const std::string& filename) {
    std::vector<X_Y> data;

    // Check if the file exists
    if (!std::filesystem::exists(filename)) {
        std::cerr << "File does not exist: " << filename << std::endl;
        return data;
    }

    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    // Skip the header line
    std::getline(file, line);

    int index = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string x_str, y_str;
        if (std::getline(ss, x_str, ',') && std::getline(ss, y_str, ',')) {
            float x = std::stof(x_str);
            float y = std::stof(y_str);
            data.push_back({x, y, index++});
        }
    }

    file.close();
    return data;
}

// Writes the sorted sequence to a new file named x_y_scan.csv with four fields per line,
// in the following order: x value, y value, cumulative y value, original row number.
void write_csv(
    const std::string& filename,
    const std::vector<X_Y>& data,
    const std::vector<float>& cumulative_y) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write the header
    file << "x,y,cumulative_y,original_index\n";

    // Write the data
    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i].x << "," << data[i].y << "," << cumulative_y[i] << "," << data[i].original_index << "\n";
    }

    file.close();
}

// Get the current working directory.
std::string get_current_directory() {
    return std::filesystem::current_path().string();
}