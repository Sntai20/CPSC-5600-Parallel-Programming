#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <utility>
#include <filesystem>

std::vector<std::pair<float, float>> read_csv(const std::string& filename) {
    std::vector<std::pair<float, float>> data;

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