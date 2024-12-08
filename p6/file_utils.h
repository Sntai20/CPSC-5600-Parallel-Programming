#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <vector>
#include <string>
#include <utility>

std::vector<std::pair<float, float>> read_csv(const std::string& filename);
void write_csv(const std::string& filename, const std::vector<std::pair<float, float>>& data, const std::vector<float>& cumulative_y, const std::vector<int>& original_indices);
std::string get_current_directory();

#endif // FILE_UTILS_H