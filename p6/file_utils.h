#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <vector>
#include <string>
#include <utility>

std::vector<X_Y> read_csv(const std::string& filename);
void write_csv(const std::string& filename, const std::vector<X_Y>& data, const std::vector<float>& cumulative_y);
std::string get_current_directory();

#endif // FILE_UTILS_H