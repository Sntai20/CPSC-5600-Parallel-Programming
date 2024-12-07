#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <vector>
#include <string>
#include <utility>

std::vector<std::pair<float, float>> read_csv(const std::string& filename);

#endif // FILE_UTILS_H