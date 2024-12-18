#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "constants.h"
#include "file_utils.h"
#include "bitonic_naive.h"
#include "reduce_scan_1block.h"

using namespace std;

const string input_filename_test = "out/test/x_y_16.csv";
const string output_filename_test = "out/test/x_y_scan_16.csv";

bool file_utils_read_csv_test(string input_filename) {
    cout << "Running file_utils_read_csv_test" << endl;
    
    // Call the read_csv function
    vector<X_Y> data = read_csv(input_filename);

    // Check if the data size is correct
    if (data.size() != 16) {
        cout << "Test failed: file_utils_read_csv_test failed ---- Expected 16 rows, got. " << data.size() << endl;
        return false;
    }

    // Check if the data values are correct
    if (data[0].x != 12.278249f || data[0].y != 1.152063f || data[0].original_index != 0 ||
        data[1].x != 13.043502f || data[1].y != 0.114110f || data[1].original_index != 1 ||
        data[2].x != 12.297193f || data[2].y != 1.154309f || data[2].original_index != 2) {
        cout << "Test failed: file_utils_read_csv_test failed ---- Data values are incorrect." << endl;
        return false;
    }

    cout << "Test passed: file_utils_read_csv_test works correctly." << endl;
    
    return true;
}

bool bitonic_naive_sort_test() {
    cout << "Running bitonic_naive_sort_test" << endl;
    
    // Initialize a sample array of X_Y structures
    vector<X_Y> data = {
        {3.0f, 1.0f, 0},
        {1.0f, 2.0f, 1},
        {4.0f, 3.0f, 2},
        {2.0f, 4.0f, 3}
    };

    int n = data.size();
    X_Y *d_data;
    cudaMalloc(&d_data, n * sizeof(X_Y));
    cudaMemcpy(d_data, data.data(), n * sizeof(X_Y), cudaMemcpyHostToDevice);

    // Call the bitonic_sort function
    bitonic_sort(d_data, n);

    // Copy sorted data back to host
    cudaMemcpy(data.data(), d_data, n * sizeof(X_Y), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Check if the data is sorted correctly
    bool sorted = true;
    for (int i = 1; i < n; ++i) {
        if (data[i - 1].x > data[i].x) {
            sorted = false;
            break;
        }
    }

    if (sorted) {
        cout << "Test passed: bitonic_sort function works correctly." << endl;
    } else {
        cout << "Test failed: Data is not sorted correctly." << endl;
    }
    
    return true;
}

bool reduce_scan_1block_test() {
    cout << "Running reduce_scan_1block_test" << endl;
    
    // Initialize a sample array of float values
    vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    vector<float> expected_prefix_sums = {1.0f, 3.0f, 6.0f, 10.0f};

    int n = data.size();
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, data.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Call the reduce_scan_1block function
    reduce_scan_1block(d_data, n);

    // Copy the result back to host
    cudaMemcpy(data.data(), d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Check if the result is correct
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (data[i] != expected_prefix_sums[i]) {
            correct = false;
            break;
        }
    }

    if (correct) {
        cout << "Test passed: reduce_scan_1block function works correctly." << endl;
    } else {
        cout << "Test failed: Prefix sums are incorrect." << endl;
    }
    
    return true;
}

bool file_utils_write_csv_test(string output_filename) {
    cout << "Running file_utils_write_csv_test" << endl;

    // Initialize a sample array of X_Y structures and cumulative y-values
    vector<X_Y> data = {
        {1.0f, 2.0f, 0},
        {3.0f, 4.0f, 1},
        {5.0f, 6.0f, 2}
    };
    vector<float> cumulative_y = {2.0f, 6.0f, 12.0f};

    // Call the write_csv function
    write_csv(output_filename, data, cumulative_y);

    // Read the file back and check its contents
    ifstream file(output_filename);
    if (!file.is_open()) {
        cout << "Test failed: Could not open file. " << output_filename << endl;
        return false;
    }

    string line;
    getline(file, line); // Skip the header line

    bool correct = true;
    for (size_t i = 0; i < data.size(); ++i) {
        if (!getline(file, line)) {
            correct = false;
            break;
        }
        stringstream ss(line);
        string x_str, y_str, cumulative_y_str, original_index_str;
        if (getline(ss, x_str, ',') && getline(ss, y_str, ',') &&
            getline(ss, cumulative_y_str, ',') && getline(ss, original_index_str, ',')) {
            float x = stof(x_str);
            float y = stof(y_str);
            float cum_y = stof(cumulative_y_str);
            int original_index = stoi(original_index_str);
            if (x != data[i].x || y != data[i].y || cum_y != cumulative_y[i] || original_index != data[i].original_index) {
                correct = false;
                break;
            }
        } else {
            correct = false;
            break;
        }
    }

    file.close();

    if (correct) {
        cout << "Test passed: write_csv function works correctly." << endl;
    } else {
        cout << "Test failed: File contents are incorrect." << endl;
    }
    
    return true;
}

// Run all tests
bool p6_test() {
    cout << "\n" << endl;
    cout << "Running in test mode" << endl;
    cout << "Current working directory: " << get_current_directory() << endl;

    string input_filename = input_filename_test;
    cout << "Reading from: " << input_filename << endl;

    string output_filename = output_filename_test;
    cout << "Writing to: " << output_filename << "\n\n" << endl;
    
    cout << "\n" << endl;
    cout << "Running tests" << endl;
    if (!file_utils_read_csv_test(input_filename_test)) {
        return false;
    }

    if (!bitonic_naive_sort_test()) {
        return false;
    }

    if (!reduce_scan_1block_test()) {
        return false;
    }

    if (!file_utils_write_csv_test(output_filename_test)) {
        return false;
    }

    cout << "\n" << endl;

    return true;
}