#include <iostream>
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
    std::vector<X_Y> data = read_csv(input_filename);

    // Check if the data size is correct
    if (data.size() != 16) {
        cout << "Test failed: file_utils_read_csv_test failed ---- Expected 16 rows, got " << data.size() << endl;
        return false;
    }

    // Check if the data values are correct
    // if (data[0].x != 1.0f || data[0].y != 2.0f || data[0].original_index != 0 ||
    //     data[1].x != 3.0f || data[1].y != 4.0f || data[1].original_index != 1 ||
    //     data[2].x != 5.0f || data[2].y != 6.0f || data[2].original_index != 2) {
    //     cout << "----file_utils_read_csv_test failed ---- Data values are incorrect" << endl;
    //     return false;
    // }

    cout << "Test passed: file_utils_read_csv_test works correctly." << endl;
    
    return true;
}

bool bitonic_naive_sort_test() {
    cout << "Running bitonic_naive_sort_test" << endl;
    
    // Initialize a sample array of X_Y structures
    std::vector<X_Y> data = {
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

    // cout << "----bitonic_naive_sort_test passed" << endl;
    
    return true;
}

// bool reduce_scan_1block_test() {
//     cout << "Running reduce_scan_1block_test" << endl;
    
//     reduce_scan_1block();

//     cout << "----reduce_scan_1block_test passed" << endl;
    
//     return true;
// }

// bool file_utils_write_csv_test(string output_filename) {
//     cout << "Running file_utils_write_csv_test" << endl;
    
//     std::vector<std::pair<float, float>> data = read_csv(output_filename);

//     if (data.size() != 16) {
//         cout << "----file_utils_write_csv_test failed ---- Expected 16 rows, got " << data.size() << endl;
//         return false;
//     }

//     cout << "----file_utils_write_csv_test passed" << endl;
    
//     return true;
// }

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

    // if (!reduce_scan_1block_test()) {
    //     return false;
    // }

    // if (!file_utils_write_csv_test(output_filename_test)) {
    //     return false;
    // }

    cout << "\n" << endl;

    return true;
}