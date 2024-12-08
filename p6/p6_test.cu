#include <iostream>
#include "constants.h"
#include "file_utils.h"
#include "bitonic_naive.h"
#include "reduce_scan_1block.h"

using namespace std;

const string input_filename_test = "out/test/x_y_16.csv";
const string output_filename_test = "out/test/x_y_scan_16.csv";

// bool file_utils_read_csv_test(string input_filename) {
//     cout << "Running file_utils_read_csv_test" << endl;
    
//     std::vector<std::pair<float, float>> data = read_csv(input_filename);

//     if (data.size() != 16) {
//         cout << "----file_utils_read_csv_test failed ---- Expected 16 rows, got " << data.size() << endl;
//         return false;
//     }

//     cout << "----file_utils_read_csv_test passed" << endl;
    
//     return true;
// }

// bool bitonic_naive_sort_test() {
//     cout << "Running bitonic_naive_sort_test" << endl;
    
//     bitonic_naive_sort(256);

//     cout << "----bitonic_naive_sort_test passed" << endl;
    
//     return true;
// }

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
    // if (!file_utils_read_csv_test(input_filename_test)) {
    //     return false;
    // }

    // if (!bitonic_naive_sort_test()) {
    //     return false;
    // }

    // if (!reduce_scan_1block_test()) {
    //     return false;
    // }

    // if (!file_utils_write_csv_test(output_filename_test)) {
    //     return false;
    // }

    cout << "\n" << endl;

    return true;
}