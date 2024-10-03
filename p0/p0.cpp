#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <omp.h>

std::string const USAGE = "Program requires exactly two arguments, both positive integers.\n";

// desc: Reports incorrect command-line usage and exits with status 1
//  pre: None
// post: In description
void bad_usage() {
    std::cerr << USAGE;
    std::exit(1);
}

// desc: Returns the value and thread counts provided by the supplied
//       command-line arguments.
//  pre: There should be exactly two arguments, both positive integers.
//       If this precondition is not met, the program will exit with
//       status 1
// post: In description
void get_args(int argc, char *argv[], int &val_count, int &thread_count) {

    if (argc <= 2) {
        bad_usage();
    } else if (argc >= 4) {
        bad_usage();
    }

    val_count    = 0;
    thread_count = 0;

    val_count    = atoi(argv[1]);
    thread_count = atoi(argv[2]);

    if ((val_count <= 0) || (thread_count <= 0)) {
        bad_usage();
    }
}

// desc: returns an array of `count` integers filled with
//       random values
//  pre: `count` should be positive
// post: In description
int *random_array(int count) {
    int *result = new int[count];
    for (int i = 0; i < count; i++) {
        result[i] = rand();
    }
    return result;
}

// class Worker {
// public:
//     void operator()(int id, void *sharedData) {
//         std::cout << "Thread " << id << " is processing" << std::endl;
//         std::string *data = static_cast<std::string *>(sharedData);
//         std::cout << "Thread " << id << " received data: " << *data << std::endl;
//     }
// }

// desc: Calculates the element-wise multiplication and
//       addition of three arrays (A * B + C)
//  pre: Command-line arguments should consist of exactly
//       two arguments, both positive integers
// post: In description
int main(int argc, char *argv[]) {

    // A simple C++ program that uses OpenMP to set the number of threads and parallelize a loop:
    // - omp_set_num_threads(4); sets the number of threads to 4.
    // - #pragma omp parallel for is the directive that tells OpenMP to parallelize the following `for` loop.
    // - omp_get_thread_num() returns the ID of the thread executing the current iteration of the loop.

    // const int THREAD_COUNT = 4;

    int val_count, thread_count;
    get_args(argc, argv, val_count, thread_count);

    // Set the number of threads
    omp_set_num_threads(thread_count);
    // std::cout << "Value count is  : " << val_count << '\n';
    // std::cout << "Thread count is : " << thread_count << '\n';

    int *a = random_array(val_count);
    int *b = random_array(val_count);
    int *c = random_array(val_count);
    
    // Refactor this loop  so that it performs its primary loop (with C[i] = A[i] * B[i] + C[i]) using multiple threads.
    // The number of threads should be set to the value of the second command-line argument.
    // The number of iterations should be set to the value of the first command-line argument.
    // The loop should be parallelized using OpenMP.
    // #pragma omp parallel for
    for (int i = 0; i < thread_count; i++) {
        c[i] = a[i] * b[i] + c[i];
        // std::cout << "Thread " << i << " is processing iteration " << i << std::endl;
        std::cout << c[i] << std::endl;
    }

    // Parallelize this loop
    // #pragma omp parallel for
    // for (int i = 0; i < val_count; ++i) {
    //     int thread_id = omp_get_thread_num();
    //     std::cout << "Thread " << thread_id << " is processing iteration " << i << std::endl;
    //     c[i] = a[i] * b[i] + c[i];
    // }

    return 0;
}
