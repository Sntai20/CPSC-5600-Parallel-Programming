#include <cstdio>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <mutex>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::steady_clock;
using std::this_thread::sleep_for;
using TimePoint = std::chrono::steady_clock::time_point;
using TimeSpan = std::chrono::duration<double>;

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
int *random_array(size_t count) {
    int *result = new int[count];
    for (size_t i = 0; i < count; i++) {
        result[i] = rand();
    }
    return result;
}

// Calculates the partial sum of a segment of the array and updates
// the global sum using a mutex to ensure thread safety.
void partial_sum(int* data, size_t start, size_t end, int& sum, std::mutex& mtx) {
    int local_sum = 0;
    for (size_t i = start; i < end; ++i) {
        local_sum += data[i];
        data[i] = local_sum;
    }
    std::lock_guard<std::mutex> lock(mtx);
    sum += local_sum;
}

// This is the function you need to parallelize
// Whatever thread logic you use should be in this function, not in
// main. I'll be running this function myself with different thread_count
// inputs to check performance and correctness.
void prefix_sum(int *data, size_t size, size_t thread_count) {
    /*
    This function performs a prefix sum over the array pointed by data, which has size elements. 
    This function is single threaded and does nothing with the thread_count parameter.
    Refactor this function to perform its work using a number of threads matching its thread_count parameter.

    This refactor should use the C++ <thread> API for the added threading logic. Additionally, the
    parallel/concurrent logic you add in your refactor should be in prefix_sum or its subroutines, not
    in main. You may feel free to modify main for other purposes, such as gathering timing data.

    The refactor should not change the signature of prefix_sum.
    */

    // Ensures that the thread count is at least 1.
    if (thread_count == 0) {
        thread_count = 1;
    }

    // Initializes the threads, partial sums, and mutex.
    std::vector<std::thread> threads;
    std::vector<int> partial_sums(thread_count, 0);
    std::mutex mtx;

    // Calculates the chunk size and remainder.
    size_t chunk_size = size / thread_count;
    size_t remainder = size % thread_count;
    
    // Creates the threads and assigns them their respective segments of the array.
    size_t start = 0;
    for (size_t i = 0; i < thread_count; ++i) {
        // Calculates the end index of the segment.
        size_t end = start + chunk_size + (i < remainder ? 1 : 0);
        
        // Creates the thread and assigns it the segment.
        threads.emplace_back(partial_sum, data, start, end, std::ref(partial_sums[i]), std::ref(mtx));
        
        // Updates the start index.
        start = end;
    }

    // Join each of the threads to make sure all work has completed.
    for (size_t i = 0; i < threads.size(); ++i) {
        threads[i].join();
    }

    // Calculates the total sum and updates the array.
    int sum = 0;
    for (size_t i = 0; i < thread_count; ++i) {
        // Calculates the total sum.
        sum += partial_sums[i];
        size_t start = i * chunk_size + std::min(i, remainder);
        size_t end = (i + 1) * chunk_size + std::min(i + 1, remainder);
        for (size_t j = start; j < end; ++j) {
            data[j] += sum - partial_sums[i];
        }
    }
}

// desc: Calculates the prefix sum of an array of random integers
// pre: Command-line arguments should consiste of exactly
//       two arguments, both positive integers
// post: In description
int main(int argc, char *argv[]) {

    int value_count, thread_count;
    get_args(argc, argv, value_count, thread_count);

    // std::cout << "Value count is  : " << value_count << '\n';
    // std::cout << "Thread count is : " << thread_count << '\n';

    int *data = random_array(value_count);
    
    TimePoint start_time = steady_clock::now();

    prefix_sum(data, value_count, thread_count);

    TimePoint end_time = steady_clock::now();

    TimeSpan span = duration_cast<TimeSpan>(end_time - start_time);

    // std::cout << "Execution time is : " << span.count() << '\n';
    std::cout << span.count();

    return 0;
}