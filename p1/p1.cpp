#include <cstdio>
#include <iostream>
#include <thread>
#include <chrono>

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

// This is the function you need to parallelize
// Whatever thread logic you use should be in this function, not in
// main. I'll be running this function myself with different thread_count
// inputs to check performance and correctness.
void prefix_sum(int *data, size_t size, size_t thread_count) {
    int sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum += data[i];
        data[i] = sum;
    }
}

// desc: Calculates the prefix sum of an array of random
//       integers
//  pre: Command-line arguments should consiste of exactly
//       two arguments, both positive integers
// post: In description
int main(int argc, char *argv[]) {

    int value_count, thread_count;
    get_args(argc, argv, value_count, thread_count);

    std::cout << "Value count is  : " << value_count << '\n';
    std::cout << "Thread count is : " << thread_count << '\n';

    int *data = random_array(value_count);
    
    TimePoint start_time = steady_clock::now();

    prefix_sum(data, value_count, thread_count);
    
    /*
    void prefix_sum(int *data, size_t size, size_t thread_count)

    This function performs a prefix sum over the array pointed by data, which has size elements. 
    As it is given to you, this function is single threaded and does nothing with the thread_count parameter.
    Your job now is to refactor p1.cpp so that its prefix_sum function performs its work using a number
    of threads matching its thread_count parameter.

    This refactor should use the C++ <thread> API for the added threading logic. Additionally, the
    parallel/concurrent logic you add in your refactor should be in prefix_sum or its subroutines, not
    in main. You may feel free to modify main for other purposes, such as gathering timing data.

    prefix_sum will be ran separately to check for correctness/performance, so your refactor should not
    change the signature of prefix_sum.
    */

    TimePoint end_time = steady_clock::now();

    TimeSpan span = duration_cast<TimeSpan>(end_time - start_time);

    // std::cout << "Execution time is : " << span.count() << '\n';
    std::cout << span.count();

    return 0;
}
