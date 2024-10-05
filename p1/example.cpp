#include <cstdio>
#include <cstdlib>
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

// desc: returns an array of `count` integers filled with random values
//  pre: `count` should be positive
// post: In description
int *random_array(int count) {
    int *result = new int[count];
    for (int i = 0; i < count; i++) {
        result[i] = rand();
    }
    return result;
}


// desc: performs a series of `size` multiply/add operations component-wise
//       across the supplied arrays
//  pre: `size` should be positive, and zero through `size` should be valid
//       indexes for all input arrays
// post: In description
void multiply_add(int *a, int *b, int *c, size_t size) {
    for(int i=0; i<size; i++){
        c[i] = a[i] * b[i] + c[i];
    }
}


// desc: performs a series of `value_count` multiply/add operations component-wise
//        across the supplied arrays using `thread_count` threads
//  pre: `size` should be positive, and zero through `size` should be valid
//        indexes for all input arrays, `thread_count` should be positive
// post: In description
void parallel_multiply_add(int *a, int *b, int *c, size_t value_count, size_t thread_count) {
    // Here, we add one less than `thread_count` to `value_count` so that
    // the result rounds up. This makes the for loop easier to implement.
    int values_per_thread = ( value_count + (thread_count-1) ) / thread_count;

    // An array for storing our std::thread handles
    std::thread mul_add_threads[thread_count];

    // Give each thread a different `values_per_thread`-sized chunks of the
    // a/b/c arrays
    for (size_t i = 0; i < thread_count; i++) {
        int  index = values_per_thread * i;
        int *sub_a = &a[index];
        int *sub_b = &b[index];
        int *sub_c = &c[index];
        int sub_size = values_per_thread;

        // Limit the size for the final thread to account for when
        // `value_count` cannot be evenly divided across threads
        if( i == thread_count-1 ) {
            sub_size = value_count - (i * values_per_thread);
        }

        // Launch the thread, supplying the function followed by its
        // argument list
        mul_add_threads[i] = std::thread(multiply_add,sub_a,sub_b,sub_c,sub_size);
    }

    // Join each of the threads to make sure all work has completed
    for(int i=0; i<thread_count; i++) {
        mul_add_threads[i].join();
    }
}



// desc: Calculates the element-wise multiplication and addition of three
//       arrays (A * B + C)
//  pre: Command-line arguments should consiste of exactly two arguments, both
//       positive integers
// post: In description
int main(int argc, char *argv[]) {

    int value_count, thread_count;
    get_args(argc, argv, value_count, thread_count);

    // std::cout << "Value count is  : " << value_count << '\n';
    // std::cout << "Thread count is : " << thread_count << '\n';

    int *a = random_array(value_count);
    int *b = random_array(value_count);
    int *c = random_array(value_count);

    TimePoint start_time = steady_clock::now();

    parallel_multiply_add(a,b,c,value_count,thread_count);

    TimePoint end_time = steady_clock::now();

    TimeSpan span = duration_cast<TimeSpan>(end_time - start_time);

    // std::cout << "Execution time is : " << span.count() << '\n';
    std::cout << span.count();

    return 0;
}