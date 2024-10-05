#include <cstdio>
#include <iostream>

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

    prefix_sum(data, value_count, thread_count);

    return 0;
}
