#include <iostream>
#include <thread>
#include <algorithm>
#include <vector>
#include <mutex>
#include <barrier>
#include <condition_variable>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::steady_clock;
using std::this_thread::sleep_for;
using TimePoint = std::chrono::steady_clock::time_point;
using TimeSpan = std::chrono::duration<double>;

#ifdef CUSTOM_BARRIER
// Implementation cannot use any synchronization primitives
// aside from mutexes.
class Barrier {
    private:
    std::mutex mutex;
    std::condition_variable conditionalVariable;
    ptrdiff_t count;
    ptrdiff_t expected;
    // const int busy_wait_duration = 10; // Busy wait duration in milliseconds

    public:
    // Do not change the signature of this method
    Barrier(ptrdiff_t expected){
        // Initialize both the initial expected count for each phase and
        // the current expected count for the first phase to expected.
        this->count = 0;
        this->expected = expected;
    }

    // Do not change the signature of this method
    // void arrive_and_wait() {
    //     // Increment the expected count by 1, then blocks at the
    //     // synchronization point for the current phase until the phase
    //     // completion step of the current phase is run.
    //     std::unique_lock<std::mutex> lock(mutex);
    //     count++;
    //     // Unlock the mutex and notify all threads if count is equal to expected.
    //     mutex.unlock();
    //     // If count is equal to expected, notify all threads and reset count to 0.
    //     if(count == expected){
    //         count = 0;
    //         conditionalVariable.notify_all();
    //     } else {
    //         // Otherwise, wait until count is equal to expected.
    //         conditionalVariable.wait(lock, [this](){return count == expected;});
    //     }
    // }

    // This is using a busy wait approach.
    // void arrive_and_wait() {
    //     std::unique_lock<std::mutex> lock(mutex);
    //     ++count;
    //     if (count < expected) {
    //         while (count < expected) {
    //             // Busy wait
    //             lock.unlock();
    //             std::this_thread::yield(); // Yield to other threads
    //             lock.lock();
    //         }
    //     } else {
    //         count = 0; // Reset for the next phase
    //     }
    // }

    // This is using a condition variable approach.
    // void arrive_and_wait() {
    //     std::unique_lock<std::mutex> lock(mutex);
    //     ++count;
    //     if (count < expected) {
    //         conditionalVariable.wait(lock, [this] { return count == 0; });
    //     } else {
    //         count = 0; // Reset for the next phase
    //         conditionalVariable.notify_all(); // Notify all waiting threads
    //     }
    // }

    // void arrive_and_wait() {
    //     std::unique_lock<std::mutex> lock(mutex);
    //     ++count;
    //     if (count < expected) {
    //         // Busy wait for a short duration.
    //         auto start = std::chrono::steady_clock::now();
    //         while (count < expected) {
    //             auto now = std::chrono::steady_clock::now();
    //             if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() > busy_wait_duration) {
    //                 // Switch to condition variable wait after busy wait duration.
    //                 conditionalVariable.wait(lock, [this] { return count == 0; });
    //                 break;
    //             }

    //             // Yield to other threads.
    //             std::this_thread::yield();
    //         }
    //     } else {
    //         count = 0; // Reset for the next phase
    //         conditionalVariable.notify_all(); // Notify all waiting threads
    //     }
    // }

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex);
        ++count;
        if (count < expected) {
            conditionalVariable.wait(lock, [this] { return count == 0; });
        } else {
            count = 0; // Reset for the next phase
            conditionalVariable.notify_all(); // Notify all waiting threads
        }
    }
};

#else
using Barrier = std::barrier<>;
#endif

// Simple function to test the barrier implementation.
void task(Barrier& barrier) {
    std::cout << "Task started\n";
    barrier.arrive_and_wait();
    std::cout << "Task completed\n";
}

void simple_barrier_test() {
    const int num_threads = 3;
    Barrier barrier(num_threads);
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(task, std::ref(barrier));
    }

    for (auto& t : threads) {
        t.join();
    }
}

// A serial implementation of bitonic sort, for reference. Feel free to
// use it as a skeleton for your parallel implementation.
void bitonic_sort(int64_t *data, size_t size) {
    for(size_t i=1; i<size; i<<=1) {
        for(size_t j=i; j>0; j>>=1) {
            size_t limit = (i==size) ? size / 2 : size;
            for (size_t k=0; k<limit; k++) {
                if( k & j ){
                    k += j-1;
                    continue;
                }
                size_t mask = i << 1;
                if(((k&mask) == 0) == (data[k] > data[k+j])){
                    std::swap(data[k],data[k+j]);
                }
            }
        }
    }
}


// Helper function for parallel bitonic sort
void parallel_bitonic_sort_helper(int64_t *data, size_t size, size_t start, size_t end, Barrier& barrier) {
    for (size_t i = 1; i < size; i <<= 1) {
        for (size_t j = i; j > 0; j >>= 1) {
            for (size_t k = start; k < end; k++) {
                if (k & j) {
                    k += j - 1;
                    continue;
                }
                size_t mask = i << 1;
                if (((k & mask) == 0) == (data[k] > data[k + j])) {
                    std::swap(data[k], data[k + j]);
                }
            }
            barrier.arrive_and_wait();
        }
    }
}

// Do not change the signature of this function
// void parallel_bitonic_sort(int64_t *data, size_t size, size_t thread_count) {

    // Parallel implementation goes here.

    // Synchronization should be achieved with the `Barrier` type, which should
    // be defined either through std::barrier or your custom Barrier class.

    // Aside from very basic things, like std::swap, you shouldn't be using any
    // std C++ library functions for implementing this function or any of its
    // subroutines. If a submission does use something like std::sort for this
    // logic, it would not get a good grade.

    // bitonic_sort(data,size);

    // Barrier barrier(thread_count);
    // std::vector<std::thread> threads;
    // size_t chunk_size = size / thread_count;

    // // Create threads to perform the parallel bitonic sort
    // for (size_t i = 0; i < thread_count; i++) {
    //     size_t start = i * chunk_size;
    //     size_t end = (i == thread_count - 1) ? size : (i + 1) * chunk_size; // Handle the last chunk
    //     threads.emplace_back(parallel_bitonic_sort_helper, data, size, start, end, std::ref(barrier));
    // }

    // // Join threads
    // for (auto& t : threads) {
    //     t.join();
    // }
// }

void bitonic_merge(int64_t *data, size_t low, size_t cnt, bool dir) {
    if (cnt > 1) {
        size_t k = cnt / 2;
        for (size_t i = low; i < low + k; ++i) {
            if (dir == (data[i] > data[i + k])) {
                std::swap(data[i], data[i + k]);
            }
        }
        bitonic_merge(data, low, k, dir);
        bitonic_merge(data, low + k, k, dir);
    }
}

void bitonic_sort_recursive(int64_t *data, size_t low, size_t cnt, bool dir) {
    if (cnt > 1) {
        size_t k = cnt / 2;
        bitonic_sort_recursive(data, low, k, true);
        bitonic_sort_recursive(data, low + k, k, false);
        bitonic_merge(data, low, cnt, dir);
    }
}

void thread_func(int64_t *data, size_t size, size_t thread_count, size_t tid, Barrier &barrier) {
    size_t chunk_size = size / thread_count;
    size_t low = tid * chunk_size;
    size_t high = (tid + 1) * chunk_size;

    for (size_t step = 2; step <= size; step *= 2) {
        for (size_t sub_step = step / 2; sub_step > 0; sub_step /= 2) {
            for (size_t i = low; i < high; ++i) {
                size_t ixj = i ^ sub_step;
                if (ixj > i) {
                    if ((i & step) == 0 && data[i] > data[ixj]) {
                        std::swap(data[i], data[ixj]);
                    }
                    if ((i & step) != 0 && data[i] < data[ixj]) {
                        std::swap(data[i], data[ixj]);
                    }
                }
            }
            barrier.arrive_and_wait();
        }
    }
}

void parallel_bitonic_sort(int64_t *data, size_t size, size_t thread_count) {
    Barrier barrier(thread_count);
    std::vector<std::thread> threads;

    for (size_t i = 0; i < thread_count; ++i) {
        threads.emplace_back(thread_func, data, size, thread_count, i, std::ref(barrier));
    }

    for (auto& t : threads) {
        t.join();
    }
}

// Helpful functions, defined below main
int64_t *random_array(size_t count);
void barrier_test();
void bitonic_test();
void get_args(int argc, char *argv[], int &val_count, int &thread_count);


int main() {
    simple_barrier_test();

    barrier_test();

    TimePoint start_time = steady_clock::now();
    bitonic_test();
    TimePoint end_time = steady_clock::now();
    
    TimeSpan runtime = duration_cast<TimeSpan>(end_time - start_time);
    std::cout << "Bitonic test runtime: " << runtime.count() << " seconds.\n";   

    return 0;
}



// desc: returns an array of `count` integers filled with random values
//  pre: `count` should be positive
// post: In description
int64_t *random_array(size_t count) {
    int64_t *result = new int64_t[count];
    for (size_t i = 0; i < count; i++) {
        result[i] = rand();
    }
    return result;
}



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


// desc: A simple test function for the parallel bitonic sorting function.
//       You will likely need to implement more fine-grained testing if you want to
//       debug your code effectively.
//  pre: None
// post: Uses the implementation of `parallel_bitonic_sort` on an array of random
//       integers and compares it with the sorting results of std::sort. Any
//       discrepancies are reported.
void bitonic_test() {

    std::cout << "Testing bitonic sort function.\n\n";

    // Make a power-of-two sized array filled with random values
    size_t magnitude = 20;
    size_t size = 1<<magnitude;
    int64_t *data = random_array(size);

    // Run random array through the C++ std library sorting function
    // to verify the results.
    int64_t *reference = new int64_t[size];
    for (size_t i=0; i<size; i++) {
        reference[i] = data[i];
    }
    std::sort(reference,reference+size);

    // Run our function
    parallel_bitonic_sort(data,size,1);

    // Report any discrepancies
    for (size_t i=0; i<size; i++) {
        if(data[i] != reference[i]){
            std::cout << "First output mismatch at index " <<i<<".\n";
            return;
        }
    }
    std::cout << "All output values match!\n";
}


// desc: A worker thread function for testing the `Barrier` implementation
//  pre: All integer inputs must be positive. `bar` must point to a valid `Barrier` object
// post: Prints a unique character based upon thread id `phase_count` times, with printing
//       synchronized between threads so that all characters appear exactly once per line.
void barrier_test_helper(size_t phase_count, size_t thread_id, size_t thread_count, Barrier* bar) {
    // Perform `phase_count` phases
    for(size_t i=0; i<phase_count; i++) {
        // Sleep for a random period of time
        uint8_t ms_sleep_count = rand() % 100;
        std::this_thread::sleep_for(std::chrono::milliseconds(ms_sleep_count));

        // Print a unique character based upon `thread_id`
        std::cout << (char)('A'+thread_id);

        // Wait for all other threads to print their unique character
        bar->arrive_and_wait();

        // Have thread 0 print a newline
        if(thread_id == 0){
            std::cout << '\n';
        }

        // Wait for the newline to print.
        bar->arrive_and_wait();
    }
}


// desc: A function for testing the `Barrier` implementation
//  pre: None
// post: Prints the characters 'A' through 'F' 10 times in parallel, with printing
//       synchronized between threads so that all characters appear exactly once per line.
void barrier_test() {

    std::cout << "Testing barrier functionality.\n\n";
    std::cout << "Each line should contain the same set of letters "
              << "with no duplicates.\n\n";

    // Seed the rng for the worker threads
    srand(time(0));
    size_t thread_count = 6;
    size_t phase_count  = 10;
    std::thread team[thread_count];

    // Set up barrier
    Barrier bar(thread_count);

    // Fork and join helper threads
    for(size_t i=0; i<thread_count; i++){
        team[i] = std::thread(barrier_test_helper,phase_count,i,thread_count,&bar);
    }
    for(size_t i=0; i<thread_count; i++){
        team[i].join();
    }

    // Extra newlines for readability
    std::cout << "\n\n";
}
