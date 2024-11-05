#include <iostream>
#include <atomic>
#include <thread>
#include <random>
#include <chrono>
#include <memory>
#include <vector>
#include <cstddef>

#ifdef CUSTOM_SHAREDPOINTER
template<typename T>
class SharedPointer {

public:
    SharedPointer() : ptr(nullptr), ref_count(new int(1)) {
        // Default constructs the pointer as null.
    }

    SharedPointer(T* raw_pointer) : ptr(raw_pointer), ref_count(new int(1)) {
        // Captures a regular pointer and begins reference counting for it.
    }

    SharedPointer(SharedPointer& other) : ptr(other.ptr), ref_count(other.ref_count) {
        // Copies the assigned SharedPointer into the constructing instance.
        ++(*ref_count);
    }

    ~SharedPointer() {
        // Destructor.
        if (--(*ref_count) == 0) {
            delete ptr;
            delete ref_count;
        }
    }

    SharedPointer& operator=(SharedPointer& other) {
        // Overwrites the assigned SharedPointer with the assigning SharedPointer.
        if (this != &other) {
            if (--(*ref_count) == 0) {
                delete ptr;
                delete ref_count;
            }
            ptr = other.ptr;
            ref_count = other.ref_count;
            ++(*ref_count);
        }

        return *this;
    }

    T& operator*() {
        // Returns a reference to the storage pointed by the wrapped pointer.
        return *ptr;
    }

    T* operator->() {
        // Returns the wrapped pointer.
        return ptr;
    }

    operator bool() {
        // Returns true if the wrapped pointer is not null.
        return ptr != nullptr;
    }

private:
    T* ptr;
    int* ref_count;
};

#else
template<typename T>
using SharedPointer = std::shared_ptr<T>;
#endif

#ifdef CUSTOM_ATOMICITERATOR
// AtomicIterator is a class that allows multiple threads to share a common iterator
// over an array. Each time the next method of an AtomicIterator is called, that call
// should return either a pointer to an element within the associated array or null.
// For an AtomicIterator with buffer size N, after N calls to next, every element in
// that array must have been returned exactly once. Additionally, after the Nth call
// to next, all subsequent calls should return null.

// Like with a SharedPointer, copies of an AtomicIterator should share the same atomic
// counter. Hence, if an AtomicIterator makes N/2 calls to next and a copy makes N/2
// calls to next, no further calls to next should return non-null pointers.
template<typename T>
class AtomicIterator {
    
public:
    AtomicIterator(T* buffer, size_t size) : buffer(buffer), size(size), index(0) {
        // Constructs a new AtomicIterator over the array buffer with size elements.
    }

    AtomicIterator(AtomicIterator& other) : buffer(other.buffer), size(other.size), index(other.index.load()) {
        // Copies the assigned AtomicIterator to the constructing instance.
    }

    AtomicIterator& operator=(AtomicIterator& other) {
        // Overwrites the assigned AtomicIterator with the assigning AtomicIterator.
        if (this != &other) {
            buffer = other.buffer;
            size = other.size;
            index.store(other.index.load());
        }
        return *this;
    }

    T* next() {
        // Atomically increments the index and returns the next element or null if out of bounds.
        size_t currentIndex = index.fetch_add(1);
        if (currentIndex < size) {
            return &buffer[currentIndex];
        } else {
            return nullptr;
        }
    }

private:
    T* buffer;
    size_t size;
    std::atomic<size_t> index;
};

#else
template<typename T>
using AtomicIterator = std::atomic<T>;
#endif

// desc: A simple class to test the shared pointer implementation.
struct MyClass {
    MyClass() { std::cout << "MyClass created\n"; }
    ~MyClass() { std::cout << "MyClass destroyed\n"; }
};

// desc: A simple test function for the shared pointer implementation.
// pre: None
// post: In description
void shared_pointer_test() {
    std::cout << "\nTesting shared pointer" << std::endl;

    SharedPointer<MyClass> ptr1 = SharedPointer<MyClass>(new MyClass());
    {
        // Shared ownership
        SharedPointer<MyClass> ptr2 = ptr1;
        std::cout << "ptr2 is sharing ownership with ptr1\n";
    }
    // ptr2 goes out of scope, but the object is not destroyed because ptr1 still owns it.

    std::cout << "ptr1 is the last owner\n";
}

// desc: A simple test function for the atomic iterator implementation.
void testAtomicIterator(AtomicIterator<int>& it, int thread_id) {
    while (true) {
        // Get the next value from the iterator.
        int* value = it.next();

        // If the value is null, we have reached the end of the buffer.
        if (value == nullptr) {
            break;
        }
        std::cout << "Thread " << thread_id << " got value: " << *value << std::endl;
    }
}

// desc: A simple test function for the atomic iterator implementation.
// pre: None
// post: In description
void atomic_iterator_test() {
    std::cout << "\nTesting atomic iterator" << std::endl;

    const size_t size = 10;
    int buffer[size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Create an AtomicIterator over the buffer.
    AtomicIterator<int> it(buffer, size);

    const int num_threads = 4;
    std::vector<std::thread> threads;

    // Start multiple threads to iterate over the buffer.
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(testAtomicIterator, std::ref(it), i);
    }

    // Wait for all threads to finish.
    for (auto& thread : threads) {
        thread.join();
    }
}

// desc: A simple test function to test self-assignment of shared pointers.
void self_assignment_test() {
    SharedPointer<int> ptr(new int(10));
    ptr = ptr; // Self-assignment
    std::cout << "\nSelf-assignment test passed\n";
}

// desc: A simple test function to test circular references in shared pointers.
struct Node {
    SharedPointer<Node> next;
};

// desc: A simple test function to test circular references in shared pointers.
void circular_reference_test() {
    SharedPointer<Node> node1(new Node());
    SharedPointer<Node> node2(new Node());
    node1->next = node2;
    node2->next = node1; // Circular reference
    std::cout << "Circular reference test passed (manual check for memory leak)\n";
}

// desc: A simple test function to test the atomic iterator with an empty buffer.
void empty_buffer_test() {
    int* buffer = nullptr;
    AtomicIterator<int> it(buffer, 0);
    int* value = it.next();
    if (value == nullptr) {
        std::cout << "Empty buffer test passed\n";
    } else {
        std::cout << "Empty buffer test failed\n";
    }
}

// desc: A simple test function to test concurrent modifications of the atomic iterator.
// void concurrent_modifications_test() {
//     const size_t size = 10;
//     int buffer[size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
//     AtomicIterator<int> it(buffer, size);

//     // Start two threads that will concurrently modify the iterator.
//     std::thread t1(&it {
//         for (int i = 0; i < 5; ++i) {
//             int* value = it.next();
//             if (value) {
//                 std::cout << "Thread 1 got value: " << *value << std::endl;
//             }
//         }
//     });

//     std::thread t2(&it {
//         for (int i = 0; i < 5; ++i) {
//             int* value = it.next();
//             if (value) {
//                 std::cout << "Thread 2 got value: " << *value << std::endl;
//             }
//         }
//     });

//     t1.join();
//     t2.join();
//     std::cout << "Concurrent modifications test passed\n";
// }


int main() {
    
    // You may call the test drivers for your implementations here
    shared_pointer_test();

    atomic_iterator_test();

    self_assignment_test();

    circular_reference_test();

    empty_buffer_test();

    // concurrent_modifications_test();

    return 0;
}