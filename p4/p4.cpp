#include <iostream>
#include <atomic>
#include <thread>
#include <random>
#include <chrono>
#include <memory>

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

    SharedPointer(const SharedPointer& other) : ptr(other.ptr), ref_count(other.ref_count) {
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
template<typename T>
class AtomicIterator {

    public:

    AtomicIterator(T* buffer, size_t size){
        // ???
    }

    AtomicIterator(AtomicIterator& other) {
        // ???
    }

    AtomicIterator& operator=(AtomicIterator& other) {
        // ???
    }

    T* next() {
        // ???
    }

};

#else
template<typename T>
using AtomicIterator = std::atomic<T>;
#endif

struct MyClass {
    MyClass() { std::cout << "MyClass created\n"; }
    ~MyClass() { std::cout << "MyClass destroyed\n"; }
};

// desc: A simple test function for the shared pointer implementation.
// pre: None
// post: In description
void shared_pointer_test() {
    std::cout << "Testing shared pointer" << std::endl;

    SharedPointer<MyClass> ptr1 = SharedPointer<MyClass>(new MyClass());
    {
        // Shared ownership
        SharedPointer<MyClass> ptr2 = ptr1;
        std::cout << "ptr2 is sharing ownership with ptr1\n";
    }
    // ptr2 goes out of scope, but the object is not destroyed because ptr1 still owns it

    std::cout << "ptr1 is the last owner\n";
}

int main() {
    
    // You may call the test drivers for your implementations here
    shared_pointer_test();

    return 0;
}