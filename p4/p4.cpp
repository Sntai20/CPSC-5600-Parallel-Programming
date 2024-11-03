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

    SharedPointer() {
        // Default constructs the pointer as null.
    }

    SharedPointer(T* raw_pointer) {
        // Captures a regular pointer and begins reference counting for it.
    }

    SharedPointer(SharedPointer& other) {
        // Copies the assigned SharedPointer into the constructing instance.
    }

    ~SharedPointer() {
        // Destructor.
    }

    SharedPointer& operator=(SharedPointer& other) {
        // Overwrites the assigned SharedPointer with the assigning SharedPointer.
    }

    T& operator*() {
        // Returns a reference to the storage pointed by the wrapped pointer.
    }

    T* operator->() {
        // Returns the wrapped pointer.
    }

    operator bool() {
        // Returns true if the wrapped pointer is not null.
        return true;
    }

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

int main() {
    std::cout << "Hello, World!" << std::endl;

    // You may call the test drivers for your implementations here
    SharedPointer<MyClass> ptr1 = SharedPointer<MyClass>(new MyClass());
    {
        SharedPointer<MyClass> ptr2 = ptr1; // Shared ownership
        std::cout << "ptr2 is sharing ownership with ptr1\n";
    } // ptr2 goes out of scope, but the object is not destroyed because ptr1 still owns it

    std::cout << "ptr1 is the last owner\n";
    return 0;
}