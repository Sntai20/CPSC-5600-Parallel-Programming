#include <iostream>
#include <atomic>
#include <thread>
#include <random>
#include <chrono>

template<typename T>
class SharedPointer {

    public:

    SharedPointer() {
        // ???
    }

    SharedPointer(T* raw_pointer) {
        // ???
    }

    SharedPointer(SharedPointer& other) {
        // ???
    }

    ~SharedPointer() {
        // ???
    }

    SharedPointer& operator=(SharedPointer& other) {
        // ???
    }

    T& operator*() {
        // ???
    }

    T* operator->() {
        // ???
    }

    operator bool() {
        // ???
    }

};



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



int main() {

    // You may call the test drivers for your implementations here

    return 0;
}