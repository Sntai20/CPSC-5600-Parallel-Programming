#include <iostream>
#include <thread>
#include <cstdio>

// g++ lambdas.cpp -o out/prog
void merge_sort(int *data, size_t size) {
    std::cout << "merge_sort" << std::endl;
    if(size <= 1) {
        return;
    }

    int *left = data;
    size_t left_size = size/2;
    int *right = data + left_size;
    size_t right_size = size - left_size;

    std::thread left_thread(merge_sort, left, left_size);
    std::thread right_thread(merge_sort, right, right_size);
    left_thread.join();
    right_thread.join();

    int *left_end = right;
    int *right_end = right + right_size;
    int *iter = data;

    while(left < left_end && right < right_end) {
        if(*left < *right) {
            *iter = *left;
            left++;
        } else {
            *iter = *right;
            right++;
        }
        iter++;
    }
}

int main() {
    size_t const THREAD_COUNT = 10;
    size_t const SIZE = 16;
    int *array = new int[SIZE];

    std::thread team[THREAD_COUNT];
    for(size_t t = 0; t < THREAD_COUNT; t++) {
        size_t start = t*SIZE/THREAD_COUNT;
        size_t end = (t+1)*SIZE/THREAD_COUNT;
        if(t == THREAD_COUNT-1) {
            end = SIZE;
        }

        team[t] = std::thread([array,start,end]() {
            for(size_t i = 0; i < end; i++) {
                array[i] = rand() % 100;
            }
        });
    }

    for(size_t t = 0; t < THREAD_COUNT; t++) {
        team[t].join();
    }

    for(size_t i = 0; i < SIZE; i++) {
        std::cout << array[i] << " ";
    }

    return 0;
}