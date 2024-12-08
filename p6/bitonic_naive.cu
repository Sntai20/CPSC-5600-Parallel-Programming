/**
 * Kevin Lundeen, Seattle University, CPSC 5600
 * bitonic_naive.cu - a bitonic sort that only works when the j-loop fits in a single block
 *                  - n must be a power of 2
 */
#include <iostream>
#include <random>
#include "bitonic_naive.h"
#include "constants.h"
using namespace std;

// Swap two elements in the data array.
__device__ void swap(X_Y *data, int a, int b) {
    X_Y temp = data[a];
    data[a] = data[b];
    data[b] = temp;
}

// Bitonic sort for j <= MAX_BLOCK_SIZE.
__global__ void bitonic_small_j(X_Y *data, int k, int j) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int ixj = i ^ j;
    if (ixj > i) {
        if ((i & k) == 0 && data[i].x > data[ixj].x)
            swap(data, i, ixj);
        if ((i & k) != 0 && data[i].x < data[ixj].x)
            swap(data, i, ixj);
    }
    __syncthreads();
}

// Bitonic sort for j > MAX_BLOCK_SIZE. 
__global__ void bitonic_large_j(X_Y *data, int k, int j) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int ixj = i ^ j;
    if (ixj > i) {
        if ((i & k) == 0 && data[i].x > data[ixj].x)
            swap(data, i, ixj);
        if ((i & k) != 0 && data[i].x < data[ixj].x)
            swap(data, i, ixj);
    }
}

// Bitonic sort for n elements.
void bitonic_sort(X_Y *data, int n) {
    int num_blocks = (n + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;

    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            if (j <= MAX_BLOCK_SIZE) {
                bitonic_small_j<<<num_blocks, MAX_BLOCK_SIZE>>>(data, k, j);
                cudaDeviceSynchronize();
            } else {
                bitonic_large_j<<<num_blocks, MAX_BLOCK_SIZE>>>(data, k, j);
                cudaDeviceSynchronize();
            }
        }
    }
}