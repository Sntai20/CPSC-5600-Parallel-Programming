#ifndef BITONIC_NAIVE_H
#define BITONIC_NAIVE_H

#include "constants.h"
// __device__ void swap(float *data, int a, int b);
// __global__ void bitonic(float *data, int k);
// int bitonic_naive_sort(float *data, int n);
// int bitonic_naive_sort(int n = 0);
__device__ void swap(X_Y *data, int a, int b);
__global__ void bitonic_small_j(X_Y *data, int k, int j);
__global__ void bitonic_large_j(X_Y *data, int k, int j);
void bitonic_sort(X_Y *data, int n);

#endif // BITONIC_NAIVE_H