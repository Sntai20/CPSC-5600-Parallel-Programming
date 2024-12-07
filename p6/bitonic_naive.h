#ifndef BITONIC_NAIVE_H
#define BITONIC_NAIVE_H

__device__ void swap(float *data, int a, int b);
__global__ void bitonic(float *data, int k);

#endif // BITONIC_NAIVE_H