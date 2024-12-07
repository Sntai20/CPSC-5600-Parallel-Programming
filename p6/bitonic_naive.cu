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

/**
  * swaps the given elements in the given array
  * (note the __device__ moniker that says it can only
  *  be called from other device code; we don't need it
  *  here, but __device__ functions can return a value
  *  even though __global__'s cannot)
  */
__device__ void swap(float *data, int a, int b) {
	float temp = data[a];
	data[a] = data[b];
	data[b] = temp;
}

/**
  * inside of the bitonic sort loop for a particular value of i for a given value of k
  * (this function assumes j <= MAX_BLOCK_SIZE--bigger than that and we'd need to 
  *  synchronize across different blocks)
  */
__global__ void bitonic(float *data, int k) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	for (int j = k/2; j > 0; j /= 2) {
		int ixj = i ^ j;
		// avoid data race by only having the lesser of ixj and i actually do the comparison
		if (ixj > i) {
			if ((i & k) == 0 && data[i] > data[ixj])
				swap(data, i, ixj);
			if ((i & k) != 0 && data[i] < data[ixj])
				swap(data, i, ixj);
		}
		// wait for all the threads to finish before the next comparison/swap
		__syncthreads();
	}
}