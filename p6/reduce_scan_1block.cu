/**
 * reduce_scan_1block.cu - using dissemination reduction for reducing and scanning a small array with CUDA
 * Kevin Lundeen, Seattle University, CPSC 5600 demo program
 * Notes:
 * - only works for one block (maximum block size for all of CUDA is 1024 threads per block)
 * - eliminated the remaining data races that were in reduce_scan_naive.cu
 * - algo requires power of 2 so we pad with zeros up to 1024 elements
 * - now much faster using block shared memory during loops (which also handily exposed the data races we had before)
 */

#include <iostream>
#include "reduce_scan_1block.h"
#include "constants.h"
using namespace std;

// Reduce the array in place. This is a parallel reduction.
__global__ void allreduce(float *data) {
	__shared__ float local[MAX_BLOCK_SIZE]; // 10x faster at least than global memory via data[]
        int gindex = threadIdx.x;
	int index = gindex;
	local[index] = data[gindex];
        for (int stride = 1; stride < blockDim.x; stride *= 2) {

		__syncthreads();  // wait for my writing partner to put his value in local before reading it
		int source = (index - stride) % blockDim.x;
		float addend = local[source];
		
		__syncthreads();  // wait for my reading partner to pull her value from local before updating it
        	local[index] += addend;
        }
	data[gindex] = local[index]; 
}

// Scan the array in place. This is a parallel prefix sum.
__global__ void scan(float *data) {
	__shared__ float local[MAX_BLOCK_SIZE];
	int gindex = threadIdx.x;
	int index = gindex;
	local[index] = data[gindex];
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		
		__syncthreads();  // cannot be inside the if-block 'cuz everyone has to call it!
		int addend = 0;
		if (stride <= index)
			addend = local[index - stride];

		__syncthreads();
		local[index] += addend;
	}
	data[gindex] = local[index];
}

// Fill the array with 1.0, 2.0, 3.0, ... n.  Pad the rest with 0.0.
void fillArray(float *data, int n, int sz) {
	for (int i = 0; i < n; i++)
		data[i] = (float)(i+1); // + (i+1)/1000.0;
	for (int i = n; i < sz; i++)
		data[i] = 0.0; // pad with 0.0's for addition
}

// Print the first and last m elements of the array.
void printArray(float *data, int n, string title, int m) {
	cout << title << ":";
	for (int i = 0; i < m; i++)
		cout << " " << data[i];
	cout << " ...";
	for (int i = n - m; i < n; i++)
		cout << " " << data[i];
	cout << endl;
}

// Original function that takes user input for the number of data elements.
int reduce_scan_1block(int n) {
	bool is_test = (n =! 0);
	float *data;
	int threads = MAX_BLOCK_SIZE;
	
	if (!is_test) {
		cout << "How many data elements? ";
		cin >> n;
	}

	if (n > threads) {
		cerr << "Cannot do more than " << threads << " numbers with this simple algorithm!" << endl;
		return 1;
	}
	
	cudaMallocManaged(&data, threads * sizeof(*data));
	fillArray(data, n, threads);
	if(!is_test) printArray(data, n, "Before");
	allreduce<<<1, threads>>>(data);
	cudaDeviceSynchronize();
	if(!is_test) printArray(data, n, "Reduce");
	fillArray(data, n, threads);
	scan<<<1, threads>>>(data);
	cudaDeviceSynchronize();
	if(!is_test) printArray(data, n, "Scan");
	cudaFree(data);
	return 0;
}

// Scans the y-values in the sorted sequence using a parallel prefix scan in CUDA.
__global__ void first_tier_scan(float *data, float *block_sums, int n) {
    __shared__ float local[MAX_BLOCK_SIZE];
    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    int index = threadIdx.x;

    if (gindex < n) {
        local[index] = data[gindex];
    } else {
        local[index] = 0.0f;
    }

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float addend = 0.0f;
        if (index >= stride) {
            addend = local[index - stride];
        }
        __syncthreads();
        local[index] += addend;
    }

    if (gindex < n) {
        data[gindex] = local[index];
    }

    if (index == blockDim.x - 1) {
        block_sums[blockIdx.x] = local[index];
    }
}

__global__ void top_tier_scan(float *block_sums, int num_blocks) {
    __shared__ float local[MAX_BLOCK_SIZE];
    int index = threadIdx.x;

    if (index < num_blocks) {
        local[index] = block_sums[index];
    } else {
        local[index] = 0.0f;
    }

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float addend = 0.0f;
        if (index >= stride) {
            addend = local[index - stride];
        }
        __syncthreads();
        local[index] += addend;
    }

    if (index < num_blocks) {
        block_sums[index] = local[index];
    }
}

// Propagates the block sums to the data array.
__global__ void propagate_prefixes(float *data, float *block_sums, int n) {
    int gindex = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && gindex < n) {
        data[gindex] += block_sums[blockIdx.x - 1];
    }
}

// Scans the y-values in the sorted sequence using a parallel prefix scan in CUDA.
void reduce_scan_1block(float *data, int n) {
    int num_blocks = (n + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
    float *block_sums;
    cudaMalloc(&block_sums, num_blocks * sizeof(float));

    first_tier_scan<<<num_blocks, MAX_BLOCK_SIZE>>>(data, block_sums, n);
    cudaDeviceSynchronize();

    if (num_blocks > 1) {
        top_tier_scan<<<1, MAX_BLOCK_SIZE>>>(block_sums, num_blocks);
        cudaDeviceSynchronize();

        propagate_prefixes<<<num_blocks, MAX_BLOCK_SIZE>>>(data, block_sums, n);
        cudaDeviceSynchronize();
    }

    cudaFree(block_sums);
}