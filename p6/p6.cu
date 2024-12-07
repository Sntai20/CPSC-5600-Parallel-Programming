
#include <iostream>
#include <random>
#include <string>

#include "constants.h"
#include "bitonic_naive.h"
#include "reduce_scan_1block.h"

using namespace std;

int bitonic_native_sort() {
	// const int MAX_BLOCK_SIZE = 1024; // true for all CUDA architectures so far
	int n;
	cout << "n = ? (must be power of 2): ";
	cin >> n;
	if (n > MAX_BLOCK_SIZE || pow(2,floor(log2(n))) != n) {
		cerr << "n must be power of 2 and <= " << MAX_BLOCK_SIZE << endl;
		return 1;
	}

	// use managed memory for the data array
	float *data;
	cudaMallocManaged(&data, n * sizeof(*data));

	// fill it with random values
	random_device r;
	default_random_engine gen(r());
	uniform_real_distribution<float> rand(-3.14, +3.14);
	for (int i = 0; i < n; i++)
		data[i] = rand(gen);

	// sort it with naive bitonic sort
	for (int k = 2; k <= n; k *= 2) {
		// coming back to the host between values of k acts as a barrier
		// note that in later hardware (compute capabilty >= 7.0), there is a cuda::barrier avaliable
		bitonic<<<1, MAX_BLOCK_SIZE>>>(data, k);
	}
	cudaDeviceSynchronize();

	// print out results
	for (int i = 0; i < n; i++)
		if (i < 3 || i >= n - 3 || i % 100 == 0)
			cout << data[i] << " ";
		else
			cout << ".";
	cout << endl;
    cudaFree(data);
	return 0;
}

int reduce_scan_1block(void) {
	int n;
	float *data;
	int threads = MAX_BLOCK_SIZE;
	cout << "How many data elements? ";
	cin >> n;
	if (n > threads) {
		cerr << "Cannot do more than " << threads << " numbers with this simple algorithm!" << endl;
		return 1;
	}
	cudaMallocManaged(&data, threads * sizeof(*data));
	fillArray(data, n, threads);
	printArray(data, n, "Before");
	allreduce<<<1, threads>>>(data);
	cudaDeviceSynchronize();
	printArray(data, n, "Reduce");
	fillArray(data, n, threads);
	scan<<<1, threads>>>(data);
	cudaDeviceSynchronize();
	printArray(data, n, "Scan");
	cudaFree(data);
	return 0;
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    // Reads a large csv file named x_y.csv, containing up to one million (x,y) pairs of floating-point numbers.
    // The file has two fields per line, separated by a comma: the x value and the y value.
    /* Example:
    x,y
    12.278249,1.152063
    13.043502,0.114110
    */

    // Sorts this sequence o (x,y) pairs by their x-values, using a parallel bitonic sort in CUDA.
    bitonic_native_sort();

    // Scans the y-values in the sorted sequence using a parallel prefix scan Links to an external site. in CUDA.
    reduce_scan_1block();
    
    // Writes the sorted sequence to a new file named x_y_scan.csv with four fields per line, in the following order: x value, y value, cumulative y value, original row number.
    
    return 0;
}