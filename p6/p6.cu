
#include <iostream>
#include <string>

#include "bitonic_naive.cu"
#include "reduce_scan_1block.cu"

using namespace std;

// Global Variables

// True for all CUDA architectures so far. This is the maximum number of threads per block.
const int MAX_BLOCK_SIZE = 1024;
const string filename = "x_y.csv";
const string output_filename = "x_y_scan.csv";


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
		// note that in later hardware (compute capabilty >= 7.0), there is a cuda::barrier available
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

    // Scans the y-values in the sorted sequence using a parallel prefix scan Links to an external site. in CUDA.
    
    
    // Writes the sorted sequence to a new file named x_y_scan.csv with four fields per line, in the following order: x value, y value, cumulative y value, original row number.
    
    return 0;
}