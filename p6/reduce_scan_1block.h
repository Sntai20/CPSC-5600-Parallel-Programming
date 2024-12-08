#ifndef REDUCE_SCAN_1BLOCK_H
#define REDUCE_SCAN_1BLOCK_H

void fillArray(float *data, int n, int seed);
void printArray(float *data, int n, std::string label, int max_elements=5);
// int reduce_scan_1block(float *data, int n, int block_size, int grid_size, int seed);
// int reduce_scan_1block(int n=0);
__global__ void allreduce(float *data);
__global__ void scan(float *data);
void reduce_scan_1block(float *data, int n);

#endif // REDUCE_SCAN_1BLOCK_H