# P6: CUDA Sort and Scan

- p6.cu -- A refactored version of the started code with parallelization. The Makefile builds the other files into an executable named p6 if the make p6 command is called.

## Build with Makefile

The `x_y.csv` is copied from the `x_y` folder to the out folder using the `make p6` command. 

```bash
cd p6
make p6
```

## Run p6

Running the following command will have the program read the `out/x_y.csv`, process the file, and write a `out/x_y_scan.csv`.

```bash
out/p6
```

## Run p6 tests

Test files are copied from the `x_y` folder to the `out/test` folder using the `make p6` command. These test files are also used to run the p6 tests and write a `out/test/x_y_scan_16.csv`

```bash
out/p6 test
```

### Expected test output

```text
Running in test mode
Current working directory: /home/st/asantana1/p6
Reading from: out/test/x_y_16.csv
Writing to: out/test/x_y_scan_16.csv


Running tests
Running file_utils_read_csv_test
Test passed: file_utils_read_csv_test works correctly.
Running bitonic_naive_sort_test
Test passed: bitonic_sort function works correctly.
Running reduce_scan_1block_test
Test passed: reduce_scan_1block function works correctly.
Running file_utils_write_csv_test
Test passed: write_csv function works correctly.
```

## Clean compiled and test files with Makefile

```bash
make clean
```

## p6 folder structure

An example of the expected folder structure after compiling and running the tests.

```text
├── bitonic_naive.cu
├── bitonic_naive.h
├── constants.h
├── file_utils.cpp
├── file_utils.h
├── Makefile
├── out
│   ├── bitonic_naive.o
│   ├── file_utils.o
│   ├── p6
│   ├── p6.o
│   ├── p6_test.o
│   ├── reduce_scan_1block.o
│   ├── test
│   │   ├── x_y_100.csv
│   │   ├── x_y_1024.csv
│   │   ├── x_y_16.csv
│   │   └── x_y_scan_16.csv
│   ├── x_y.csv
│   └── x_y_scan.csv
├── p6.cu
├── p6_test.cu
├── p6_test.h
├── README.md
├── reduce_scan_1block.cu
├── reduce_scan_1block.h
└── x_y
    ├── x_y_100.csv
    ├── x_y_1024.csv
    ├── x_y_16.csv
    ├── x_y.csv
    ├── x_y_scan_100.csv
    ├── x_y_scan_16.csv
    └── x_y_scan.csv
```

## Profiling

```text
nvprof ./out/p6
Running in normal mode
==621365== NVPROF is profiling process 621365, command: ./out/p6
==621365== Profiling application: ./out/p6
==621365== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   64.91%  22.979ms       165  139.27us  105.31us  239.48us  bitonic_small_j(X_Y*, int, int)
                   15.54%  5.5004ms        45  122.23us  111.20us  147.93us  bitonic_large_j(X_Y*, int, int)
                    9.83%  3.4782ms         2  1.7391ms  764.23us  2.7140ms  [CUDA memcpy HtoD]
                    8.94%  3.1658ms         2  1.5829ms  561.04us  2.6047ms  [CUDA memcpy DtoH]
                    0.62%  218.17us         1  218.17us  218.17us  218.17us  first_tier_scan(float*, float*, int)
                    0.15%  53.214us         1  53.214us  53.214us  53.214us  propagate_prefixes(float*, float*, int)
                    0.02%  5.6640us         1  5.6640us  5.6640us  5.6640us  top_tier_scan(float*, int)
      API calls:   73.34%  109.81ms         3  36.602ms  130.75us  109.54ms  cudaMalloc
                   19.54%  29.259ms       213  137.36us  7.4570us  239.31us  cudaDeviceSynchronize
                    5.14%  7.6889ms         4  1.9222ms  878.09us  3.0121ms  cudaMemcpy
                    0.80%  1.1907ms       213  5.5900us  3.5040us  214.92us  cudaLaunchKernel
                    0.71%  1.0630ms         3  354.34us  122.68us  521.12us  cudaFree
                    0.45%  672.21us       228  2.9480us     260ns  139.79us  cuDeviceGetAttribute
                    0.01%  22.109us         2  11.054us  7.4330us  14.676us  cuDeviceGetName
                    0.01%  8.2890us         2  4.1440us  1.6400us  6.6490us  cuDeviceGetPCIBusId
                    0.00%  3.1260us         3  1.0420us     383ns  2.2030us  cuDeviceGetCount
                    0.00%  1.5530us         4     388ns     247ns     793ns  cuDeviceGet
                    0.00%     948ns         2     474ns     376ns     572ns  cuDeviceTotalMem
                    0.00%     734ns         2     367ns     282ns     452ns  cuDeviceGetUuid
                    0.00%     550ns         1     550ns     550ns     550ns  cuModuleGetLoadingMode
```