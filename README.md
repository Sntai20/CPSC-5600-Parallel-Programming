# CPSC-5600-Parallel-Programming

## To build and run the program

Build the project using the MakeFile in the project folder.

```bash
cd repos/CPSC-5600-Parallel-Programming
make -C p0
python AutomateDataCollection.py p0 p0
```

## Collect the data using the project folder name and the program name to test

The following is an example of collecting data using the p0 program from the p0 project:

```python
make -C p0
python AutomateDataCollection.py p0 p0
```

Example of collecting data using the example program from the p1 project:

```python
make -C p1
python AutomateDataCollection.py p1 example
```

Example of collecting data without sampling the data, using the example program from the p1 project:

```python
make -C p1
python AutomateDataCollection.py p1 example --nosample
```

## P2: Parallel Simulation

The following is an example of running the p2 program in the p2 folder:

```bash
make -C p2
p2/out/p2 3
```

### Gathering Data

To collect performance measurements gather the runtime of parallel_simulate method using the following step counts and division counts when simulating the provided WallOrbit input (specifically, WallOrbit(128,<duration>)) with display==0:

- Durations: [ 64, 128, 256, 512, 1024, 2048, 4096 ]
- Thread Counts: [ 1 2 3 4 5 6 7 8 ]

## Import CSV files into Excel (for Macs)

1. Save the output into output.csv.
1. Open Excel: Open a new or existing workbook.
1. Go to the Data Tab: Click on the Data tab on the Ribbon.
1. Get Data: Click on the Get Data (Power Query).
1. Select Text/CSV: Import data from a text or CSV file.
1. Select the CSV file to import: Click browse, select output.csv, Click Get Data, then Next.
1. Preview file data: Excel will show a preview of the data. You can adjust settings like delimiter (comma, semicolon, etc.) if needed. Once everything looks good, click Load.
1. Data Loaded: The CSV data will be imported into your Excel worksheet.

## P3: Implementing Barriers and Bitonic Sort

The program requires exactly two arguments, both positive integers. The following is an example of running the p3 program in the p3 folder:

```bash
make -C p3
p3/out/p3 3 3
```

### Compile the program with the custom class enabled

To compile the program with the custom class enabled, comment out line 7 and 8, then uncomment lines 10 and 11 in the MakeFile. An example is provided below.

```text
# p3: p3.cpp Makefile
# 	g++ p3.cpp -o out/p3 -lpthread -Wall -Werror --std=c++20

p3: p3.cpp Makefile
	g++ p3.cpp -o out/p3 -lpthread -Wall -Werror --std=c++20 -DCUSTOM_BARRIER
```

## P5: K-Means Clustering using MPI

The program requires exactly two arguments, both positive integers. The first argument is the cluster count and the second argument is the point count. The following is an example of running the p5 program in the p5 folder:

```bash
module try-add mpi
make -C p5
mpiexec -np 30 p5/out/p5 3 10000
```

### Valgrind

Valgrind is a memory management analysis tool.

```bash
mpirun -n 2 valgrind ./p5/out/p5 3 10000
```

### Gathering Data

Gather the runtime of parallel_k_means for the following cluster counts and points counts (per generated cluster):

- Cluster Counts: [ 2, 3, 4, 5, 6 ]
- Point Counts: [ 100000, 200000, 300000, 400000, 500000 ]

```bash
module try-add mpi
make -C p5
python AutomateDataCollection.py --mpi p5 p5
```