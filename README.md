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