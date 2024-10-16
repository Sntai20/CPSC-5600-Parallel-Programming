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
p2/out/p2 1
```

### Gathering Data

To collect performance measurements gather the runtime of parallel_simulate method using the following step counts and division counts when simulating the provided WallOrbit input (specifically, WallOrbit(128,<duration>)) with display==0:

- Durations: [ 64, 128, 256, 512, 1024, 2048, 4096 ]
- Thread Counts: [ 1 2 3 4 5 6 7 8 ]