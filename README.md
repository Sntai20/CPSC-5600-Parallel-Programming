# CPSC-5600-Parallel-Programming

## To build and run the program

Build the project using the MakeFile in the project folder.

```bash
cd repos/CPSC-5600-Parallel-Programming
make -C p0
python AutomateDataCollection.py p0 p0
```

## Collect the data using the project folder name and the program name to test

The followling is an example of collecting data using the p0 program from the p0 project:

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