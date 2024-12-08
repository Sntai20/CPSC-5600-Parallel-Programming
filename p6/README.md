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

## Clean compilied and test files with Makefile

```bash
make clean
```