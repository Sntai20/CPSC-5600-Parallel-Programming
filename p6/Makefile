# Makefile for compiling p6.cu

# Compiler
NVCC = nvcc

# Source file
SRC = p6.cu bitonic_naive.cu reduce_scan_1block.cu

# Output directory and file
OUT_DIR = out
OUT = $(OUT_DIR)/p6

# Default target
all: $(OUT)

# Rule to compile the CUDA source file
$(OUT): $(SRC)
	@mkdir -p $(OUT_DIR)
	$(NVCC) $(SRC) -o $(OUT)

# Clean target to remove compiled files
clean:
	rm -f $(OUT)

.PHONY: all clean