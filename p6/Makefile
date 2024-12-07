# Makefile for compiling p6.cu

NVCC = nvcc
CFLAGS = -arch=sm_60
TARGET_DIR = out
TARGET = $(TARGET_DIR)/p6

# Source files
SRCS = p6.cu bitonic_naive.cu reduce_scan_1block.cu

# Object files
OBJS = $(patsubst %.cu, $(TARGET_DIR)/%.o, $(SRCS))

# Header files
HEADERS = bitonic_naive.h reduce_scan_1block.h

# Default target
all: $(TARGET_DIR) $(TARGET)

$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

# Target to compile the CUDA source files
$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) -o $@ $^

# Target to compile the header files
$(TARGET_DIR)/%.o: %.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -c $< -o $@

# Target to clean the directory
clean:
	rm -f $(TARGET_DIR)/*.o $(TARGET)
	rm -rf $(TARGET_DIR)

.PHONY: clean