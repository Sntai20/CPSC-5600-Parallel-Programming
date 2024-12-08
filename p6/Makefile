# Makefile for compiling p6.cu

NVCC = nvcc
GPP = g++
CFLAGS = -arch=sm_60 -std=c++14
GPPFLAGS = -std=c++17
LDFLAGS = -lstdc++fs
TARGET_DIR = out
TEST_DIR = $(TARGET_DIR)/test
TARGET = $(TARGET_DIR)/p6

# Source files
SRCS = p6.cu bitonic_naive.cu reduce_scan_1block.cu p6_test.cu
CPP_SRCS = file_utils.cpp

# Object files
OBJS = $(patsubst %.cu, $(TARGET_DIR)/%.o, $(SRCS))
CPP_OBJS = $(patsubst %.cpp, $(TARGET_DIR)/%.o, $(CPP_SRCS))

# Header files
HEADERS = bitonic_naive.h reduce_scan_1block.h file_utils.h p6_test.h

# Default target
all: $(TARGET_DIR) $(TEST_DIR) $(TARGET) copy_test_files

# Makefile should build the other files into an executable named p6 if the make command make p6 is called. 
p6: $(TARGET_DIR) $(TEST_DIR) $(TARGET) copy_test_files

$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

$(TEST_DIR):
	mkdir -p $(TEST_DIR)

# Target to compile the CUDA and CPP source files
$(TARGET): $(OBJS) $(CPP_OBJS)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Target to compile the CUDA object files
$(TARGET_DIR)/%.o: %.cu $(HEADERS)
	$(NVCC) $(CFLAGS) -c $< -o $@

# Target to compile the C++ object files
$(TARGET_DIR)/%.o: %.cpp $(HEADERS)
	$(GPP) $(GPPFLAGS) -c $< -o $@

# Target to copy test files
copy_test_files: $(TEST_DIR)
	cp x_y/x_y_1*.csv $(TEST_DIR)
	cp x_y/x_y.csv $(TARGET_DIR)

# Target to clean the directory
clean:
	rm -f $(TARGET_DIR)/*.o $(TARGET)
	rm -rf $(TARGET_DIR)

.PHONY: clean