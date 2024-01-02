# Specify the CUDA Toolkit path if necessary
CUDA_PATH ?= $(CONDA_PREFIX)

# Architecture
GPU_ARCH ?= sm_60

# Compilers
NVCC := $(CUDA_PATH)/bin/nvcc
# Flags
NVCC_FLAGS := -arch=$(GPU_ARCH) -lineinfo
LDFLAGS := -lmpi -lnccl -lcudart

# Directories
SRC_DIR := examples
BIN_DIR := build

SOURCES := $(wildcard $(SRC_DIR)/*.cu)
EXECS := $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%.exe,$(SOURCES))

# Rules
.PHONY : all clean

default : all

all: $(EXECS)

$(BIN_DIR)/%.exe: $(SRC_DIR)/%.cu $(SRC_DIR)/%.h ; $(NVCC) $(LDFLAGS) $(NVCC_FLAGS) $< -o $@

clean: ; rm -f $(BIN_DIR)/*.exe
