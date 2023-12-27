# Compiler settings
NVCC        = nvcc
MPICXX      = mpicxx
CUDA_FLAGS  = -lnccl
MPI_FLAGS   = 
LINK_FLAGS  = -lcudart

# Object files
OBJ         = main.o mul.o

# Executable name
EXECUTABLE  = main

# Make rules
all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJ)
	$(MPICXX) -o $@ $^ $(LINK_FLAGS)

main.o: main.cpp
	$(MPICXX) $(MPI_FLAGS) -c $< -o $@

mul.o: mul.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXECUTABLE)

.PHONY: all clean
