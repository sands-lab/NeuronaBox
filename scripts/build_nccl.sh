#!/usr/bin/bash -i
eval "$(conda shell.bash hook)" 

# return error if env install path is not set
if [ -z "$ENV_PATH" ]; then
    echo "ENV_PATH is not set"
    exit 1
fi

# conda activate ~/env/nccl_mod
conda activate $ENV_PATH

set -ex
export CUDA_HOME=$CONDA_PREFIX

# NCCL_BUILD_PATH is used to build nccl, where local storage is preferred to reduce build time
# return error if NCCL_BUILD_PATH is not set
if [ -z "$NCCL_BUILD_PATH" ]; then
    echo "NCCL_BUILD_PATH is not set"
    exit 1
fi

rm -rf $NCCL_BUILD_PATH/nccl
cp -r nccl $NCCL_BUILD_PATH
cd $NCCL_BUILD_PATH/nccl
rm -rf build
make -j src.build
make install PREFIX=$CONDA_PREFIX
cd -
cp $NCCL_BUILD_PATH/nccl/build/obj/device/gensrc/host_table.cc ./build

# # cd nccl-tests
# rm -rf build
# make -j MPI=1 MPI_HOME=$CONDA_PREFIX NCCL_HOME=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX 
# rm -rf build_mod
# mv build build_mod
# cd ..

# set +ex

# conda activate ~/env/nccl_ori

# set -ex
# export CUDA_HOME=$CONDA_PREFIX

# cd nccl-tests
# rm -rf build
# make -j MPI=1 MPI_HOME=$CONDA_PREFIX NCCL_HOME=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX 
# rm -rf build_ori
# mv build build_ori
# cd ..

# build examples
nvcc -lmpi -lnccl -lcudart ./tests/nccl/ex0.cu -o ./build/ex0
nvcc -lmpi -lnccl -lcudart ./tests/nccl/ex0_ori.cu -o ./build/ex0_ori
nvcc -lmpi -lnccl -lcudart ./tests/nccl/ex1.cu -o ./build/ex1
nvcc -lmpi -lnccl -lcudart ./tests/nccl/ex1_ori.cu -o ./build/ex1_ori
nvcc -lmpi -lnccl -lcudart ./tests/nccl/ex2.cu -o ./build/ex2
nvcc -lmpi -lnccl -lcudart ./tests/nccl/ex3.cu -o ./build/ex3 
nvcc -lmpi -lnccl -lcudart ./tests/nccl/ex4.cu -o ./build/ex4
nvcc -lmpi -lnccl -lcudart ./tests/nccl/ex5.cu -o ./build/ex5
 
mkdir -p /tmp/nccl-emulator

set +ex
