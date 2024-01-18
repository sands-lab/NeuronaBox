#!/usr/bin/bash -i
eval "$(conda shell.bash hook)" 
conda activate ~/env/nccl_mod



set -ex
export CUDA_HOME=$CONDA_PREFIX

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

nvcc -lmpi -lnccl -lcudart ./examples/ex1.cu -o ./build/ex1

mkdir -p /tmp/nccl-emulator

set +ex
