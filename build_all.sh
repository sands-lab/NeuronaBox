#!/usr/bin/bash -i
eval "$(conda shell.bash hook)" 
conda activate ~/env/nccl_mod

set -ex
export CUDA_HOME=$CONDA_PREFIX

# cd nccl
# rm -rf build
# make -j src.build
# make install PREFIX=$CONDA_PREFIX
# cd ..

cd nccl-tests
rm -rf build
make -j MPI=1 MPI_HOME=$CONDA_PREFIX NCCL_HOME=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX 
rm -rf build_mod
mv build build_mod
cd ..
#cd ..
#make MPI=1 MPI_HOME=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX NCCL_HOME=$CONDA_PREFIX
set +ex

conda activate ~/env/nccl_ori

set -ex
export CUDA_HOME=$CONDA_PREFIX

cd nccl-tests
rm -rf build
make -j MPI=1 MPI_HOME=$CONDA_PREFIX NCCL_HOME=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX 
rm -rf build_ori
mv build build_ori
cd ..

set +ex
