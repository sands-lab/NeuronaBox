#!/usr/bin/bash -i
eval "$(conda shell.bash hook)" 
set -ex
conda activate ~/env/nccl_mod

export CUDA_HOME=$CONDA_PREFIX

cd nccl
rm -rf build
make -j src.build
make install PREFIX=$CONDA_PREFIX

cd ../nccl-tests
rm -rf build
make -j NCCL_HOME=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX 

#cd ..
#make MPI=1 MPI_HOME=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX NCCL_HOME=$CONDA_PREFIX
set +ex