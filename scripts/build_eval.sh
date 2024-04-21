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

nvcc -lmpi -lnccl -lcudart -O3 ./eval/coll/all_reduce.cu -o ./build/all_reduce
