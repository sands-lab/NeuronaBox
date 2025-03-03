#!/usr/bin/bash -i
eval "$(conda shell.bash hook)" 

# call this after installing nccl

# return error if env install path is not set
if [ -z "$ENV_PATH" ]; then
    echo "ENV_PATH is not set"
    exit 1
fi

conda activate $ENV_PATH

set -ex

export CONDA_LIBMAMBA_SOLVER_NO_CHANNELS_FROM_INSTALLED=1

mamba install -y cmake ninja
mamba install -y mkl-static mkl-include -c https://software.repos.intel.com/python/conda --override-channels
mamba install -y -c pytorch magma-cuda118 # 11.8 is our cuda version

cd pytorch
yes | pip install -r requirements.txt

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export NCCL_INCLUDE_DIR="$CONDA_PREFIX/include/" 
export NCCL_LIB_DIR="$CONDA_PREFIX/lib" 

USE_SYSTEM_NCCL=1 python setup.py develop

# then wait for the build to finish

set +ex
