#!/bin/bash -i

# return error if env install path is not set
if [ -z "$1" ]; then
    echo "ENV_INSTALL_PATH is not set"
    exit 1
fi

ENV_INSTALL_PATH=$1

set -ex
mamba create --strict-channel-priority --yes -p $ENV_INSTALL_PATH \
python=3 pip setuptools \
compilers compilers sysroot_linux-64=2.17 gcc=11 ninja cmake make rust git pkg-config boost-cpp libboost-devel libprotobuf protobuf mkl mkl-include `# general dev tools` \
cuda-cudart-dev cuda-toolkit cuda-nvcc cuda-version cuda-libraries-dev `# CUDA dev tools` \
openssl=3.0.2 `# openmpi` \
-c pytorch -c nvidia/label/cuda-11.8.0 -c conda-forge

eval "$(conda shell.bash hook)" 
conda activate $ENV_INSTALL_PATH 
# openmpi
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.0.tar.bz2
tar jxf openmpi-5.0.0.tar.bz2
rm openmpi-5.0.0.tar.bz2
cd openmpi-5.0.0/
./configure --prefix=$ENV_INSTALL_PATH
make -j
make install
cd ..

