# DDL Emulator

## Prerequisite

We need `mamba` or `conda` installed before. You can install mamba [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)

We also need to setup remote nodes for `mpirun` or `torchrun`, make sure that you have `/etc/hosts` and `~/.ssh/config` properly configured. (i.e. you can ssh to the remote nodes using hostname without password)

## Conda Environment

Suppose $ENV_PATH is the path to the conda environment, then we can use the following command to create a conda environment. We need two environments, we call it `ori` and `emu`. `ori` is the unmodified original code, `emu` is the code with emulator:

We need to install dependencies for both.

```bash
mkdir ~/my_env
export $ENV_PATH=~/my_env
git submodule update --init --recursive
bash ./scripts/create_env.sh $ENV_PATH # takes about 30 minutes
nvcc --version
# expect:
# Cuda compilation tools, release 11.8, V11.8.89
# Build cuda_11.8.r11.8/compiler.31833905_0
```

and keep $ENV_PATH in the config.sh

```bash
touch config.sh
# in config.sh
export ENV_PATH=your_env_path
```

## NCCL

### Build

First, build the nccl and nccl make sure you have conda environment properly configurated.

We need to build nccl from source, for both `emu` and `ori`.

```bash
cd nccl
git switch [emu/ori] # switch to the branch you want to build 
cd ..
. ./config.sh
bash ./scripts/build_nccl.sh # should take less than 5 minutes
```

Add the following variables to config.sh that sets up the required environments:
You can use `query_gpu.py` to get the compute capability of your GPU.

```bash
#config for debug
export NCCL_DEBUG=MOD
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE=your_debug_file.$(date "+%Y-%m-%d %H:%M:%S")_%h:%p%h:%p
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
export NCCL_BUILD_PATH=your_nccl_build_path, a local file system like /tmp is recommended
export NVCC_GENCODE="-gencode=arch=compute_[your_compute],code=sm_[your_sm]"
# export ONLY_FUNCS="AllReduce Sum (f16|f32) RING SIMPLE"
# you can specify ONLY_FUNCS to reduce compile time
export ONLY_FUNCS="AllReduce Sum (f16|f32) RING SIMPLE"
export DEBUG=1
# for current experiments, we only use 2 nodes, 1 gpu per node
export CUDA_VISIBLE_DEVICES=0
export OMPI_COMM_WORLD_SIZE=2
export OMPI_COMM_WORLD_LOCAL_RANK=0
```

```bash
#config for release
export NCCL_DEBUG=VERSION
export NCCL_DEBUG_SUBSYS=INIT
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
export NCCL_BUILD_PATH=your_nccl_build_path, a local file system like /tmp is recommended
export NVCC_GENCODE="-gencode=arch=compute_[your_compute],code=sm_[your_sm]"
unset ONLY_FUNCS
export DEBUG=0
# for current experiments, we only use 2 nodes, 1 gpu per node
export CUDA_VISIBLE_DEVICES=0
export OMPI_COMM_WORLD_SIZE=2
export OMPI_COMM_WORLD_LOCAL_RANK=0
```

## Pytorch

After testing the nccl, we can use the nccl in pytorch.

### Build

We have to build python from source, given that we want to use our nccl.

```bash
cd pytorch
git switch [emu/ori] # switch to the branch you want to build
cd ..
bash ./scripts/build_pytorch.sh # takes about 30 mintues
conda activate $ENV_PATH
python
```

```python
import torch
torch.__version__ # expect 2.2.0a0+git8ac9b20
torch.cuda.is_available()
torch.cuda.nccl.version() # expect 2.19.4
```

## Eval

Please check the README.md in the eval folder for more details.

<!-- ### BERT

We need two terminal to run BERT experiments, one with emulator enabled, one with oringal(unmodifed) code, we refer the first as `emu` and the second as `ori`.

```bash
# emu
conda activate emu
. ./config_release.sh # use release build for nccl and pytorch
cd eval/BERT
CUDA_VISIBLE_DEVICES=0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_LOCAL_RANK=0 OMPI_COMM_WORLD_RANK=0 MOD_KERNEL_BYPASS=1 ./run.sh
```

```bash
# ori
conda activate ori
. ./config_release.sh # use release build for nccl and pytorch
cd eval/BERT
CUDA_VISIBLE_DEVICES=0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_LOCAL_RANK=0 OMPI_COMM_WORLD_RANK=1 MOD_KERNEL_BYPASS=0 ./run.sh
```

The result is saved in `eval/BERT/results`. -->