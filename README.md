# DDL Emulator

## Prerequisite

We need `mamba` or `conda` installed before. You can install mamba [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)

We also need to setup remote nodes for `mpirun` or `torchrun`, make sure that you have `/etc/hosts` and `~/.ssh/config` properly configured. (i.e. you can ssh to the remote nodes using hostname without password)

## Conda Environment

Suppose $ENV_PATH is the path to the conda environment, then we can use the following command to create a conda environment:

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

If you want to build emulator, make sure `nccl` repo is in branch `emu`, if you want to build $N_0$, make sure `nccl` repo is in branch `ori`.

```bash
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

### Run

Simple example (2 nodes, 1 gpu per node):

```bash
. ./config.sh
mpirun -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_DEBUG_FILE -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H [node1]:1,[node2]:1  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include ens3f1 sh -c "./build/ex1 [#size] [#loop] > /tmp/nccl-emulator/log_debug$(date "+%m-%d-%H:%M:%S")"
```

<!-- Advanced example (2 nodes, 2 gpus per node):

```bash
mpirun -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG_FILE -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H [node1]:1,[node2]:1  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include ens3f1 sh -c "./build/ex2 [#size] [#loop] 2 > /tmp/nccl-emulator/log_debug$(date "+%m-%d-%H:%M:%S")"
``` -->

<!-- mpirun -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG_FILE -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H mcnode02:1,mcnode06:1  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include ens3f1 sh -c "/mnt/scratch/liub0a/nccl-emulator/ex1 100000 10" -->

 <!-- mpirun -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG_FILE -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H mcnode39:1,mcnode40:1  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include enp1s0f0  sh -c "./build/ex2 1000 1 2" -->

## Pytorch

After testing the nccl, we can use the nccl in pytorch.

### Build

We have to build python from source, given that we want to use our nccl.


If you want to build emulator, make sure `pytorch` repo is in branch `emu`, if you want to build $N_0$, make sure `pytorch` repo is in branch `ori`.

```bash
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

### All Reduce Test

todo!
<!-- ```bash
export WORLD_SIZE=2
export RANK=0
export LOCAL_RANK=1
export MASTER_ADDR=your master ip
export MASTER_PORT=any unused port
export MOD_KERNEL_BYPASS=1
export MOD_NNODES=2
export MOD_MY_NODE=0
python ./py-examples/all_reduce.py
``` -->
