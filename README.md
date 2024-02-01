# Conda Environment

Suppose $ENV_PATH is the path to the conda environment, then we can use the following command to create a conda environment:

```
git submodule update --init --recursive
bash ./scripts/create_env.sh $ENV_PATH
```

and keep $ENV_PATH in the config.sh

```bash
# config.sh
export ENV_PATH=your_env_path

```

# NCCL

## Build

First, build the nccl and nccl make sure you have conda environment properly configurated.

```
bash ./scripts/build_nccl.sh
```

Add the following variables to config.sh that sets up the required environments:

```bash
export NCCL_DEBUG=MOD
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE=your_debug_file.$(date "+%Y-%m-%d %H:%M:%S")_%h:%p%h:%p
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
export NCCL_BUILD_PATH=your_nccl_build_path, a local file system like /tmp is recommended
export NVCC_GENCODE="-gencode=arch=compute_[your_compute],code=sm_[your_sm]"
export ONLY_FUNCS="AllReduce Sum (f16|f32) RING SIMPLE"
```

## Run

### examples

Simple example (2 nodes, 1 gpu per node): 

```
. ./config.sh
mpirun -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG_FILE -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H [node1]:1,[node2]:1  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include ens3f1 sh -c "./build/ex1 [#size] [#loop] > /tmp/nccl-emulator/log_debug$(date "+%m-%d-%H:%M:%S")"
```

Advanced example (2 nodes, 2 gpus per node): 

```
mpirun -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG_FILE -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H [node1]:1,[node2]:1  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include ens3f1 sh -c "./build/ex2 [#size] [#loop] 2 > /tmp/nccl-emulator/log_debug$(date "+%m-%d-%H:%M:%S")"
```

<!-- mpirun -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG_FILE -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H mcnode02:1,mcnode06:1  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include ens3f1 sh -c "/mnt/scratch/liub0a/nccl-emulator/ex1 100000 10" -->

 <!-- mpirun -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG_FILE -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H mcnode39:1,mcnode40:1  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include enp1s0f0  sh -c "./build/ex2 1000 1 2" -->


# Pytorch

After testing the nccl, we can use the nccl in pytorch.

## Build

We have to build python from source, given that we use modified nccl.


```
bash ./scripts/build_pytorch.sh
```

## All Reduce Test

```
export WORLD_SIZE=2
export RANK=0
export LOCAL_RANK=1
export MASTER_ADDR=your master ip
export MASTER_PORT=any unused port
export MOD_KERNEL_BYPASS=1
export MOD_NNODES=2
export MOD_MY_NODE=0
python ./py-examples/all_reduce.py
```