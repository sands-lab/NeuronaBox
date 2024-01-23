# nccl-emulator

## Build

First, build the nccl and nccl-tests, make sure you have conda environment properly configurated.

```
bash ./build_all.sh
```

Have a config.sh that sets up the required environments:

```bash
# config.sh
export NCCL_DEBUG=MOD
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=your_if_name
export NCCL_DEBUG_FILE=your_debug_file.$(date "+%Y-%m-%d %H:%M:%S")_%h:%p%h:%p
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring
export NCCL_BUILD_PATH=your_nccl_build_path
export NVCC_GENCODE="-gencode=arch=compute_[your_compute],code=sm_[your_sm]"
export ONLY_FUNCS="AllReduce Sum (f16|f32) RING SIMPLE"
```

## Run

To run a single node nccl-tests, use the following command:

```
mkdir log
. ./config.sh
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1 -t 2
# g for gpu per thread
# t for #thread
```

To run multi node job using mpirun, use:

### examples

Simple example: 
#### compile
```
nvcc -lmpi -lnccl -lcudart ex1.cu -o ex1
```

#### run 

run with debug

```
. ./config.sh
mpirun -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG_FILE -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H [node1]:1,[node2]:1  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include ens3f1 sh -c "./build/ex1 [#size] [#loop] > /tmp/nccl-emulator/log_debug$(date "+%m-%d-%H:%M:%S")"
```

run release
```
mpirun -x NCCL_DEBUG=VERSION -x NCCL_DEBUG_SUBSYS=INIT -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG_FILE -x NCCL_ALGO -x NCCL_TREE_THRESHOLD --prefix $CONDA_PREFIX -np 2 -H [node1]:1,[node2]:1  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include ens3f1 sh -c "./build/ex1 [#size] [#loop] > /tmp/nccl-emulator/log_release$(date "+%m-%d-%H:%M:%S")"
```

<!-- mpirun -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG_FILE -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H mcnode02:1,mcnode06:1  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include ens3f1 sh -c "/mnt/scratch/liub0a/nccl-emulator/ex1 100000 10" -->

 <!-- mpirun -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG_FILE -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H mcnode39:1,mcnode40:1  --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include enp1s0f0  sh -c "./build/ex2 1000 1 2" -->