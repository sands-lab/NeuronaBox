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

```
. ./config.sh
mpirun -x NCCL_DEBUG_FILE -x NCCL_DEBUG -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG_SUBSYS --prefix $CONDA_PREFIX -np 2 -H [nodename1]:1,[nodename2]:1 --mca pml ob1 --mca btl tcp,self ./nccl-tests/build/all_reduce_perf -g 1 -f 2 -b 1K -e 8G
```