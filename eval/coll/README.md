## Collective Communication Evaluation

We need two terminal to run this test, one is using `emu`, the other using `ori`.

The bash script assume the `pwd` is the root of the project.

### All Reduce

```bash
# emu
conda activate emu
. ./config_release.sh # use release build for nccl and pytorch
nvcc -lmpi -lnccl -lcudart -O3 ./eval/coll/all_reduce.cu -o ./build/all_reduce
cp ./build/all_reduce /tmp/ex
```

```bash
# ori
conda activate ori
. ./config_release.sh # use release build for nccl and pytorch
nvcc -lmpi -lnccl -lcudart -O3 ./eval/coll/all_reduce_ori.cu -o ./build/all_reduce_ori
cp ./build/all_reduce_ori /tmp/ex

mpirun -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H [node1]:1,[node2]:1 --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include enp1s0f0 sh -c "/tmp/ex [#size] [#loop]"
```

The results will be output to the terminal.

### All Gather

```bash
# emu
conda activate emu
. ./config_release.sh # use release build for nccl and pytorch
nvcc -lmpi -lnccl -lcudart -O3 ./eval/coll/all_gather.cu -o ./build/all_gather
cp ./build/all_gather /tmp/ex

```

```bash
# ori
conda activate ori
. ./config_release.sh # use release build for nccl and pytorch
nvcc -lmpi -lnccl -lcudart -O3 ./eval/coll/all_gather_ori.cu -o ./build/all_gather_ori
cp ./build/all_gather_ori /tmp/ex

mpirun -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H [node1]:1,[node2]:1 --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include enp1s0f0 sh -c "/tmp/ex [#size] [#loop]"
```

The results will be output to the terminal.


### Broadcast

```bash
# emu
conda activate emu
. ./config_release.sh # use release build for nccl and pytorch
nvcc -lmpi -lnccl -lcudart -O3 ./tests/nccl/broadcast.cu -o ./build/broadcast
cp ./build/broadcast /tmp/ex

```

```bash
# ori
conda activate ori
. ./config_release.sh # use release build for nccl and pytorch
nvcc -lmpi -lnccl -lcudart -O3 ./eval/coll/broadcast_ori.cu -o ./build/broadcast_ori
cp ./build/broadcast_ori /tmp/ex

mpirun -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H [node1]:1,[node2]:1 --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include enp1s0f0 sh -c "/tmp/ex [#size] [#loop]"
```

The results will be output to the terminal.


### ReduceScatter

```bash
# emu
conda activate emu
. ./config_release.sh # use release build for nccl and pytorch
nvcc -lmpi -lnccl -lcudart -O3 ./tests/nccl/reducescatter.cu -o ./build/reducescatter
cp ./build/reducescatter /tmp/ex

```

```bash
# ori
conda activate ori
. ./config_release.sh # use release build for nccl and pytorch
nvcc -lmpi -lnccl -lcudart -O3 ./eval/coll/reducescatter_ori.cu -o ./build/reducescatter_ori
cp ./build/reducescatter_ori /tmp/ex

mpirun -x NCCL_PROTO -x NCCL_ALGO --prefix $CONDA_PREFIX -np 2 -H [node1]:1,[node2]:1 --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include enp1s0f0 sh -c "/tmp/ex [#size] [#loop]"
```

The results will be output to the terminal.