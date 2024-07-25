export TRANSFORMERS_CACHE=/tmp/yh_fsdp_transformers_cache/
export HF_HOME=/tmp/yh_fsdp_model/
export DATASETS_CACHE=/tmp/yh_fsdp_datasets_cache/
export HF_DATASETS_CACHE=/tmp/yh_fsdp_datasets_cache/
export NCCL_SOCKET_IFNAME=enp1s0f1



MOD_N_MPI_RANKS/OMPI_COMM_WORLD_SIZE
OMPI_COMM_WORLD_RANK/MOD_MY_MPI_RANK
CUDA_VISIBLE_DEVICES=0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=0 OMPI_COMM_WORLD_LOCAL_RANK=0 MOD_KERNEL_BYPASS=1 ./run.sh 

. ../../config.sh
CUDA_VISIBLE_DEVICES=0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=0 OMPI_COMM_WORLD_LOCAL_RANK=0 ./run.sh
OMPI_COMM_WORLD_RANK=0 MOD_KERNEL_BYPASS=1 ./run.sh 

CUDA_VISIBLE_DEVICES=0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=1 OMPI_COMM_WORLD_LOCAL_RANK=0 MOD_KERNEL_BYPASS=0 ./run.sh 


. ../../config.sh
CUDA_VISIBLE_DEVICES=0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=1 OMPI_COMM_WORLD_LOCAL_RANK=0 ./run.sh
cat '/home/x_dingy/debug_file/your_debug_file.2024-07-25 04:22:16_mcnode39:4100386mcnode39:4100386' | grep -i -E 'allgather|reducescatter|broad|cast'

#node39:
export LOCAL_RANK=0
export RANK=0
export LOCAL_WORLD_SIZE=1 #--nproc-per-node
export WORLD_SIZE=2
export MASTER_ADDR=172.18.0.50
export MASTER_PORT=29500

export MOD_KERNEL_BYPASS=1
export OMPI_COMM_WORLD_SIZE=2
export OMPI_COMM_WORLD_RANK=0

#node38:
export LOCAL_RANK=0
export RANK=1
export LOCAL_WORLD_SIZE=1 #--nproc-per-node
export WORLD_SIZE=2
export MASTER_ADDR=172.18.0.50
export MASTER_PORT=29500



2-2:94 [17:01<00:00, 10.86s/it]
Train Epoch:    1, Loss:        0.3653
Validation Epoch: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19/19 [00:17<00:00,  1.07it/s]
Validation Loss: 0.3190
--> epoch 1 completed...entering save and stats zone
-->>>> New Val Loss Record: 0.3190145790576935
r0 Training Epoch: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 94/94 [08:02<00:00,  5.14s/it]
Train Epoch:    2, Loss:        0.2556
Validation Epoch: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19/19 [00:17<00:00,  1.07it/s]
Validation Loss: 0.3112
--> epoch 2 completed...entering save and stats zone
-->>>> New Val Loss Record: 0.31121280789375305


2-1: