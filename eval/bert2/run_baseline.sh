#!/bin/bash
#SBATCH -J MyJob
#SBATCH -o result.out
#SBATCH -e error.err


#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --partition=batch
#SBATCH --gpus-per-task=8
#SBATCH --mem=512G
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --constraint=v100
#SBATCH --mail-type=ALL
#SBATCH --time=00:05:00


source conda activate SNIC

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

torchrun \
--nnodes=2 \
--nproc_per_node=1 \
--node_rank=0 \
--rdzv_id=123 \
--rdzv_backend=c10d \
--rdzv_endpoint=mcnode39:3335 \
/home/liub0a/emulator/eval/bert2/baseline.py