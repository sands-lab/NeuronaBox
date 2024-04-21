# DeepLight
DeepLight is a sparse DeepFwFM which is a click-through rate (CTR) prediction model. We modify [this repo](https://github.com/WayneDW/DeepLight_Deep-Lightweight-Feature-Interactions) script to support distributed data parallelism by using PyTorch DistributedDataParallel (DDP) package. The training dataset we use is [Criteo’s 1TB Click Prediction Dataset](https://docs.microsoft.com/en-us/archive/blogs/machinelearning/now-available-on-azure-ml-criteos-1tb-click-prediction-dataset). The folder dataset has a tiny dataset ((several batches)) for testing.

## Requirements

**Install Dependencies** :

```
pip install -U scikit-learn
```

## DeepLight Training

Worker 0:
```bash
CUDA_VISIBLE_DEVICES=0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=0 OMPI_COMM_WORLD_LOCAL_RANK=0 MOD_KERNEL_BYPASS=1 python main_all.py -l2 6e-7 -n_epochs 2 -warm 2 -prune 1 -sparse 0.90  -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1. -backend nccl -batch_size 2048  -init tcp://${MASTER_ADDR}:${MASTER_PORT}
```
Worker 1:
```bash
CUDA_VISIBLE_DEVICES=0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=1 OMPI_COMM_WORLD_LOCAL_RANK=0 python main_all.py -l2 6e-7 -n_epochs 2 -warm 2 -prune 1 -sparse 0.90  -prune_deep 1 -prune_fm 1 -prune_r 1 -use_fwlw 1 -emb_r 0.444 -emb_corr 1. -backend nccl -batch_size 2048  -init tcp://${MASTER_ADDR}:${MASTER_PORT}
```