## FSDP T5
original link:https://github.com/pytorch/examples/tree/main/distributed/FSDP
To run the T5 example with FSDP for text summarization:

## Get the wikihow dataset
```bash

sh download_dataset.sh

```

## Install the requirements:
~~~
pip install -r requirements.txt
~~~


## Start the training 



```bash
# emu
conda activate emu

export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=2

export MOD_KERNEL_BYPASS=1
export OMPI_COMM_WORLD_SIZE=2
export OMPI_COMM_WORLD_RANK=0

export TRANSFORMERS_CACHE=/tmp/fsdp_transformers_cache/
export HF_HOME=/tmp/fsdp_hf_home/
export HF_DATASETS_CACHE=/tmp/fsdp_datasets_cache/

python T5_training.py
```

```bash
# ori
conda activate ori

export LOCAL_RANK=0
export RANK=1
export WORLD_SIZE=2

export TRANSFORMERS_CACHE=/tmp/fsdp_transformers_cache/
export HF_HOME=/tmp/fsdp_hf_home/
export HF_DATASETS_CACHE=/tmp/fsdp_datasets_cache/

python T5_training.py
```
