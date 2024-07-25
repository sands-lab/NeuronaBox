## FSDP T5
To run the T5 example with FSDP for text summarization:

## Original Link
For more details, visit the original link: https://github.com/pytorch/examples/tree/main/distributed/FSDP


## Get the wikihow dataset
```bash

sh download_dataset.sh

```

## Install the requirements:
~~~
pip install -r requirements.txt
~~~


## Start the training 

Configuration for Node 1 (emu)

```bash
# emu
conda activate emu
. ../../config.sh

export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=2

export MOD_KERNEL_BYPASS=1
export OMPI_COMM_WORLD_SIZE=2
export OMPI_COMM_WORLD_RANK=0

# a local file system like /tmp/*** is recommended
export TRANSFORMERS_CACHE=/tmp/fsdp_transformers_cache/
export HF_HOME=/tmp/fsdp_hf_home/
export HF_DATASETS_CACHE=/tmp/fsdp_datasets_cache/# a local file system like /tmp is recommended

python T5_training.py
```

Configuration for Node 2 (ori)

```bash
# ori
conda activate ori
. ../../config.sh

export LOCAL_RANK=0
export RANK=1
export WORLD_SIZE=2

# a local file system like /tmp/*** is recommended
export TRANSFORMERS_CACHE=/tmp/fsdp_transformers_cache/
export HF_HOME=/tmp/fsdp_hf_home/
export HF_DATASETS_CACHE=/tmp/fsdp_datasets_cache/

python T5_training.py
```
