# BERT
We fine-tune a pretrained BERT model on [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) dataset. Our script is based on [this nvidia repo](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT). The dataset is already in the dataset folder. The checkpoint needs to be placed in `./dataset/checkpoint`, which can be downloaded from [NGC](https://ngc.nvidia.com/catalog/models/nvidia:bert_pyt_ckpt_large_qa_squad11_amp/files).

## Requirements
Aside from PyTorch with OmniReduce, ensure you have `tqdm`, `dllogger` and `apex`.

**Install Dependencies** :
    pip3 install --upgrade pip
    pip3 install packaging
    pip install six
    pip install tqdm
    pip install nvidia-pyindex
    pip install nvidia-dllogger

    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

**Dowload model checkpoint** :

    cd ./dataset/checkpoint
    wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_large_qa_squad11_amp/versions/19.09.0/zip -O bert_pyt_ckpt_large_qa_squad11_amp_19.09.0.zip
    unzip bert_pyt_ckpt_large_qa_squad11_amp_19.09.0.zip
    cd ../../ && mkdir -p results

## BERT Training

###  Run workers
Worker 0:
    . ../../config.sh
    CUDA_VISIBLE_DEVICES=0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=0 OMPI_COMM_WORLD_LOCAL_RANK=0 ./run.sh --init tcp://ip:port --backend nccl 

Worker 1:
    . ../../config.sh
    CUDA_VISIBLE_DEVICES=0 GLOO_SOCKET_IFNAME=eth0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=1 OMPI_COMM_WORLD_LOCAL_RANK=0 ./run.sh --init tcp://ip:port --backend nccl
