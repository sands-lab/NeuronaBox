# CNN
We use the training scripts provided by [Nvidia DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/use_cases/pytorch/resnet50/pytorch-resnet50.html) which implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset. Before training, you should download the [ImageNet dataset](https://www.image-net.org/). The folder dataset has a tiny dataset just for testing.

## Requirements
Aside from PyTorch with OmniReduce, ensure you have 
`Pillow`, `torchvision`, `DALI` and `apex`.

**Install Dependencies** :

```bash
pip install Pillow
pip install torchvision===0.12.0 --no-dependencies
pip install --extra-index-url https://pypi.nvidia.com  --upgrade nvidia-dali-cuda120
pip install nvidia-nvjpeg-cu12
git clone https://github.com/NVIDIA/apex
cd apex
git reset --hard a651e2c2
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## CNN Training

### ResNet152
Worker 0:

```bash
CUDA_VISIBLE_DEVICES=0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=0 OMPI_COMM_WORLD_LOCAL_RANK=0 python main.py -a resnet152 --lr 0.1 --world-size 2 --rank 0 --dist-url tcp://${MASTER_ADDR}:${MASTER_PORT} --dist-backend nccl  ./dataset/
```

Worker 1:

```bash
CUDA_VISIBLE_DEVICES=0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_RANK=1 OMPI_COMM_WORLD_LOCAL_RANK=0 python main.py -a resnet152 --lr 0.1 --world-size 2 --rank 1 --dist-url tcp://${MASTER_ADDR}:${MASTER_PORT} --dist-backend nccl ./dataset/
```

### VGG19
Worker 0:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py -a vgg19 --lr 0.01 --world-size 2 --rank 0 --dist-url tcp://${MASTER_ADDR}:${MASTER_PORT} --dist-backend nccl  ./dataset/
```

Worker 1:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py -a vgg19 --lr 0.01 --world-size 2 --rank 1 --dist-url tcp://${MASTER_ADDR}:${MASTER_PORT} --dist-backend nccl  ./dataset/
```