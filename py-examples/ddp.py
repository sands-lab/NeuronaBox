import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

def example(world_rank, local_rank, world_size):
    # create default process group
    dist.init_process_group("nccl", rank=world_rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(local_rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[local_rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    print("init finish, rank = ", world_rank);
    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(local_rank))
    labels = torch.randn(20, 10).to(local_rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()
    print("Done rk=", world_rank);

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    example(WORLD_RANK, LOCAL_RANK, WORLD_SIZE)

