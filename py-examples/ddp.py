import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        return x

def example():
    local_rank = 0
    # create default process group
    # create local model
    model = Net()
    model.to(local_rank)
    
    # construct DDP model
    ddp_model = DDP(model, device_ids=[local_rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(local_rank))
    print("Forward pass successful")
    labels = torch.randn(20, 10).to(local_rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    print("Backward pass successful")
    # update parameters
    optimizer.step()

if __name__ == "__main__":
    argc = len(os.sys.argv)
    assert argc == 3, "Usage: {} <rank> <nranks>".format(os.sys.argv[0])
    myrank = int(os.sys.argv[1])
    nranks = int(os.sys.argv[2])
    os.environ["MOD_MY_MPI_RANK"] = str(myrank)
    os.environ["MOD_N_MPI_RANKS"] = str(nranks)
    # if myrank == 0:
    #     os.environ["MOD_KERNEL_BYPASS"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = str(myrank)
    os.environ["WORLD_SIZE"] = str(nranks)
    
    print("Rank ", myrank, " Initializing")
    dist.init_process_group("nccl")
    print("Rank ", myrank, " Initialized")
    
    example()