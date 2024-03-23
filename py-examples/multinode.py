import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import torch
import time

from models import Model0, Model1, Dataset0, Dataset1

ITER = 16
BATCH_SIZE = 64
DATA_SIZE = ITER * BATCH_SIZE
    
def ddp_setup():
    print("Initializing process group for rank", os.environ["RANK"], "and world size", os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")
    print("Process group initialized")

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _run_batch(self, source, targets, index):
        self.optimizer.zero_grad()
        print(f"[pytorch:{self.global_rank}] forward pass")
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        print(f"[pytorch:{self.global_rank}] backward pass")
        loss.backward()
        self.optimizer.step()
        print(f"[pytorch:{self.global_rank}] {index}th batch done")

    def _run_epoch(self, epoch):
        print(f"[pytorch:{self.global_rank}] epoch {epoch}")
        self.train_data.sampler.set_epoch(epoch)
        index = 0
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets, index)
            index += 1
 
    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)


def load_train_objs():
    global DATA_SIZE
    # train_set = Dataset0(DATA_SIZE) 
    # model = Model0((20,),(1,)) 
    train_set = Dataset1(DATA_SIZE) 
    model = Model1((200,),(1,))   
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, 0, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    assert(len(os.sys.argv) == 3)
    myrank = int(os.sys.argv[1])
    nranks = int(os.sys.argv[2])
    os.environ["MOD_MY_MPI_RANK"] = str(myrank)
    os.environ["MOD_N_MPI_RANKS"] = str(nranks)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = str(myrank)
    os.environ["WORLD_SIZE"] = str(nranks)

    if myrank == 0:
        os.environ["MOD_KERNEL_BYPASS"] = "1"
    
    # total epochs, batch size
    main(ITER, BATCH_SIZE)