import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        return x

# Modified function to print grad_fn
def print_computation_graph(module, input, output):
    print(f"Inside {module.__class__.__name__}'s forward pass")
    # grad_fn for the input might not be set if the input does not require gradients
    print(f"Input grad_fn: {[i.grad_fn for i in input if i.requires_grad]}")
    print(f"Output grad_fn: {output.grad_fn}")

def example():
    model = Net()
    model.to(0)
    _ = model.fc1.register_forward_hook(print_computation_graph)

    for p in model.parameters():
        p.data.fill_(1)
    labels = torch.ones(20, 10).to(0) * 2
    inputs = torch.ones(20, 10).to(0)
    loss_fn = nn.MSELoss()
    print("Forward:")
    outputs = model(inputs)
    print("Backward:")
    loss_fn(outputs, labels).backward()
    

    # should be [1.8000....] for all parameters 
    for param in model.parameters():
        print(param.grad[:10][:2])

if __name__ == "__main__":    
    example()