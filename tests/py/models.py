#!/usr/bin/env python

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset


#! All model should output only one value

class Model0(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model0, self).__init__()
        print("Model0 input_size:", input_size)
        assert(input_size == (20,))
        assert(output_size == (1,))
        self.fc = torch.nn.Linear(20, 1)  
        
    def forward(self, x):
        return self.fc(x)
    
class Dataset0(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]   
     
class Model1(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model1, self).__init__()
        print("Model1 input_size:", input_size)
        assert(input_size == (200,))
        assert(output_size == (1,))
        self.fc1 = torch.nn.Linear(200, 2000)  
        self.fc2 = torch.nn.Linear(2000, 4000)  
        self.fc3 = torch.nn.Linear(4000, 4000)  
        self.fc4 = torch.nn.Linear(4000, 100)
        self.fc5 = torch.nn.Linear(100, 1)  
  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
class Dataset1(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(200), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]  