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
        assert(output_size == (10,))
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
class Dataset1(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(256), torch.rand(256), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]  