import os
import torch
import torch.distributed as dist

LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

def all_reduce(tensor):
  """Performs an all-reduce operation on the given tensor."""
  dist.all_reduce(tensor)

def main():
  # Initialize distributed training
  print("Rank {} Initializing".format(WORLD_RANK))
  dist.init_process_group(backend='nccl')
  print("Rank {} Initialized".format(WORLD_RANK))
  device = torch.device("cuda:{}".format(LOCAL_RANK))
  # Generate a random tensor
  tensor = torch.rand(10000, dtype=torch.float32, device=device)
  print("before allreduce", tensor[:10])
  # Perform all-reduce
  all_reduce(tensor)

  # Print the result
  print("after allreduce", tensor[:10])

if __name__ == '__main__':
  main()
