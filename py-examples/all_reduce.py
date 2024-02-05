import os
import torch
import torch.distributed as dist

myrank = 0
nranks = 0

def init():
  global myrank, nranks
  os.environ["MOD_MY_MPI_RANK"] = str(myrank)
  os.environ["MOD_N_MPI_RANKS"] = str(nranks)
  if myrank == 0:
    os.environ["MOD_KERNEL_BYPASS"] = "1"
  
  os.environ["LOCAL_RANK"] = "0"
  os.environ["RANK"] = str(myrank)
  os.environ["WORLD_SIZE"] = str(nranks)
  
  print("I am rank {} of {}, kbypass={}".format(myrank, nranks, os.environ.get("MOD_KERNEL_BYPASS")))

def all_reduce(tensor):
  """Performs an all-reduce operation on the given tensor."""
  dist.all_reduce(tensor)

def main():
  global myrank, nranks
  argc = len(os.sys.argv)
  assert argc == 5, "Usage: {} <rank> <nranks> <loop> <count>".format(os.sys.argv[0])
  myrank = int(os.sys.argv[1])
  nranks = int(os.sys.argv[2])
  loop = int(os.sys.argv[3])
  count = int(os.sys.argv[4])
  init()
  # Initialize distributed training
  print("Rank {} Initializing".format(myrank))
  dist.init_process_group(backend='nccl')
  print("Rank {} Initialized".format(myrank))
  device = torch.device("cuda:{}".format(0))
  
  for i in range(loop):
    # Generate a random tensor
    tensor = torch.rand(count, dtype=torch.float32, device=device)
    print("before allreduce", tensor[:3])
    # Perform all-reduce
    all_reduce(tensor)
    # Print the result
    print("after allreduce", tensor[:3])

if __name__ == '__main__':
  main()
