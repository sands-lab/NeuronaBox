#include "assert.h"
#include "cuda_runtime.h"
#include "mpi.h"
#include "nccl.h"
#include <cstdlib>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <vector>
using namespace std;

#define MPICHECK(cmd)                                                          \
  do {                                                                         \
    int e = cmd;                                                               \
    if (e != MPI_SUCCESS) {                                                    \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,            \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static void getHostName(char *hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

ncclComm_t comms[16];

int run(int nRanks, vector<int> myRanks, int size, int loop, int mpirank) {
  int nDev = myRanks.size();

  float **sendbuff = (float **)malloc(nDev * sizeof(float *));
  float **recvbuff = (float **)malloc(nDev * sizeof(float *));
  cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);

  // picking GPUs based on localRank
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMallocManaged(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMallocManaged(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s + i));
  }

  for (int i = 0; i < nDev; ++i) {

    for (int j = 0; j < size; ++j) {
      sendbuff[i][j] = (float)(myRanks[i] + 1);
    }
    // communicating using NCCL
    printf("send[0] %f at rank %d node %d\n", sendbuff[i][0], myRanks[i],
           mpirank);
  }

  for (int l = 0; l < loop; ++l) {
    // calling NCCL communication API. Group API is required when using
    // multiple devices per thread/process
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; i++)
      NCCLCHECK(ncclAllReduce((const void *)sendbuff[i], (void *)recvbuff[i],
                              size, ncclFloat, ncclSum, comms[i], s[i]));
    NCCLCHECK(ncclGroupEnd());

    // synchronizing on CUDA stream to complete NCCL communication
    for (int i = 0; i < nDev; i++) {
      CUDACHECK(cudaStreamSynchronize(s[i]));
    }
  }

  for (int i = 0; i < nDev; ++i) {
    printf("recv[0] %f at rank %d node %d\n", recvbuff[i][0], myRanks[i],
           mpirank);
  }

  // freeing device memory
  for (int i = 0; i < nDev; i++) {
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }

  return 0;
}

int main(int argc, char *argv[]) {
  assert(argc == 4);

  int size = atoi(argv[1]);
  int loop = atoi(argv[2]);
  int ndev_per_node = atoi(argv[3]);

  printf("size %d loop %d ndev_per_node %d\n", size, loop, ndev_per_node);

  int myMPIRank, nMPIRanks;

  // initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myMPIRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nMPIRanks));

  if (myMPIRank == 0) {
    setenv("NCCL_KERNEL_BYPASS", "1", 1);
  } else {
    setenv("NCCL_KERNEL_BYPASS", "0", 1);
  }
  setenv("MY_MPI_RANK", std::to_string(myMPIRank).c_str(), 1);
  setenv("N_MPI_RANKS", std::to_string(nMPIRanks).c_str(), 1);

  printf("NCCL_KERNEL_BYPASS = %s\n", getenv("NCCL_KERNEL_BYPASS"));

  int totalDev = nMPIRanks * ndev_per_node; // = nRanks
  int nRanks = totalDev;
  // calculating localRank which is used in selecting a GPU
  char hostname[1024];
  getHostName(hostname, 1024);
  printf("[MPI]: myMPIrank %d nMPIranks %d hostname %s\n", myMPIRank, nMPIRanks,
         hostname);

  int nDev = ndev_per_node;

  ncclUniqueId id;

  // generating NCCL unique ID at one process and broadcasting it to all
  if (myMPIRank == 0) {
    ncclGetUniqueId(&id);
  }
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // initializing NCCL, group API is required around ncclCommInitRank as it is
  // called across multiple GPUs in each thread/process
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; i++) {
    CUDACHECK(cudaSetDevice(i));
    NCCLCHECK(ncclCommInitRank(comms + i, nRanks, id, myMPIRank * nDev + i));
  }
  NCCLCHECK(ncclGroupEnd());

  vector<int> myRanks;
  for (int i = 0; i < nDev; i++) {
    myRanks.push_back(myMPIRank * nDev + i);
  }

  run(nRanks, myRanks, size, loop, myMPIRank);

  // finalizing NCCL
  for (int i = 0; i < nDev; i++) {
    ncclCommDestroy(comms[i]);
  }

  // finalizing MPI
  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success \n", myMPIRank);
  return 0;
}