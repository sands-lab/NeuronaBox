#include "cuda_runtime.h"
#include "mpi.h"
#include "nccl.h"
#include "timer.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include<fstream>
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

// size is the number of elements
int run(int myRank, int nRanks, int localRank, uint64_t size, int loop,
        ncclComm_t &comm) {
  float *sendbuff, *recvbuff;
  cudaStream_t s1;
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaMemset(sendbuff, 1, size * sizeof(float)));
  CUDACHECK(cudaMemset(recvbuff, 0, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s1));

  // warmup
  for (int i = 0; i < 10; ++i) {
    auto &s = s1;
    NCCLCHECK(ncclBroadcast((const void *)sendbuff, (void *)recvbuff, size,
                            ncclFloat, 0, comm, s));
    CUDACHECK(cudaStreamSynchronize(s));
    NCCLCHECK(ncclModStreamSync(s));
  }
  Timer timer0;
  uint64_t  sum_e2e = 0;
  timer0.begin();
  for (int i = 0; i < loop; ++i) {
    auto &s = s1;
    NCCLCHECK(ncclBroadcast((const void *)sendbuff, (void *)recvbuff, size,
                            ncclFloat, 0, comm, s));

    NCCLCHECK(ncclModStreamSync(s));
  }
  CUDACHECK(cudaStreamSynchronize(s1));
  sum_e2e = timer0.end(1);

  printf("[rk%d] average time for sum_e2e: %.3f\n", myRank, 1.0*sum_e2e/loop);

  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  return 0;
}

extern char **environ;
int main(int argc, char *argv[]) {
  setbuf(stdout, NULL);

  assert(argc == 3);

  uint64_t size = atoll(argv[1]);
  int loop = atoi(argv[2]);
  printf("size = %lu, loop = %d\n", size, loop);

  int myRank, nRanks, localRank = 0;
  // initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  if (myRank == 0) {
    setenv("MOD_KERNEL_BYPASS", "1", 1);
  } else {
    setenv("MOD_KERNEL_BYPASS", "0", 1);
  }
  setenv("MOD_BYPASS_NUM", "0", 1);

  printf("MOD_KERNEL_BYPASS = %s\n", getenv("MOD_KERNEL_BYPASS"));
  setenv("MOD_N_MPI_RANKS", std::to_string(nRanks).c_str(), 1);
  setenv("MOD_MY_MPI_RANK", std::to_string(myRank).c_str(), 1);

  char hostname[1024];
  getHostName(hostname, 1024);

  printf("myrank %d nranks %d hostname %s\n", myRank, nRanks, hostname);
  localRank = 0;

  ncclUniqueId id;

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) {
    ncclGetUniqueId(&id);
  }
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
  run(myRank, nRanks, localRank, size, loop, comm);

  ncclCommDestroy(comm);

  MPICHECK(MPI_Finalize());

  return 0;
}
