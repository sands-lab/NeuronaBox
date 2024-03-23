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
int run(int myRank, int nRanks, int localRank, int size, int loop,
        ncclComm_t &comm) {
  float *sendbuff, *recvbuff, *gathersendbuff, *gatherrecvbuff;
  cudaStream_t s1, s2;
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMallocManaged(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMallocManaged(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaMallocManaged(&gathersendbuff, size * sizeof(float)));
  CUDACHECK(cudaMallocManaged(&gatherrecvbuff, size * sizeof(float)));

  CUDACHECK(cudaStreamCreate(&s1));
  CUDACHECK(cudaStreamCreate(&s2));

  // initializing NCCL
  for (int i = 0; i < size; ++i) {
    sendbuff[0] = myRank + 1;
  }
  // communicating using NCCL
  printf("send[0] %f at rank %d\n", sendbuff[0], myRank);
  Timer timer;
  timer.begin();
  for (int i = 0; i < loop; ++i) {
    //    printf("rk%d loop %dth start\n", myRank, i);
    auto &s = s1;
    NCCLCHECK(ncclAllReduce((const void *)sendbuff, (void *)recvbuff, size,
                            ncclFloat, ncclSum, comm, s));
    NCCLCHECK(ncclAllGather(gathersendbuff, gatherrecvbuff, size, ncclFloat,
                            comm, s));

    CUDACHECK(cudaStreamSynchronize(s));

    NCCLCHECK(ncclModStreamSync(s));
    printf("rk%d loop %dth finished\n", myRank, i);
  }
  timer.end_print(loop);
  printf("recv[0] %f at rank %d\n", recvbuff[0], myRank);

  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));

  return 0;
}

extern char **environ;
int main(int argc, char *argv[]) {
  setbuf(stdout, NULL);
  // int i = 0;
  // while (environ[i]) {
  //   // if (environ[i][0] == 'N') {
  //   printf("%s\n", environ[i]);
  //   //}
  //   i++;
  // }
  assert(argc == 3);

  // for (int i = 0; i < argc; ++i) {
  //   printf("%s\n", argv[i]);
  // }
  int size = atoi(argv[1]);
  int loop = atoi(argv[2]);
  printf("size = %d, loop = %d\n", size, loop);

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
  printf("MOD_KERNEL_BYPASS = %s\n", getenv("MOD_KERNEL_BYPASS"));
  setenv("MOD_N_MPI_RANKS", std::to_string(nRanks).c_str(), 1);
  setenv("MOD_MY_MPI_RANK", std::to_string(myRank).c_str(), 1);

  // calculating localRank based on hostname which is used in selecting a GPU
  // uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  // hostHashs[myRank] = getHostHash(hostname);

  printf("myrank %d nranks %d hostname %s\n", myRank, nRanks, hostname);
  localRank = 0;

  ncclUniqueId id;

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) {
    ncclGetUniqueId(&id);
  }
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  printf("Get nccl id at rank %d \n", myRank);

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
  run(myRank, nRanks, localRank, size, loop, comm);

  ncclCommDestroy(comm);

  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}