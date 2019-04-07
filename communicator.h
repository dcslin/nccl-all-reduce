#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "nccl.h"

#include "mpi.h"
#include <unistd.h>
#include <stdint.h>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

struct AllReduceHandler {
  int MPIRankInGlobal;
  int totalMPIRanksInGlobal;
  int MPIRankInLocal;
  cudaStream_t* s;
  ncclComm_t* comms;
};


 void allReduce(
	int nDev,
	int size,
	float** sendbuff,
	float** recvbuff,
	struct AllReduceHandler * h
	)
{
  //printf("perform ncclAllReduce \n");
  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread/process
  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<nDev; i++)
     NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i],
							 size, ncclFloat, ncclSum, h->comms[i], h->s[i]));
  NCCLCHECK(ncclGroupEnd());
}

void wait(int nDev, struct AllReduceHandler * h){
  //synchronizing on CUDA stream to complete NCCL communication
  for (int i=0; i<nDev; i++)
      CUDACHECK(cudaStreamSynchronize(h->s[i]));
}

void AllReduceInit(struct AllReduceHandler * h, int nDev) {
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &h->MPIRankInGlobal));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &h->totalMPIRanksInGlobal));

  //calculating MPIRankInLocal which is used in selecting a GPU
  h->MPIRankInLocal=0;
  uint64_t hostHashs[h->totalMPIRanksInGlobal];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[h->MPIRankInGlobal] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
			 sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<h->totalMPIRanksInGlobal; p++) {
     if (p == h->MPIRankInGlobal) break;
     if (hostHashs[p] == hostHashs[h->MPIRankInGlobal]) h->MPIRankInLocal++;
  }


  //picking GPUs based on MPIRankInLocal
  h->s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(h->MPIRankInLocal*nDev + i));
    CUDACHECK(cudaStreamCreate(h->s+i));
  }


  // NCCL
  ncclUniqueId id;
  h->comms=(ncclComm_t*)malloc(sizeof(ncclComm_t)*nDev);


  //printf("MPI bcast nccl unique id %d\n",id);
  //generating NCCL unique ID at one process and broadcasting it to all
  if (h->MPIRankInGlobal == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


  //initializing NCCL, group API is required around ncclCommInitRank as it is
  //called across multiple GPUs in each thread/process
  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<nDev; i++) {
    CUDACHECK(cudaSetDevice(h->MPIRankInLocal*nDev + i));
    NCCLCHECK(ncclCommInitRank(h->comms+i, h->totalMPIRanksInGlobal*nDev, id, 
							   h->MPIRankInGlobal*nDev + i));
    //printf("cuda set device %d\n",(h->MPIRankInLocal*nDev + i));
  }
  //printf("nccl set device, init rank done\n");
  NCCLCHECK(ncclGroupEnd());
}
