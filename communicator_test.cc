#include "communicator.h"

int main(int argc, char* argv[])
{
  // init MPI env
  MPICHECK(MPI_Init(&argc, &argv));

  // N Dev per Rank
  int nDev=1;
  //int size = 32*1024*1024;
  int size = 2;

  struct AllReduceHandler *allReduceHandler=malloc(sizeof(*allReduceHandler));

  // wrapper function
  AllReduceInit(allReduceHandler, nDev);

  //printf("global rank      %d\n", allReduceHandler->MPIRankInGlobal);
  //printf("# of global rank %d\n", allReduceHandler->totalMPIRanksInGlobal);
  //printf("local rank       %d\n", allReduceHandler->MPIRankInLocal);


  // alloc mem assume this is done before hand
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));

  for (int i=0; i<nDev; i++) {
    CUDACHECK(cudaSetDevice(allReduceHandler->MPIRankInLocal*nDev + i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
  }


  // wrapper function
  allReduce(nDev, size, sendbuff, recvbuff, allReduceHandler);
  wait(nDev, allReduceHandler);


  // clean up
  for (int i=0; i<nDev; i++) {
     CUDACHECK(cudaFree(sendbuff[i]));
     CUDACHECK(cudaFree(recvbuff[i]));
  }
  free(allReduceHandler->s);
  free(allReduceHandler->comms);
  free(allReduceHandler);


  //finalizing MPI
  MPICHECK(MPI_Finalize());
  return 0;
}

