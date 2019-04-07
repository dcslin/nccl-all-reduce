#include "communicator.h"

int main(int argc, char *argv[])
{
  // init MPI env
  MPICHECK(MPI_Init(&argc, &argv));

  int nDev = 1;
  int size = 3;

  Communicator c(nDev);


  // alloc mem assume this is done before hand
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));

  for (int i=0; i<nDev; i++) {
    CUDACHECK(cudaSetDevice(c.MPIRankInLocal*nDev + i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
  }

  // main process of all reduce
  std::cout<<"doning all reduce..\n";
  c.allReduce(size, sendbuff, recvbuff);
  std::cout<<"doning waiting..\n";
  c.wait();

  // clean up
  for (int i=0; i<nDev; i++) {
     CUDACHECK(cudaFree(sendbuff[i]));
     CUDACHECK(cudaFree(recvbuff[i]));
  }

  //finalizing MPI
  MPICHECK(MPI_Finalize());
  return 0;
}
