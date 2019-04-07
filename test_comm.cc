#include "communicator.h"

void test(){
  int size=4;
  float send[size] = {1.0,2.0,1.0,2.0};
  float* send_d;

  CUDACHECK(cudaSetDevice(0));
  CUDACHECK(cudaMalloc((void **)&send_d, size*sizeof(float)));
  CUDACHECK(cudaMemcpy(send_d,send,size*sizeof(float),cudaMemcpyHostToDevice));

  float receive[size];
  CUDACHECK(cudaMemcpy(receive,send_d,size*sizeof(float),cudaMemcpyDeviceToHost));

  std::cout << "received" << receive[0] << "\n";

  cudaFree(&send_d);
}

int main(int argc, char *argv[])
{
  // init MPI env
  MPICHECK(MPI_Init(&argc, &argv));

  int nDev = 1;
  int size = 3;

  Communicator c(nDev);


  // alloc mem assume this is done before hand
  // nDev pointers to cudaMem
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));

  float send[size];
  for(int i=0; i<size; i++)
    send[i]=i+10.0;

  for (int i=0; i<nDev; i++) {
    CUDACHECK(cudaSetDevice(c.MPIRankInLocal*nDev + i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));

    CUDACHECK(cudaMemcpy(sendbuff[i],send,size*sizeof(float),cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
  }

  //test();

  // main process of all reduce
  std::cout<<"doing all reduce..\n";
  c.allReduce(size, sendbuff, recvbuff);
  std::cout<<"doing waiting..\n";
  c.wait();

  //just test for first gpu
  float receive[size];
  CUDACHECK(cudaSetDevice(0));
  CUDACHECK(cudaMemcpy(receive,recvbuff[0],size*sizeof(float),cudaMemcpyDeviceToHost));
  std::cout << "received buff: " << receive[0] << " vs 40 expected"<< "\n";
  std::cout << "received buff: " << receive[1] << " vs 44 expected"<< "\n";

  // clean up
  for (int i=0; i<nDev; i++) {
     CUDACHECK(cudaFree(sendbuff[i]));
     CUDACHECK(cudaFree(recvbuff[i]));
  }

  //finalizing MPI
  MPICHECK(MPI_Finalize());
  return 0;
}
