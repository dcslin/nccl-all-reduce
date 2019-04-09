#include "communicator.h"
#include <iomanip>
#include <chrono>

using namespace std;

int main(int argc, char *argv[])
{
  // init MPI env
  MPICHECK(MPI_Init(&argc, &argv));

  int nDev = 2;
  int size = 100000;
  int repeats = 1;

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

  auto start = chrono::high_resolution_clock::now();
  // unsync the I/O of C and C++.
  ios_base::sync_with_stdio(false);


  // main process of all reduce
  for(int i=0; i<repeats; i++){
    //std::cout<<"doing all reduce..\n";
    c.allReduce(size, sendbuff, recvbuff);
    //std::cout<<"doing waiting..\n";
    c.wait();
  }

  auto end = chrono::high_resolution_clock::now();
  // Calculating total time taken by the program.
  double time_taken =
      chrono::duration_cast<chrono::nanoseconds>(end - start).count();

  time_taken *= 1e-9;

  cout << "Time taken by program is : " << fixed << time_taken << setprecision(9);
  cout << " sec" << endl;


  //just test for first gpu
  float receive[size];
  CUDACHECK(cudaSetDevice(0));
  CUDACHECK(cudaMemcpy(receive,recvbuff[0],size*sizeof(float),cudaMemcpyDeviceToHost));
  std::cout << "Before: " << send[0]    << ", "<< send[1]    << ",...\n";
  std::cout << "After:  " << receive[0] << ", "<< receive[1] << ",...\n";


  // clean up
  for (int i=0; i<nDev; i++) {
     CUDACHECK(cudaFree(sendbuff[i]));
     CUDACHECK(cudaFree(recvbuff[i]));
  }

  //finalizing MPI
  MPICHECK(MPI_Finalize());
  return 0;
}
