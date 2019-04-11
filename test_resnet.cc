#include "communicator.h"
#include <iomanip>
#include <chrono>

using namespace std;


struct BuffPair {
  /*
   * send buffer
   * receive buffer
   */
  float** sb;
  float** rb;   
  int nDev;
  BuffPair(int size, Communicator* c): nDev(c->nDev){
    sb = (float**)malloc(nDev * sizeof(float*));
    rb = (float**)malloc(nDev * sizeof(float*));

    for (int i=0; i<nDev; i++) {
      CUDACHECK(cudaSetDevice(c->MPIRankInLocal*nDev + i));
      CUDACHECK(cudaMalloc(sb + i, size * sizeof(float)));
      CUDACHECK(cudaMalloc(rb + i, size * sizeof(float)));
      CUDACHECK(cudaMemset(sb[i], 1, size * sizeof(float)));
      CUDACHECK(cudaMemset(rb[i], 0, size * sizeof(float)));
    }
  }
  ~BuffPair() {
    for (int i=0; i<nDev; i++) {
      CUDACHECK(cudaFree(sb[i]));
      CUDACHECK(cudaFree(rb[i]));
    }
    free(sb);
    free(rb);
  }
};


int main(int argc, char *argv[])
{
  // init MPI env
  MPICHECK(MPI_Init(&argc, &argv));

  int nDev = atoi(argv[1]);
  int repeats = 10;

  // list harvested from ResNet20v1,
  // by keras implemenation and printed by model.summary()
  /*
  int totalLayers = 41;
  int paramCounts[41] = {448, 64, 2320, 64, 2320,
      64, 2320, 64, 2320, 64,
      2320, 64, 2320, 64, 4640,
      128, 9248, 544, 128, 9248,
      128, 9248, 128, 9248, 128,
      9248, 128, 18496, 256, 36928,
      2112, 256, 36928, 256, 36928,
      256, 36928, 256, 36928, 256,
      650};
      */
  int totalLayers = 2;
  int paramCounts[2] = {2,2};


  Communicator c(nDev);


  BuffPair** layerBuffPairs = new BuffPair*[totalLayers];
  for (int l=0;l<totalLayers;l++) {
    layerBuffPairs[l] = new BuffPair(paramCounts[l], &c);
  }

  auto start = chrono::high_resolution_clock::now();
  // unsync the I/O of C and C++.
  ios_base::sync_with_stdio(false);


  // main process of all reduce
  for(int i=0; i<repeats; i++){
    for(int l=0; l<totalLayers; l++){
      c.allReduce(paramCounts[l], layerBuffPairs[l]->sb, layerBuffPairs[l]->rb);
      c.wait();
    }
  }

  auto end = chrono::high_resolution_clock::now();

  // Calculating total time taken by the program.
  double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

  time_taken *= 1e-9;

  if (c.MPIRankInGlobal == 0){
    cout << "nDev per thread: " << nDev;
    //cout << " - number of float: " << setw(9) << size;
    cout << " - Time: " << fixed << time_taken/repeats << setprecision(9);
    cout << " sec (avg over repeated " << repeats << " times)";
    cout << endl;
  }


  // clean up
  for (int i=0; i<nDev; i++) {
    delete[] layerBuffPairs;
  }


  //finalizing MPI
  MPICHECK(MPI_Finalize());
  return 0;
}
