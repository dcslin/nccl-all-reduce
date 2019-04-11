#include "communicator.h"
#include <iomanip>
#include <chrono>
#include <vector>

using namespace std;


struct BuffPair {
  /*
   * send buffer: size is n_gpu
   * receive buffer: size is n_gpu
   */
  float** sb;
  float** rb;   
  int nDev;
  BuffPair(int size, Communicator* c): nDev(c->nDev){
    sb = (float**)malloc(nDev*sizeof(float*));
    rb = (float**)malloc(nDev*sizeof(float*));

    for (int i=0; i<nDev; i++) {
      CUDACHECK(cudaSetDevice(c->MPIRankInLocal*nDev + i));
      CUDACHECK(cudaMalloc(sb+i, size * sizeof(float)));
      CUDACHECK(cudaMalloc(rb+i, size * sizeof(float)));
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


void runResNet(int argc, char *argv[], int threshold){

  int nDev = atoi(argv[1]);
  int repeats = 10;

  /*
   * list harvested from ResNet56v1,
   * by keras implemenation and printed by model.summary()
   */

  int totalLayers = 113;
  int paramCounts[113] = {448, 64,
      2320, 64, 2320, 64, 2320, 64,
      2320, 64, 2320, 64, 2320, 64,
      2320, 64, 2320, 64, 2320, 64,
      2320, 64, 2320, 64, 2320, 64,
      2320, 64, 2320, 64, 2320, 64,
      2320, 64, 2320, 64, 2320, 64, 4640,
      128, 9248, 544,
      128, 9248,
      128, 9248,
      128, 9248,
      128, 9248,
      128, 9248,
      128, 9248,
      128, 9248,
      128, 9248,
      128, 9248,
      128, 9248,
      128, 9248,
      128, 9248,
      128, 9248,
      128, 9248,
      128, 9248,
      128, 9248,
      128, 18496,
      256, 36928, 2112,
      256, 36928,
      256, 36928,
      256, 36928,
      256, 36928,
      256, 36928,
      256, 36928,
      256, 36928,
      256, 36928,
      256, 36928,
      256, 36928,
      256, 36928,
      256, 36928,
      256, 36928,
      256, 36928,
      256, 36928,
      256, 36928,
      256, 650};


  // do fusion
  vector<int> fusionParamCounts;
  int accumulator=0;
  for (int l=0; l<totalLayers; l++) {
    accumulator=accumulator+paramCounts[l];
    // we push accumulator to fusion list when it is sufficiently large
    if( accumulator*sizeof(float) > threshold*1024 ) {
      fusionParamCounts.push_back(accumulator);
      // reset accumulator for next batch
      accumulator=0;
    }
  }
  // if die die can not fusion even for one batch, sum up all params
  if(fusionParamCounts.size() == 0) {
    fusionParamCounts.push_back(accumulator);
  }


  Communicator c(nDev);


  // init buff pair for all the fusion batches
  BuffPair** bps = new BuffPair*[fusionParamCounts.size()];
  for (int l=0; l<fusionParamCounts.size(); l++) {
    bps[l] = new BuffPair(fusionParamCounts[l], &c);
  }


  auto start = chrono::high_resolution_clock::now();
  // unsync the I/O of C and C++.
  ios_base::sync_with_stdio(false);


  // main process of all reduce
  for(int i=0; i<repeats; i++){
    for(int l=0; l<fusionParamCounts.size(); l++){
      c.allReduce(fusionParamCounts[l], bps[l]->sb, bps[l]->rb);
      // blocking execution, run wait following allReduce
      c.wait();
    }
  }

  auto end = chrono::high_resolution_clock::now();

  // Calculating total time taken by the program.
  double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();

  time_taken *= 1e-9;

  // print report
  if (c.MPIRankInGlobal == 0){
    cout << "Fusion Threshold is " << threshold << " KB - Total Batches: "
        << fusionParamCounts.size()
        << "\nBatches of params after fusion: "
        << fixed << setprecision(1);
    for(int l=0; l<fusionParamCounts.size(); l++){
      cout  << (fusionParamCounts[l]*4.0)/1024.0 << "KB, ";
    }
    cout << "\n\n";
    cout << "*****gpu per thread: " << nDev;
    cout << " - Time: "<< fixed<< setprecision(9)  << time_taken/repeats ;
    cout << " sec (avg over repeated " << repeats << " times)*****\n";
    cout << "=======================================================";
    cout << endl;
  }


  // clean up
  for(int l=0; l<fusionParamCounts.size(); l++){
    delete(bps[l]);
  }
  delete(bps);
}

int main(int argc, char *argv[])
{
  // init MPI env
  MPICHECK(MPI_Init(&argc, &argv));

  // last param is fusion threshold in KB
  runResNet(argc, argv, 1);
  runResNet(argc, argv, 10);
  runResNet(argc, argv, 100);
  runResNet(argc, argv, 1000);

  //finalizing MPI
  MPICHECK(MPI_Finalize());
  return 0;
}
