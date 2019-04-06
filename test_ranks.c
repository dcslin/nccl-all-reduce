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



void one_process_multi_device(){


    int MPIRankInGlobal, totalMPIRanksInGlobal, processRankInLocal=0;

    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &MPIRankInGlobal));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &totalMPIRanksInGlobal));


    //calculating processRankInLocal based on hostname which is used in selecting a GPU
    uint64_t hostHashs[totalMPIRanksInGlobal];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[MPIRankInGlobal] = getHostHash(hostname);
	printf("name: %s\n", hostname);
	printf("hash: %d\n", hostHashs[MPIRankInGlobal]);
	// could not run in multi nodes
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p=0; p<totalMPIRanksInGlobal; p++) {
       if (p == MPIRankInGlobal) break;
       if (hostHashs[p] == hostHashs[MPIRankInGlobal]) processRankInLocal++;
    }


    printf("MPIRankInGlobal %d - processRankInLocal %d | totalMPIRanksInGlobal %d\n", MPIRankInGlobal, processRankInLocal, totalMPIRanksInGlobal);

}



int main(int argc, char* argv[])
{
    MPICHECK(MPI_Init(&argc, &argv));
	one_process_multi_device();
    MPICHECK(MPI_Finalize());
	return 0;
}
