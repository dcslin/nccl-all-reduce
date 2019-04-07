#include "communicator.h"

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


Communicator::Communicator(int nDev): nDev(nDev){
  // get MPI Global Ranks and total Ranks
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &MPIRankInGlobal));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &totalMPIRanksInGlobal));
  std::cout<<"g rank " << MPIRankInGlobal << "\n";

  //calculating MPIRankInLocal which is used in selecting a GPU
  MPIRankInLocal=0;
  uint64_t hostHashs[totalMPIRanksInGlobal];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[MPIRankInGlobal] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs,
    		 sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<totalMPIRanksInGlobal; p++) {
     if (p == MPIRankInGlobal) break;
     if (hostHashs[p] == hostHashs[MPIRankInGlobal]) MPIRankInLocal++;
  }

  std::cout<<"l rank " << MPIRankInLocal << "\n";

  //picking GPUs based on MPIRankInLocal
  //create cuda stream s
  s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(MPIRankInLocal*nDev + i));
    CUDACHECK(cudaStreamCreate(s+i));
  }

  // create nccl comms 
  ncclUniqueId id;
  comms=(ncclComm_t*)malloc(sizeof(ncclComm_t)*nDev);
  

  //generating NCCL unique nccl ID at one process and broadcasting it to all
  if (MPIRankInGlobal == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


  //initializing NCCL, group API is required around ncclCommInitRank as it is
  //called across multiple GPUs in each thread/process
  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<nDev; i++) {
    CUDACHECK(cudaSetDevice(MPIRankInLocal*nDev + i));
    NCCLCHECK(ncclCommInitRank(comms+i,
                               totalMPIRanksInGlobal*nDev,
                               id, 
    						     MPIRankInGlobal*nDev + i));
  }
  NCCLCHECK(ncclGroupEnd());
} // end of constructor 


void Communicator::allReduce(int size, float** sendbuff, float** recvbuff)
{
  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread/process
  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<nDev; i++)
     NCCLCHECK(ncclAllReduce((const void*)sendbuff[i],
                             (void*)recvbuff[i],
    						   size,
                             ncclFloat,
                             ncclSum,
                             comms[i],
                             s[i]));
  NCCLCHECK(ncclGroupEnd());
}

void Communicator::wait(){
  //synchronizing on CUDA stream to complete NCCL communication
  for (int i=0; i<nDev; i++)
    CUDACHECK(cudaStreamSynchronize(s[i]));
}

Communicator::~Communicator(){
  free(s);
  free(comms);
}
