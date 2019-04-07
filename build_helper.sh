BINARY_NAME=test_comm
g++ -g -Wall -std=c++11 \
    -I/usr/local/cuda-10.0/include/ \
    -I/home/shicong/mpich-3.3/build/include/ \
    -L/home/shicong/mpich-3.3/build/lib/ \
    -L/usr/local/cuda-10.0/lib64/ \
    test_comm.cc communicator.cc  \
    -lmpi -lmpicxx -lcuda -lcudart -lnccl \
    -o $BINARY_NAME \
&& mpiexec --hostfile host_file_ip ./$BINARY_NAME
