gcc -o communicator_test communicator_test.cc \
    -L/home/shicong/mpich-3.3/build/lib/ \
    -L/usr/local/cuda-10.0/lib64/ \
    -I/usr/local/cuda-10.0/include/ \
    -I/home/shicong/mpich-3.3/build/include/ \
    -lcuda -lcudart -lnccl -lmpi \
    && mpiexec --hostfile host_file_ip ./communicator_test
