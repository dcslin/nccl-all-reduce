# c wrapper around nccl all reduce api

# dependencies
- nccl 2.4.2
- cuda 10.0
- mpich 3.3

# compile and run
gcc -o wrapper wrapper.c \
    -L/home/shicong/mpich-3.3/build/lib/ \
    -L/usr/local/cuda-10.0/lib64/ \
    -I/usr/local/cuda-10.0/include/ \
    -I/home/shicong/mpich-3.3/build/include/ \
    -lcuda -lcudart -lnccl -lmpi \
    && mpiexec --hostfile host_file_ip ./wrapper

## notes:
- need to specify host by ip, otherwise it will failed.
