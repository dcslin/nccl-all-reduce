gcc -o test_ranks test_ranks.c \
    -L/home/shicong/mpich-3.3/build/lib/ \
    -L/usr/local/cuda-10.0/lib64/ \
    -I/usr/local/cuda-10.0/include/ \
    -I/home/shicong/mpich-3.3/build/include/ \
    -lcuda -lcudart -lnccl -lmpi \
    && mpiexec --hostfile host_file_ip ./test_ranks

gcc -o origin origin.c \
    -L/home/shicong/mpich-3.3/build/lib/ \
    -L/usr/local/cuda-10.0/lib64/ \
    -I/usr/local/cuda-10.0/include/ \
    -I/home/shicong/mpich-3.3/build/include/ \
    -lcuda -lcudart -lnccl -lmpi \
    && mpiexec --hostfile host_file_ip ./origin

gcc -o wrapper wrapper.c \
    -L/home/shicong/mpich-3.3/build/lib/ \
    -L/usr/local/cuda-10.0/lib64/ \
    -I/usr/local/cuda-10.0/include/ \
    -I/home/shicong/mpich-3.3/build/include/ \
    -lcuda -lcudart -lnccl -lmpi \
    && mpiexec --hostfile host_file_ip ./wrapper
