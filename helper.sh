rm -rf ./test
gcc -o nccl_test test.c -L/usr/local/cuda-9.0/targets/x86_64-linux/lib -I/usr/local/cuda-9.0/targets/x86_64-linux/include -lcuda -lcudart -lnccl -lmpi
#OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun  --allow-run-as-root -n 1 -hostfile host_file ./test
#OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun  --allow-run-as-root -n 2 -hostfile host_file nccl_test
#OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun  --allow-run-as-root -n 2 -hostfile host_file hostname
#OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun  --allow-run-as-root -n 1 nccl_test
#OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun  --allow-run-as-root -n 1 hostname
#OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun --allow-run-as-root --help
#OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun --allow-run-as-root -n 1 -host 10.0.0.131 uptime
#OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun --allow-run-as-root -n 1 -hostfile host_file uptime

/home/shicong/git/openmpi-4.0.1/build/bin/mpirun -hostfile host_file hostname
LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/:/home/shicong/git/openmpi-4.0.1/build/lib/ \
    /home/shicong/git/openmpi-4.0.1/build/bin/mpirun -hostfile host_file nccl_test

/home/shicong/git/openmpi-4.0.1/build/bin/mpirun -x LD_LIBRARY_PATH -hostfile host_file nccl_test

/home/shicong/git/openmpi-4.0.1/build/bin/mpirun -n 1 nccl_test


export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/:/home/shicong/git/openmpi-4.0.1/build/lib/
gcc -o nccl_test test.c \
    -L/home/shicong/git/openmpi-4.0.1/build/lib/ \
    -L/usr/local/cuda-10.0/lib64/ \
    -I/usr/local/cuda-10.0/include/ \
    -I/home/shicong/git/openmpi-4.0.1/build/include/ \
    -lcuda -lcudart -lnccl -lmpi



gcc -o wrapper wrapper.cc \
    -L/home/shicong/git/openmpi-4.0.1/build/lib/ \
    -L/usr/local/cuda-10.0/lib64/ \
    -I/usr/local/cuda-10.0/include/ \
    -I/home/shicong/git/openmpi-4.0.1/build/include/ \
    -lcuda -lcudart -lnccl -lmpi

gcc -o wrapper wrapper.c \
    -L/home/shicong/git/openmpi-4.0.1/build/lib/ \
    -L/usr/local/cuda-10.0/lib64/ \
    -I/usr/local/cuda-10.0/include/ \
    -I/home/shicong/git/openmpi-4.0.1/build/include/ \
    -lcuda -lcudart -lnccl -lmpi \
    && ./wrapper


#shicong@ncrh:~/git/nccl-tests$
/home/shicong/git/openmpi-4.0.1/build/bin/mpirun -x LD_LIBRARY_PATH -hostfile host_file build/all_reduce_perf -b 1 -e 1M -f 2 -g 2 -c 0
# nThread 1 nGpus 2 minBytes 1 maxBytes 1048576 step: 2(factor) warmup iters: 5 iters: 20 validation: 0
# NCCL Tests compiled with NCCL 2.4
# Using devices
#   Rank  0 on       ncrh device  0 [0x05] GeForce RTX 2080 Ti
#   Rank  1 on       ncrh device  1 [0x06] GeForce RTX 2080 Ti

#                                                 out-of-place                    in-place
#      bytes             N    type      op     time  algbw  busbw      res     time  algbw  busbw      res
#           0             0   float     sum    0.009   0.00   0.00       N/A    0.010   0.00   0.00      N/A
#           0             0   float     sum    0.009   0.00   0.00       N/A    0.009   0.00   0.00      N/A
#           4             1   float     sum    0.010   0.00   0.00       N/A    0.009   0.00   0.00      N/A
#           8             2   float     sum    0.010   0.00   0.00       N/A    0.009   0.00   0.00      N/A
#          16             4   float     sum    0.010   0.00   0.00       N/A    0.009   0.00   0.00      N/A
#          32             8   float     sum    0.010   0.00   0.00       N/A    0.009   0.00   0.00      N/A
#          64            16   float     sum    0.010   0.01   0.01       N/A    0.009   0.01   0.01      N/A
#         128            32   float     sum    0.010   0.01   0.01       N/A    0.009   0.01   0.01      N/A
#         256            64   float     sum    0.010   0.03   0.03       N/A    0.009   0.03   0.03      N/A
#         512           128   float     sum    0.010   0.05   0.05       N/A    0.009   0.05   0.05      N/A
#        1024           256   float     sum    0.010   0.10   0.10       N/A    0.010   0.11   0.11      N/A
#        2048           512   float     sum    0.010   0.20   0.20       N/A    0.010   0.20   0.20      N/A
#        4096          1024   float     sum    0.012   0.35   0.35       N/A    0.011   0.36   0.36      N/A
#        8192          2048   float     sum    0.016   0.52   0.52       N/A    0.016   0.52   0.52      N/A
#       16384          4096   float     sum    0.025   0.66   0.66       N/A    0.024   0.67   0.67      N/A
#       32768          8192   float     sum    0.041   0.80   0.80       N/A    0.040   0.81   0.81      N/A
#       65536         16384   float     sum    0.041   1.59   1.59       N/A    0.041   1.59   1.59      N/A
#      131072         32768   float     sum    0.058   2.25   2.25       N/A    0.058   2.26   2.26      N/A
#      262144         65536   float     sum    0.092   2.85   2.85       N/A    0.092   2.85   2.85      N/A
#      524288        131072   float     sum    0.158   3.31   3.31       N/A    0.158   3.31   3.31      N/A
#     1048576        262144   float     sum    0.293   3.58   3.58       N/A    0.294   3.57   3.57      N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0.778073
