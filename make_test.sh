rm -rf ./test
gcc -o test test.c -L/usr/local/cuda-9.0/targets/x86_64-linux/lib -I/usr/local/cuda-9.0/targets/x86_64-linux/include -lcuda -lcudart -lnccl -lmpi
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun  --allow-run-as-root -n 1 -hostfile host_file ./test
#OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun --allow-run-as-root --help
#OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun --allow-run-as-root -v
