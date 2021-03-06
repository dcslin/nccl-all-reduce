export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/:/home/shicong/mpich-3.3/build/lib/
BINARY_NAME=test_resnet
g++ -g -Wall -std=c++11 \
    -I/usr/local/cuda-10.0/include/ \
    -I/home/shicong/mpich-3.3/build/include/ \
    -L/home/shicong/mpich-3.3/build/lib/ \
    -L/usr/local/cuda-10.0/lib64/ \
    $BINARY_NAME.cc communicator.cc  \
    -lmpi -lmpicxx -lcuda -lcudart -lnccl \
    -o $BINARY_NAME \
&& echo "tested on 2 nodes, 2 gpu each" \
&& echo "Total params: 25,636,712" \
&& echo "Trainable params: 25,583,592" \
&& echo "Non-trainable params: 53,120" \
&& mpiexec --hostfile host_file_ip_1_process ./$BINARY_NAME 2 \
&& mpiexec --hostfile host_file_ip_2_process ./$BINARY_NAME 1 \
&& echo ""
#&& ./$BINARY_NAME 2 \
