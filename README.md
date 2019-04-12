# cc wrapper around nccl all reduce api

# API
```c++
/* 
 *   params: nDev - N GPU per thread, e.g. 1 if 2 threads are launched on each node with 2 GPUs
 */
  Communicator(int nDev);
 /* 
 *   params: size - number of params in the buffer
 *           sendbuff - an array of cuda pointers points to cuda 
 *                      memory on all GPU on 1 node and share same size,
 *                      e.g. [ GPU_0_params, GPU_1_params, ...]
 *           recvbuff - same structure as sendbuff, if sendbuff is used
 *                      here, then in-place operation will be done
 */
  void allReduce(int size, float** sendbuff, float** recvbuff);
  /*
   * Cuda stream synchronization, function will return when in sync
   */
  void wait();
```

# dependencies
- nccl 2.4.2
- cuda 10.0
- mpich 3.3

# testing on ResNet56v1

### network reference: keras cifar10 [example](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
### the param counts are extracted from network summary, which could be found [here](https://gist.github.com/dcslin/837788ff63f5cfc5204e6d5bb719937d)

The simulation is done with the extracted param counts: e.g.

| layer | param count |
|--|--|
| conv2d_1 | 448 |
| batch_normalization_1 | 64 |
| conv2d_2 | 2320 |
| ... | ... |

### tested with fusion
#### Fusion threshold set to 1KB, 10KB, 100KB, 1000KB, ran on 2 nodes, 2 gpu each node.
2 modes are ran:
- mode 1: 1 threads per node, 2 gpu per thread
- mode 2: 2 threads per node, 1 gpu per thread

#### all reduce setup:
Blocking execution or in other words, run `wait()` following each `allReduce()`.

#### Conclusion:
- Small fusion Threshold does not affect much


``` bash
$ bash run_resnet_test.sh
tested on 2 nodes, 2 gpu each
Total params: 861,770
Fusion Threshold is 1 KB - Total Batches: 58
Batches of params after fusion: 1.8KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 18.4KB, 36.6KB, 2.1KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 72.8KB, 145.2KB, 8.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 3.5KB, 

*****gpu per thread: 2 - Time: 0.128103123 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 10 KB - Total Batches: 45
Batches of params after fusion: 11.1KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 27.7KB, 36.6KB, 38.8KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 72.8KB, 145.2KB, 153.5KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 

*****gpu per thread: 2 - Time: 0.098126687 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 100 KB - Total Batches: 25
Batches of params after fusion: 104.2KB, 120.2KB, 112.0KB, 109.9KB, 109.9KB, 109.9KB, 109.9KB, 109.4KB, 145.2KB, 153.5KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 

*****gpu per thread: 2 - Time: 0.097817127 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 1000 KB - Total Batches: 3
Batches of params after fusion: 1030.5KB, 1025.0KB, 1016.8KB, 

*****gpu per thread: 2 - Time: 0.059677801 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 1 KB - Total Batches: 58
Batches of params after fusion: 1.8KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 18.4KB, 36.6KB, 2.1KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 72.8KB, 145.2KB, 8.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 3.5KB, 

*****gpu per thread: 1 - Time: 0.107954671 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 10 KB - Total Batches: 45
Batches of params after fusion: 11.1KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 27.7KB, 36.6KB, 38.8KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 72.8KB, 145.2KB, 153.5KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 

*****gpu per thread: 1 - Time: 0.084592842 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 100 KB - Total Batches: 25
Batches of params after fusion: 104.2KB, 120.2KB, 112.0KB, 109.9KB, 109.9KB, 109.9KB, 109.9KB, 109.4KB, 145.2KB, 153.5KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 

*****gpu per thread: 1 - Time: 0.087390314 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 1000 KB - Total Batches: 3
Batches of params after fusion: 1030.5KB, 1025.0KB, 1016.8KB, 

*****gpu per thread: 1 - Time: 0.054903144 sec (avg over repeated 10 times)*****
=======================================================
```

# testing output validating all reduce values:
```
$ bash value_test.sh
tested on 2 nodes, 2gpu each
nDev per thread: 2 - number of float:         5 - Time: 0.000711 sec (avg over repeated 10 times)
Before: 10.000000000, 11.000000000,...
After:  40.000000000, 44.000000000,...
nDev per thread: 1 - number of float:         5 - Time: 0.001008 sec (avg over repeated 10 times)
Before: 10.000000000, 11.000000000,...
After:  40.000000000, 44.000000000,...
```

# testing output measuring `allReduce()` and `wait()` combined
```
$ bash time_test.sh
tested on 2 nodes, 2gpu each
nDev per thread: 2 - number of float:         1 - Time: 0.002367 sec (avg over repeated 10 times)
nDev per thread: 2 - number of float:       100 - Time: 0.002360 sec (avg over repeated 10 times)
nDev per thread: 2 - number of float:     10000 - Time: 0.003880 sec (avg over repeated 10 times)
nDev per thread: 2 - number of float:   1000000 - Time: 0.052831 sec (avg over repeated 10 times)
nDev per thread: 1 - number of float:         1 - Time: 0.002359 sec (avg over repeated 10 times)
nDev per thread: 1 - number of float:       100 - Time: 0.002393 sec (avg over repeated 10 times)
nDev per thread: 1 - number of float:     10000 - Time: 0.003907 sec (avg over repeated 10 times)
nDev per thread: 1 - number of float:   1000000 - Time: 0.051799 sec (avg over repeated 10 times)
```

## notes:
- need to specify host by ip, otherwise it will failed (on mpich).
