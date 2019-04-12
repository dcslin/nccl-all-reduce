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

# testing on ResNet50 

### network reference on keras: [Classify ImageNet classes with ResNet50](https://keras.io/applications/#classify-imagenet-classes-with-resnet50)
### the param counts are extracted from network summary, which could be found [here](https://gist.github.com/dcslin/5586c635afae59ed6a27b41c95408c77)

The simulation is done with the extracted param counts: e.g.

| layer | param count |
|--|--|
| conv1 | 9472 |
| bn_conv1 | 256 |
| res2a_branch2a | 4160 |
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
Total params: 25,636,712
Trainable params: 25,583,592
Non-trainable params: 53,120
Fusion Threshold is 1 KB - Total Batches: 100
Batches of params after fusion: 37.0KB, 17.2KB, 145.2KB, 66.0KB, 65.0KB, 4.0KB, 4.0KB, 64.2KB, 145.2KB, 66.0KB, 4.0KB, 64.2KB, 145.2KB, 66.0KB, 4.0KB, 128.5KB, 2.0KB, 576.5KB, 2.0KB, 258.0KB, 514.0KB, 8.0KB, 8.0KB, 256.5KB, 2.0KB, 576.5KB, 2.0KB, 258.0KB, 8.0KB, 256.5KB, 2.0KB, 576.5KB, 2.0KB, 258.0KB, 8.0KB, 256.5KB, 2.0KB, 576.5KB, 2.0KB, 258.0KB, 8.0KB, 513.0KB, 4.0KB, 2305.0KB, 4.0KB, 1028.0KB, 2052.0KB, 16.0KB, 16.0KB, 1025.0KB, 4.0KB, 2305.0KB, 4.0KB, 1028.0KB, 16.0KB, 1025.0KB, 4.0KB, 2305.0KB, 4.0KB, 1028.0KB, 16.0KB, 1025.0KB, 4.0KB, 2305.0KB, 4.0KB, 1028.0KB, 16.0KB, 1025.0KB, 4.0KB, 2305.0KB, 4.0KB, 1028.0KB, 16.0KB, 1025.0KB, 4.0KB, 2305.0KB, 4.0KB, 1028.0KB, 16.0KB, 2050.0KB, 8.0KB, 9218.0KB, 8.0KB, 4104.0KB, 8200.0KB, 32.0KB, 32.0KB, 4098.0KB, 8.0KB, 9218.0KB, 8.0KB, 4104.0KB, 32.0KB, 4098.0KB, 8.0KB, 9218.0KB, 8.0KB, 4104.0KB, 32.0KB, 8003.9KB, 

*****gpu per thread: 2 - Time: 2.310482182 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 10 KB - Total Batches: 66
Batches of params after fusion: 37.0KB, 17.2KB, 145.2KB, 66.0KB, 65.0KB, 72.2KB, 145.2KB, 66.0KB, 68.2KB, 145.2KB, 66.0KB, 132.5KB, 578.5KB, 260.0KB, 514.0KB, 16.0KB, 256.5KB, 578.5KB, 260.0KB, 264.5KB, 578.5KB, 260.0KB, 264.5KB, 578.5KB, 260.0KB, 521.0KB, 2309.0KB, 1032.0KB, 2052.0KB, 16.0KB, 16.0KB, 1025.0KB, 2309.0KB, 1032.0KB, 16.0KB, 1025.0KB, 2309.0KB, 1032.0KB, 16.0KB, 1025.0KB, 2309.0KB, 1032.0KB, 16.0KB, 1025.0KB, 2309.0KB, 1032.0KB, 16.0KB, 1025.0KB, 2309.0KB, 1032.0KB, 16.0KB, 2050.0KB, 9226.0KB, 4112.0KB, 8200.0KB, 32.0KB, 32.0KB, 4098.0KB, 9226.0KB, 4112.0KB, 32.0KB, 4098.0KB, 9226.0KB, 4112.0KB, 32.0KB, 8003.9KB, 

*****gpu per thread: 2 - Time: 2.282010829 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 100 KB - Total Batches: 48
Batches of params after fusion: 199.5KB, 131.0KB, 217.5KB, 134.2KB, 145.2KB, 198.5KB, 578.5KB, 260.0KB, 514.0KB, 272.5KB, 578.5KB, 260.0KB, 264.5KB, 578.5KB, 260.0KB, 264.5KB, 578.5KB, 260.0KB, 521.0KB, 2309.0KB, 1032.0KB, 2052.0KB, 1057.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 2066.0KB, 9226.0KB, 4112.0KB, 8200.0KB, 4162.0KB, 9226.0KB, 4112.0KB, 4130.0KB, 9226.0KB, 4112.0KB, 8035.9KB, 

*****gpu per thread: 2 - Time: 2.353430335 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 1000 KB - Total Batches: 34
Batches of params after fusion: 1026.0KB, 1352.5KB, 1111.0KB, 1103.0KB, 1103.0KB, 2830.0KB, 1032.0KB, 2052.0KB, 1057.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 2066.0KB, 9226.0KB, 4112.0KB, 8200.0KB, 4162.0KB, 9226.0KB, 4112.0KB, 4130.0KB, 9226.0KB, 4112.0KB, 8035.9KB, 

*****gpu per thread: 2 - Time: 2.365061736 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 10000 KB - Total Batches: 8
Batches of params after fusion: 11609.5KB, 12130.0KB, 11862.0KB, 13338.0KB, 12362.0KB, 13338.0KB, 13356.0KB, 12147.9KB, 

*****gpu per thread: 2 - Time: 1.996319349 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 1 KB - Total Batches: 100
Batches of params after fusion: 37.0KB, 17.2KB, 145.2KB, 66.0KB, 65.0KB, 4.0KB, 4.0KB, 64.2KB, 145.2KB, 66.0KB, 4.0KB, 64.2KB, 145.2KB, 66.0KB, 4.0KB, 128.5KB, 2.0KB, 576.5KB, 2.0KB, 258.0KB, 514.0KB, 8.0KB, 8.0KB, 256.5KB, 2.0KB, 576.5KB, 2.0KB, 258.0KB, 8.0KB, 256.5KB, 2.0KB, 576.5KB, 2.0KB, 258.0KB, 8.0KB, 256.5KB, 2.0KB, 576.5KB, 2.0KB, 258.0KB, 8.0KB, 513.0KB, 4.0KB, 2305.0KB, 4.0KB, 1028.0KB, 2052.0KB, 16.0KB, 16.0KB, 1025.0KB, 4.0KB, 2305.0KB, 4.0KB, 1028.0KB, 16.0KB, 1025.0KB, 4.0KB, 2305.0KB, 4.0KB, 1028.0KB, 16.0KB, 1025.0KB, 4.0KB, 2305.0KB, 4.0KB, 1028.0KB, 16.0KB, 1025.0KB, 4.0KB, 2305.0KB, 4.0KB, 1028.0KB, 16.0KB, 1025.0KB, 4.0KB, 2305.0KB, 4.0KB, 1028.0KB, 16.0KB, 2050.0KB, 8.0KB, 9218.0KB, 8.0KB, 4104.0KB, 8200.0KB, 32.0KB, 32.0KB, 4098.0KB, 8.0KB, 9218.0KB, 8.0KB, 4104.0KB, 32.0KB, 4098.0KB, 8.0KB, 9218.0KB, 8.0KB, 4104.0KB, 32.0KB, 8003.9KB, 

*****gpu per thread: 1 - Time: 2.533200475 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 10 KB - Total Batches: 66
Batches of params after fusion: 37.0KB, 17.2KB, 145.2KB, 66.0KB, 65.0KB, 72.2KB, 145.2KB, 66.0KB, 68.2KB, 145.2KB, 66.0KB, 132.5KB, 578.5KB, 260.0KB, 514.0KB, 16.0KB, 256.5KB, 578.5KB, 260.0KB, 264.5KB, 578.5KB, 260.0KB, 264.5KB, 578.5KB, 260.0KB, 521.0KB, 2309.0KB, 1032.0KB, 2052.0KB, 16.0KB, 16.0KB, 1025.0KB, 2309.0KB, 1032.0KB, 16.0KB, 1025.0KB, 2309.0KB, 1032.0KB, 16.0KB, 1025.0KB, 2309.0KB, 1032.0KB, 16.0KB, 1025.0KB, 2309.0KB, 1032.0KB, 16.0KB, 1025.0KB, 2309.0KB, 1032.0KB, 16.0KB, 2050.0KB, 9226.0KB, 4112.0KB, 8200.0KB, 32.0KB, 32.0KB, 4098.0KB, 9226.0KB, 4112.0KB, 32.0KB, 4098.0KB, 9226.0KB, 4112.0KB, 32.0KB, 8003.9KB, 

*****gpu per thread: 1 - Time: 2.364754549 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 100 KB - Total Batches: 48
Batches of params after fusion: 199.5KB, 131.0KB, 217.5KB, 134.2KB, 145.2KB, 198.5KB, 578.5KB, 260.0KB, 514.0KB, 272.5KB, 578.5KB, 260.0KB, 264.5KB, 578.5KB, 260.0KB, 264.5KB, 578.5KB, 260.0KB, 521.0KB, 2309.0KB, 1032.0KB, 2052.0KB, 1057.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 2066.0KB, 9226.0KB, 4112.0KB, 8200.0KB, 4162.0KB, 9226.0KB, 4112.0KB, 4130.0KB, 9226.0KB, 4112.0KB, 8035.9KB, 

*****gpu per thread: 1 - Time: 2.256444651 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 1000 KB - Total Batches: 34
Batches of params after fusion: 1026.0KB, 1352.5KB, 1111.0KB, 1103.0KB, 1103.0KB, 2830.0KB, 1032.0KB, 2052.0KB, 1057.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 1041.0KB, 2309.0KB, 1032.0KB, 2066.0KB, 9226.0KB, 4112.0KB, 8200.0KB, 4162.0KB, 9226.0KB, 4112.0KB, 4130.0KB, 9226.0KB, 4112.0KB, 8035.9KB, 

*****gpu per thread: 1 - Time: 2.276045096 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 10000 KB - Total Batches: 8
Batches of params after fusion: 11609.5KB, 12130.0KB, 11862.0KB, 13338.0KB, 12362.0KB, 13338.0KB, 13356.0KB, 12147.9KB, 

*****gpu per thread: 1 - Time: 2.188430121 sec (avg over repeated 10 times)*****
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
