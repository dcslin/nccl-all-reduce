# cc wrapper around nccl all reduce api

# dependencies
- nccl 2.4.2
- cuda 10.0
- mpich 3.3

# testing on ResNet56v1

### network is implemented in keras cifar10 [example](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
### network summary could be found [here](https://gist.github.com/dcslin/837788ff63f5cfc5204e6d5bb719937d)

### with fusion:
``` bash
$ bash run_resnet_test.sh
tested on 2 nodes, 2 gpu each
Total params: 861,770
Fusion Threshold is 1 KB
Batches of params after fusion: 1.8KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 18.4KB, 36.6KB, 2.1KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 72.8KB, 145.2KB, 8.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 3.5KB, 

*****gpu per thread: 2 - Time: 0.168035321 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 10 KB
Batches of params after fusion: 11.1KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 27.7KB, 36.6KB, 38.8KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 72.8KB, 145.2KB, 153.5KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 

*****gpu per thread: 2 - Time: 0.175501675 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 100 KB
Batches of params after fusion: 104.2KB, 120.2KB, 112.0KB, 109.9KB, 109.9KB, 109.9KB, 109.9KB, 109.4KB, 145.2KB, 153.5KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 

*****gpu per thread: 2 - Time: 0.160774826 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 1000 KB
Batches of params after fusion: 1030.5KB, 1025.0KB, 1016.8KB, 

*****gpu per thread: 2 - Time: 0.063409082 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 1 KB
Batches of params after fusion: 1.8KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 9.3KB, 18.4KB, 36.6KB, 2.1KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 72.8KB, 145.2KB, 8.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 3.5KB, 

*****gpu per thread: 1 - Time: 0.169025111 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 10 KB
Batches of params after fusion: 11.1KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 18.6KB, 27.7KB, 36.6KB, 38.8KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 36.6KB, 72.8KB, 145.2KB, 153.5KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 

*****gpu per thread: 1 - Time: 0.183856286 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 100 KB
Batches of params after fusion: 104.2KB, 120.2KB, 112.0KB, 109.9KB, 109.9KB, 109.9KB, 109.9KB, 109.4KB, 145.2KB, 153.5KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 145.2KB, 

*****gpu per thread: 1 - Time: 0.128925127 sec (avg over repeated 10 times)*****
=======================================================
Fusion Threshold is 1000 KB
Batches of params after fusion: 1030.5KB, 1025.0KB, 1016.8KB, 

*****gpu per thread: 1 - Time: 0.080684796 sec (avg over repeated 10 times)*****
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
