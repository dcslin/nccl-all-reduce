# cc wrapper around nccl all reduce api

# dependencies
- nccl 2.4.2
- cuda 10.0
- mpich 3.3

# testing on ResNet56v1

### network is implemented in keras cifar10 [example](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
### network summary could be found [here](https://gist.github.com/dcslin/837788ff63f5cfc5204e6d5bb719937d)

``` bash
$ bash run_resnet_test.sh
tested on 2 nodes, 2 gpu each
Total params: 861,770
gpu per thread: 2 - Time: 0.209903 sec (avg over repeated 10 times)
gpu per thread: 1 - Time: 0.274776 sec (avg over repeated 10 times)
```

### with fusion(WIP):
``` bash
$ 
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
