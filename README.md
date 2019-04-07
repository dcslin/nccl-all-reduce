# cc wrapper around nccl all reduce api

# dependencies
- nccl 2.4.2
- cuda 10.0
- mpich 3.3

# compile and run
```bash
$ bash build_helper.sh
```

# output of 2 hosts, 2 threads per host, 1 gpu per thread
```
g rank 2
g rank 3
g rank 1
g rank 0
l rank 1
l rank 1
l rank 0
l rank 0
doing all reduce..
doing waiting..
doing all reduce..
doing waiting..
doing all reduce..
doing waiting..
doing all reduce..
doing waiting..
received buff: 40 vs 40 expected
received buff: 44 vs 44 expected
received buff: 40 vs 40 expected
received buff: 44 vs 44 expected
received buff: 40 vs 40 expected
received buff: 44 vs 44 expected
received buff: 40 vs 40 expected
received buff: 44 vs 44 expected
```

## notes:
- need to specify host by ip, otherwise it will failed (on mpich).
