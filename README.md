# cc wrapper around nccl all reduce api

# dependencies
- nccl 2.4.2
- cuda 10.0
- mpich 3.3

# compile and run
```bash
$ bash build_helper.sh
```

## notes:
- need to specify host by ip, otherwise it will failed (on mpich).
