#!/usr/bin/env python3
import torch

def run(
    local_rank, num_proc, func, init_method, shard_id, num_shards, backend, cfg
):

    # local_rank = process index
    # The function is called as func(i, *args), where i is the process index and args is the passed through tuple of arguments.
    world_size = num_proc * num_shards
    rank = shard_id * num_proc + local_rank

    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        raise e
    
    torch.cuda.set_device(local_rank)
    func(cfg)