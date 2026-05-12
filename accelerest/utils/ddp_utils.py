import os
import builtins

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def setup_distributed(args):
    if 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
    
    if args.local_rank != -1 and args.world_size > 1:
        # Initialize distributed training
        dist.init_process_group(backend=args.dist_backend,world_size=args.world_size)
        # Set device to the local rank (GPU)
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda',args.local_rank)
        # Suppress output from all GPUs except the master
        suppress_output_if_not_master(args.local_rank)
        # Check if distributed training is enabled
        distributed = is_distributed()
        if distributed:
            print(f'Distributed traininng with {dist.get_world_size()} gpu(s)')
        else:
            print('Distributed training not enabled.')

    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}, not distributed.')
        distributed = False
        args.local_rank = 0
    return device, distributed

def suppress_output_if_not_master(local_rank:int):
    # Suppress printing if not on master gpu
    if local_rank != 0:
        def print_pass(*args):
            pass
        # builtins.print = print_pass
    return

def is_distributed():
    dist_init = dist.is_available() and dist.is_initialized()
    multi_gpu = dist.get_world_size() > 1
    distributed = multi_gpu and dist_init
    return distributed

def all_reduce_mean(x:float):
    '''Get the mean of the input value across all GPUs'''
    tensor = torch.tensor(x).cuda()#.to(device)
    dist.all_reduce(tensor)
    if torch.numel(tensor) == 1:
        reduced_value = tensor.item()/dist.get_world_size()
    elif torch.numel(tensor) > 1:
        reduced_value = tensor.cpu()/dist.get_world_size()
    return reduced_value