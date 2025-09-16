import os
import yaml
import random
import argparse
import numpy as np
import torch
from datetime import datetime

def setup_config(args):
    # Load the config file
    with open(args.config,'r') as f:
        config = yaml.safe_load(f)
    if args.local_rank == 0: 
        # Save the config to the output directory
        with open(args.output_dir+'config.yaml','w') as f:
            yaml.dump(config,f)
        # Save the passed command line arguments to a separte yaml file
        with open(args.output_dir+'args.yaml','w') as f:
            yaml.dump(args.__dict__,f)
    # Update the args with the config file
    args.__dict__.update(config)
    
def setup_output_dir(args):
    ## Setup output directory 
    if args.local_rank == 0:
        try:
            os.makedirs(args.output_dir)
        # If directory already exists, append current time to directory name
        except FileExistsError:
            current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            args.output_dir = f'{args.output_dir[:-1]}_{current_time}/'
            os.makedirs(args.output_dir)

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)