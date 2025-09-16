import os
import glob
import json
import time
import yaml
import wandb
import argparse
from datetime import datetime

import torch
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = True

from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.optim.lr_scheduler import LinearLR

from src.datasets.contiguous_dataset import ContigDataset
from src.datasets.sleep_dataset import ContigDatasetNight

import src.utils as utils
from src.trainers.pretrainer import Pretrainer
from src.models.roformer import MultitaskRoFormerMaskedAutoEncoder
from src.loss.band_amplification_loss import BandAmplificationLoss
from src.loss.scg_loss import SCGLoss
from src.loss.bwm_loss import BWMLoss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set_json', type=str, required=True,
                        help='Path to json with list of train file paths')
    parser.add_argument('--val_set_json', type=str, required=True,
                        help='Path to json with list of val file paths')
    # Wandb arguments
    parser.add_argument('--run_name', type=str, required=True,
                        help='Name of the run for wandb')
    parser.add_argument('--run_id', type=str, required=True,
                        help='ID of the run for wandb')
    parser.add_argument('--run_group', type=str, required=True,
                        help='Group of the run for wandb')
    # Local setup arguments
    parser.add_argument('--config',type=str,required=True,
                        help='Path to pretraining config file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for logs and checkpoints')
    # const and nargs used to make arguments optional so that it can be 
    # specified with no value passed
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        nargs='?', const=None, help=(
                            'Path to a checkpoint from which to resume '
                            'training. If not specified, training will '
                            'start from scratch.'
                            )
                        )
    # DDP arguments handled by torchrun
    parser.add_argument('--dist_backend', type=str,default='nccl',
                        help='Distributed backend')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of GPUs available')
    args = parser.parse_args()
    return args

def get_dataloaders(train_files, val_files, distributed, args):
    if args.night_dataset:
        train_set = ContigDatasetNight(
            files = train_files,
            windows_per_night = args.windows_per_night,
            window_size_seconds = args.window_size_seconds,
            hours_per_night = args.hours_per_night,
            fit_overlap = args.fit_overlap,
            overlap = args.overlap,
        )
        val_set = ContigDatasetNight(
            files = val_files,
            windows_per_night = args.windows_per_night,
            window_size_seconds = args.window_size_seconds,
            hours_per_night = args.hours_per_night,
            fit_overlap = args.fit_overlap,
            overlap = args.overlap,
        )
    else:
        labels = args.labels if 'labels' in args else None
        seconds_per_label = (
            args.seconds_per_label 
            if 'seconds_per_label' 
            in args else 1
        )
        train_set = ContigDataset(
            train_files,
            args.windows_per_file,
            args.window_size_seconds,
            args.fit_overlap,
            args.overlap,
            args.min_hours_per_file,
            labels,
            seconds_per_label
        )
        val_set = ContigDataset(
            val_files,
            args.windows_per_file,
            args.window_size_seconds,
            args.fit_overlap,
            args.overlap,
            args.min_hours_per_file,
            labels,
            seconds_per_label
        )
    # Create distributed samplers if needed
    if distributed:
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set)
    else:
        train_sampler = None
        val_sampler = None

    # def seed_worker(worker_id):
    #     # Set seed for Python and NumPy in each worker
    #     worker_seed = torch.initial_seed() % 2**32
    #     np.random.seed(worker_seed)
    #     random.seed(worker_seed)
        
    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.files_per_batch,
        sampler=train_sampler,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        # worker_init_fn=seed_worker,
        shuffle=(train_sampler is None)
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.files_per_batch,
        sampler=val_sampler,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        # worker_init_fn=seed_worker,
        shuffle=False
    )
    return train_loader, val_loader

def get_criterion(args):
    if args.criterion == 'scg':
        criterion = SCGLoss(
            window_samples = args.patch_size,
            sample_freq = args.sample_freq,
            std_cutoff = args.std_cutoff,
            jerk_band = args.jerk_band,
            pulse_band = args.pulse_band,
            reference_band = args.reference_band,
            amplification = args.amplification,
            downsample = args.downsample,
            mode = args.mode,
        )

    elif args.criterion == 'bwm':
        criterion = BWMLoss(
            window_samples = args.patch_size,
            sample_freq = args.sample_freq,
            std_cutoff = args.std_cutoff,
            breathing_band = args.breathing_band,
            reference_band = args.reference_band,
            amplification = args.amplification,
            downsample = args.downsample,
        )

    elif args.criterion == 'bandamp':
        criterion = BandAmplificationLoss(
            patch_size = args.patch_size,
            sample_freq = args.sample_freq,
            weighted_downsampling_scheme = args.weighted_downsampling_scheme,
            log_transform = args.log_transform,
            loss_norm = args.loss_norm,
            std_cutoff = args.std_cutoff,
            std_cutoff_type = args.std_cutoff_type,
            invert_cutoff = args.invert_cutoff,
            weight_clamp = args.weight_clamp,
            patchwise_fft_kwargs = args.patchwise_fft_kwargs,
            reduction = 'mean',
        )
    return criterion

def get_linear_scheduler(optimizer, warmup_steps:int) -> LinearLR:
    '''Get a linear learning rate scheduler for the optimizer'''
    scheduler = LinearLR(
        optimizer,
        start_factor=0.2,
        total_iters=warmup_steps,
    )
    return scheduler

def main(args):
    print(f'Current run name: {args.run_name}')
    # Set up distributed training
    device, distributed = utils.ddp_utils.setup_distributed(args)
    # If output_dir already exists the date and time will be appended
    utils.setup_utils.setup_output_dir(args)
    print(f'Output directory: {args.output_dir}')
    # Loads the config file params into args and saves the config file 
    utils.setup_utils.setup_config(args)
    utils.setup_utils.set_seed(args.seed)

    # Initialize wandb logging
    if args.local_rank == 0:  # only on main process
        run = wandb.init(
            project='wassed_pt',
            name=args.run_name,
            id=args.run_id,
            group=args.run_group,
            config=args.__dict__,
            dir=args.output_dir,
            resume='allow',
            settings=wandb.Settings(code_dir="./src"),
        )

    # Load train and val sets
    with open(args.train_set_json,'r') as f:
        train_files = json.load(f)
    print(f'Training on {len(train_files)} files')
    with open(args.val_set_json,'r') as f:
        val_files = json.load(f)
    print(f'Validating on {len(val_files)} files')

    # Truncate train set if train_prcnt < 1
    train_files = train_files[:int(args.train_prcnt*len(train_files))]

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        train_files,
        val_files,
        distributed,
        args,
    )

    # Define MAE model
    model = MultitaskRoFormerMaskedAutoEncoder(
        patch_size = args.patch_size,
        in_channels = args.in_channels,
        embed_dim = args.embed_dim,
        num_heads = args.num_heads,
        mlp_ratio = args.mlp_ratio,
        num_layers = args.n_encoder_layers,
        max_seq_len = args.seq_len,
        dropout = args.dropout,
        num_tasks = args.num_tasks,
    )

    # Get the number of trainable parameters
    num_params = sum(
            [p.numel() for p in model.parameters() if p.requires_grad]
    )
    if args.local_rank == 0:
        wandb.run.summary['num_params'] = num_params
    print(f'Model has {num_params} trainable parameters.')
    
    windows_per_file = args.windows_per_night if hasattr(args, 'windows_per_night') else args.windows_per_file
    # Calculate effective batch size
    eff_batch_size = (
        args.files_per_batch 
        * windows_per_file
        * args.grad_accumulation_steps 
        * args.world_size
    )
    print(f'Effective batch size: {eff_batch_size}')
    
    # Base LR is scaled by effective batch size unless LR is provided
    lr = args.lr if args.lr else args.blr*eff_batch_size/256
    print(f'Learning rate: {lr}')
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )

    # Get learning rate scheduler
    steps_per_epoch = len(train_loader) // args.grad_accumulation_steps
    scheduler = get_linear_scheduler(optimizer, steps_per_epoch * args.warmup_epochs)
    scaler = GradScaler('cuda')

    criterion = get_criterion(args).to(device)

    model = model.to(device)

    if args.checkpoint_path:
        # Load checkpoint if resuming training
        checkpoint = torch.load(
            args.checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
        step_count = checkpoint['step']
        patience_counter = checkpoint['patience']
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
    else:
        # Initialize counters if starting from scratch 
        start_epoch = step_count = patience_counter = 0

    ## Set up model for distributed training
    #---------------------------
    if distributed:
        model = DDP(
            model,
            device_ids = [args.local_rank],
            output_device = args.local_rank,
        )
        bare_model = model.module
    else:
        bare_model = model
    #---------------------------

    ## Trainer
    trainer = Pretrainer(
        model,
        optimizer,
        criterion,
        scaler,
        scheduler,
        distributed,
    )
    trainer.epoch = start_epoch
    trainer.step_count = step_count
    trainer.patience_counter = patience_counter

    ## Training loop
    # ---------------------------
    start_time = time.time()
    print(f'Starting training at {datetime.now()}')
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch,args.max_epochs):
        trainer.train_epoch(train_loader,args)
        trainer.validate(val_loader,args)
        print(f'Epoch: {epoch}')
        print(f'training time: {(time.time() - start_time)/60:.2f}min')

        # save checkpoint after each epoch
        if args.local_rank == 0:
            utils.model_io.save_checkpoint(
                args.output_dir,
                bare_model,
                optimizer,
                scheduler,
                scaler,
                trainer.epoch,
                trainer.step_count,
                trainer.patience_counter
            )
        
    # Log training time
    print(f'Training finished at {datetime.now()}')
    print(f'Training took {(time.time() - start_time)/60:.2f} minutes.')
    
    # TODO Save model in onnx format with wandb

    # Terminate wandb run
    if args.local_rank == 0:
        run.finish()
    #---------------------------

if __name__ == '__main__':
    args = parse_args()
    main(args)