import os
import math
import time
import yaml
import json
import wandb
import argparse
from datetime import datetime

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import src.utils as utils
from src.models.roformer import RoFormerRegression
from src.models.harnet_lstm import HARNetLSTM
from src.models.sleepnet import sleepnet
from src.models.accnet import AcceleroNet, AccelFormer
from src.datasets.sleep_dataset import WindowDataset
from src.trainers.finetuner import FinetuneRegression

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cv_folds_path', type=str, required=True,
    #                     help='Path to cross validation fold json files')
    # Wandb arguments
    parser.add_argument('--run_name', type=str, required=True,
                        help='Name of the run for wandb and output path')
    parser.add_argument('--run_id', type=str, required=True,
                        help='ID of the run for wandb')

    # Local setup arguments
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file for a finetuning run.')
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='Path to pretrained model checkpoint.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for logs and checkpoints')
    # const and nargs used to make arguments optional so that it can be 
    # specified with no value passed
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        nargs='?', const=None, help=(
                            'Path to a checkpoint from which to resume '
                            'finetuning. If not specified, training will '
                            'start from scratch.'
                            )
                        )

    # DDP configs
    parser.add_argument('--dist_backend', type=str, default='nccl',
                        help='distributed backend')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of nodes for distributed training')
    args = parser.parse_args()
    return args

def get_window_dataloaders(
        train_files: list[str],
        val_files: list[str],
        distributed: bool,
        args,
    ) -> tuple[DataLoader, DataLoader, WindowDataset, WindowDataset]:
    
    train_set = WindowDataset(
        files = train_files,
        patch_size_samples = args.patch_size,
        context_window_patches = args.seq_len,
        step_patches = args.step_patches,
        labels = args.labels,
        label_map = args.label_map,
        label_priority = args.label_priority,
        patches_per_label = args.patches_per_label,
        ignore_index = args.ignore_index,
        class_weights = args.class_weights,
        downsample_negative = args.downsample_negative
    )
    val_set = WindowDataset(
        files = val_files,
        patch_size_samples = args.patch_size,
        context_window_patches = args.seq_len,
        step_patches = args.step_patches,
        labels = args.labels,
        label_map = args.label_map,
        label_priority = args.label_priority,
        patches_per_label = args.patches_per_label,
        ignore_index = args.ignore_index,
    )
    # Create distributed samplers if needed
    if distributed:
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set)
    else:
        train_sampler = None
        val_sampler = None

    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        shuffle=(train_sampler is None),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        shuffle=False
    )
    return train_loader, val_loader, train_set, val_set

def freeze_encoder(model):
    for param in model.patch_embedding.parameters():
        param.requires_grad = False
    for param in model.encoder.parameters():
        param.requires_grad = False

def get_roformer_regression(args):
    # Get transformer classifier model
    model = RoFormerRegression(
            mode = args.finetune_mode,
            num_targets = args.num_targets,
            patch_size = args.patch_size,
            in_channels = args.in_channels,
            embed_dim = args.embed_dim,
            num_heads = args.num_heads,
            mlp_ratio = args.mlp_ratio,
            num_layers = args.n_encoder_layers,
            num_lstm_layers = args.num_lstm_layers,
            max_seq_len = args.seq_len,
            encoder_dropout = args.encoder_dropout,
            head_dropout = args.head_dropout,
            head = args.head,
            lstm_dim = args.lstm_dim,
    ) 
  
    if args.pretrained_model == 'none':
        print('Training from scratch')
    else:
        # Load pretrained encoder parameters
        checkpoint = torch.load(
            args.pretrained_model,
            map_location='cpu',
            weights_only=True,
        )
        loded_model_params = set(checkpoint["model"].keys())
        model_params = set(model.state_dict().keys())
        print(f'The following loaded parameters will be used:\n{sorted(list(loded_model_params.intersection(model_params)))}')
        print(f'The following loaded parameters will not be used:\n{list(loded_model_params.difference(model_params))}')
        print(f'The following model parameters will be trained from scratch:\n{list(model_params.difference(loded_model_params))}')
        model.load_state_dict(checkpoint['model'], strict=False)

    if args.finetune_train == "head_only":
        freeze_encoder(model)
    elif args.finetune_train == "full":
        pass
    else:
        raise ValueError("finetune_train must be one of 'head_only' or 'full'")    

    return model

def freeze_feature_extractor(model):
    # Set Batch_norm running stats to be frozen
    # Only freezing ConV layers for now
    # or it will lead to bad results
    # http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/
    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm1d") != -1:
            m.eval()
    i = 0
    for name, param in model.named_parameters():
        if name.split(".")[0] == "feature_extractor":
            param.requires_grad = False
            i += 1
    print("Weights being frozen: %d" % i)
    model.apply(set_bn_eval)

def get_accelformer(args):
    model = AccelFormer(
        sample_freq = args.sample_freq,
        acc_decomp = args.acc_decomp,
        mode = args.finetune_mode,
        num_classes = args.num_classes,
        patch_size = args.patch_size,
        embed_dim = args.embed_dim,
        num_heads = args.num_heads,
        mlp_ratio = args.mlp_ratio,
        num_layers = args.n_encoder_layers,
        num_lstm_layers = args.num_lstm_layers,
        max_seq_len = args.seq_len,
        encoder_dropout = args.encoder_dropout,
        head_dropout = args.head_dropout,
    )
    return model

def setup_output(args):
        # If output_dir already exists the date and time will be appended
    ## Setup output directory 
    if args.local_rank == 0:
        try:
            os.makedirs(args.output_dir)
        # If directory already exists, append current time to directory name
        except FileExistsError:
            # Check if training is already finished for this fold
            if os.path.exists(os.path.join(args.output_dir, 'validation_metrics.pt')):
                print(f'Fold {fold} already finished. Skipping...')
                return
            else:
                current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
                args.output_dir = f'{args.output_dir}_{current_time}/'
                os.makedirs(args.output_dir)

    print(f'Output directory: {args.output_dir}')

def finetune(args, device, distributed, fold = None):
    fold_str = f'_fold{fold}' if fold is not None else ''
    tp_str = f'_tp{args.train_prcnt}' if args.train_prcnt < 1.0 else ''
    print(f'Current run name: {args.run_name}' + fold_str + tp_str)

    # Dataloaders
    with open(args.train_set_json,'r') as f:
        train_files = json.load(f)
    train_files = train_files[:int(args.train_prcnt*len(train_files))]
    with open(args.val_set_json,'r') as f:
        val_files = json.load(f)
    print(f'Num files in train_set {len(train_files)}')
    print(f'Num files in val_set {len(val_files)}')

    (train_loader, 
    val_loader,
    train_set,
    val_set
    ) = get_window_dataloaders(
        train_files,
        val_files,
        distributed,
        args,
    )
    print(f'Training set size: {len(train_set)}, Validation set size: {len(val_set)}')

    # Calculate effective batch size
    eff_batch_size = (
        args.batch_size
        * args.grad_accumulation_steps 
        * args.world_size
    )
    print(f'Effective batch size: {eff_batch_size}')
        
    # Base LR is scaled by effective batch size unless LR is provided
    lr = args.lr if args.lr else args.blr * eff_batch_size / 256
    print(f'Learning rate: {lr}')
    
    if args.model_type == 'mae':
        model = get_roformer_regression(args)
    elif args.model_type == 'accelformer':
        model = get_accelformer(args)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler('cuda')
    scheduler = None
    
    criterion = torch.nn.MSELoss()

    ## Load checkpoint if resuming training
    #---------------------------
    if args.checkpoint_path:
        (start_epoch,
        step_count,
        patience_counter
        ) = utils.model_io.load_checkpoint(
            args.checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
        )    
    else:
        start_epoch = step_count = patience_counter = 0
    #---------------------------
    
    # Initialize wandb logging
    if args.local_rank == 0:  # only on main process
        run = wandb.init(
            project = 'wassed_ft',
            name = args.run_name + fold_str + tp_str,
            id = args.run_id + fold_str + tp_str,
            group = args.run_name,
            config = args.__dict__,
            dir = args.output_dir,
            resume = 'allow',
            settings = wandb.Settings(code_dir="./src"),
        )
        # Log setup
        wandb.run.summary['train_set_files'] = len(train_files)
        wandb.run.summary['val_set_files'] = len(val_files)
        wandb.run.summary['train_set_size'] = len(train_set)
        wandb.run.summary['val_set_size'] = len(val_set)
        wandb.run.summary['effective_batch_size'] = eff_batch_size
        wandb.run.summary['learning_rate'] = lr

        # Log trainable parameters
        trainable_params = []
        frozen_params = []
        for pname, param in model.named_parameters():
            print(f'{pname}: requires_grad={param.requires_grad}')
            if param.requires_grad:
                trainable_params.append(pname)
            else:
                frozen_params.append(pname)
        wandb.run.summary['trainable_params'] = trainable_params
        wandb.run.summary['frozen_params'] = frozen_params

    ## Set up model for distributed training
    #---------------------------
    model = model.to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )
        bare_model = model.module
    else:
        bare_model = model
    #---------------------------

    ## Trainer
    trainer = FinetuneRegression(
        model,
        optimizer,
        criterion,
        scaler,
        scheduler,
        distributed,
        target_scale = args.target_scale,
    )
    trainer.epoch = start_epoch
    trainer.step_count = step_count
    trainer.patience_counter = patience_counter

    ## Training loop
    # ---------------------------
    start_time = time.time()
    print(f'Starting training at {datetime.now()}')
    print(f'Training for {args.max_epochs} epochs')
    for epoch in range(start_epoch,args.max_epochs):
        trainer.train_epoch(train_loader,args)        
        trainer.validate(val_loader,args)
        print(f'Epoch: {epoch}')
        print(f'training time: {(time.time() - start_time)/60:.2f}min')

        if (args.local_rank == 0
            and (
                (epoch + 1) % args.log_interval == 0 # Save every log_interval epochs
                or epoch + 1 == args.max_epochs # Save at end of training
                or trainer.patience_counter == 0 # Save if best model
                or trainer.patience_counter == args.patience # Save if early stopping
            )
        ):
            if trainer.patience_counter == 0:
                print(f'Saving current best model at epoch {epoch}')
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
            
        if trainer.patience_counter == args.patience:
            print(f'Early stopping at epoch {epoch}')
            break

    if args.local_rank == 0:
        # save run metrics 
        torch.save(
            trainer.validation_metrics,
            os.path.join(args.output_dir, 'validation_metrics.pt')
        )
        summary_results = {}
        for key, value in trainer.validation_metrics.items():
            wandb.run.summary[f'{key}_last'] = value[-1]
            wandb.run.summary[f'{key}_min'] = torch.min(torch.tensor(value)).item()
            wandb.run.summary[f'{key}_max'] = torch.max(torch.tensor(value)).item()
            wandb.run.summary[f'{key}_mean_last_10'] = torch.mean(torch.tensor(value[-10:])).item()

    # Log training time
    print(f'Training finished at {datetime.now()}')
    print(f'Training took {time.time() - start_time}')
    
    # Terminate wandb run
    if args.local_rank == 0:
        run.finish()
    
def main(args):
    # Set up distributed training
    device, distributed = utils.ddp_utils.setup_distributed(args)
    # Create base output dir
    utils.setup_utils.setup_output_dir(args)
    # Loads the config file params into args and saves the config file 
    utils.setup_utils.setup_config(args)
    # Set random seeds
    utils.setup_utils.set_seed(args.seed)
    # Set benchmarking true for faster training
    torch.backends.cudnn.benchmark = True

    if isinstance(args.train_prcnt, list):
        prcnt_list = args.train_prcnt
        max_epochs = args.max_epochs
        output_dir = args.output_dir
        for train_prcnt in prcnt_list:
            args.output_dir = os.path.join(output_dir, f'tp_{train_prcnt}/')
            args.train_prcnt = train_prcnt

            if args.adjust_max_epochs:
                args.max_epochs = int(math.ceil(max_epochs / train_prcnt))

            if args.n_folds is not None:
                output_dir = args.output_dir
                for fold in range(args.n_folds):
                    args.train_set_json = f'{args.cv_folds_path}/fold_{fold}_train_files.json'
                    args.val_set_json = f'{args.cv_folds_path}/fold_{fold}_validation_files.json'
                    args.output_dir = os.path.join(output_dir, f'fold_{fold}/')
                    # Create fold output dir
                    setup_output(args)
                    finetune(args, device, distributed, fold)
            else:
                finetune(args, device, distributed)
    else:
        if args.adjust_max_epochs:
            args.max_epochs = int(math.ceil(args.max_epochs / args.train_prcnt))

        if args.n_folds is not None:
            output_dir = args.output_dir
            for fold in range(args.n_folds):
                args.train_set_json = f'{args.cv_folds_path}/fold_{fold}_train_files.json'
                args.val_set_json = f'{args.cv_folds_path}/fold_{fold}_validation_files.json'
                args.output_dir = os.path.join(output_dir, f'fold_{fold}/')
                # Create fold output dir
                setup_output(args)
                finetune(args, device, distributed, fold)
        else:
            finetune(args, device, distributed)
if __name__ == '__main__':
    args = parse_args()
    main(args)