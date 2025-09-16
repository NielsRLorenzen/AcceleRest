import time
import sys
from collections import defaultdict
import wandb

import torch

from src.trainers import utils
from src.utils import ddp_utils

class Trainer():
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scaler: torch.cuda.amp.GradScaler,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        distributed: bool,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.scheduler = scheduler
        self.distributed = distributed

        self.step_count = 0
        self.epoch = 0

        self.running_loss = list()
        self.running_diagnostics = defaultdict(list)

        self.smallest_val_loss = float('inf')
        self.patience_counter = 0

        self.log = {}

    def forward(self):
        raise NotImplementedError(
            'Forward method not implemented in base Trainer class'
            )

    def forward_backward(self, x, args):
        loss = self.forward(x, args)

        # Backward pass
        self.scaler.scale(loss).backward()

    def step(self,args):
        # Unscale the gradients before clipping
        self.scaler.unscale_(self.optimizer)

        # Clip gradients to prevent exploding gradients
        if args.grad_norm_max is not None:
            l2_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                args.grad_norm_max,
            )
        else:
            l2_norm = utils.get_grad_norm(self.model.parameters())
        self.log['l2_norm'] = l2_norm

        # Step the optimizer
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Zero the gradients
        self.optimizer.zero_grad()

        # Update the scheduler
        self.log['lr'] = self.optimizer.param_groups[0]['lr']
        if self.scheduler is not None:
            self.scheduler.step()

        self.step_count += 1

    def log_wandb(self):
        # Log to wandb if initialized
        if wandb.run is not None:
            wandb.log(self.log,step=self.step_count)
        self.log = {}

    def log_losses(self, mode='train'):
        # Calculate average loss and diagnostics from running lists
        avg_loss = torch.tensor(self.running_loss).mean().item()
        avg_diagnostics = {
            key: (
                torch.stack(self.running_diagnostics[key], dim=-1).mean(dim=-1).tolist()
                if torch.is_tensor(self.running_diagnostics[key][0])
                else torch.tensor(self.running_diagnostics[key]).mean().item()
            )
            for key 
            in self.running_diagnostics.keys()
        }

        # Gather losses and diagnostics from all GPUs
        if self.distributed:
            avg_loss = ddp_utils.all_reduce_mean(avg_loss)
            for key in avg_diagnostics.keys():
                avg_diagnostics[key] = ddp_utils.all_reduce_mean(
                    avg_diagnostics[key]
                )

        # Add losses and diagnostics to log        
        self.log[f'{mode}/loss'] = avg_loss
        self.log.update(
            {
                f'{mode}/{key}': (
                    wandb.Histogram(avg_diagnostics[key])
                    if torch.is_tensor(avg_diagnostics[key])
                    else avg_diagnostics[key]
                )
                for key 
                in avg_diagnostics.keys()
            }
        )

        # Reset running loss and diagnostics
        self.running_loss = []
        self.running_diagnostics = defaultdict(list)

        return avg_loss, avg_diagnostics

    def train_epoch(self, train_loader, args):
        if self.distributed: 
            # Needed for shuffling in distributed training
            train_loader.sampler.set_epoch(self.epoch)

        self.model.train()
        for idx,x in enumerate(train_loader):
            # Update optimizer every grad_accumulation_steps
            # Note: Updating on the last batch can lead to a smaller 
            # effective batch size for that step. So we only update
            # if we have accumulated the desired number of gradients
            if ((idx+1) % args.grad_accumulation_steps == 0): 
                self.forward_backward(x,args)
                self.step(args)

                # Log the loss and diagnostics
                if self.step_count % args.log_interval == 0:
                    self.log['epoch'] = self.epoch
                    # Gather losses and diagnostics from all GPUs if using DDP
                    _ = self.log_losses('train')
                    self.log_wandb()
            else:
                # Accumulate gradients
                if self.distributed:
                    with self.model.no_sync():
                        self.forward_backward(x,args)
                else:
                    self.forward_backward(x,args)
        self.epoch += 1

    def validate(self, val_loader, args):
        self.model.eval()
        self.running_loss = []
        self.running_diagnostics = defaultdict(list)

        with torch.no_grad():
            for x in val_loader:
                loss = self.forward(x, args)
                    
        # Gather losses and diagnostics from all GPUs
        avg_loss, avg_diagnostics = self.log_losses('validation')
        self.log_wandb()
        print(f'Validation loss: {avg_loss}')

        if avg_loss < self.smallest_val_loss:
            self.smallest_val_loss = avg_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
