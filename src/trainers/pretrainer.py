import sys
import torch
from torch.amp import autocast
from src.trainers.trainer import Trainer

class Pretrainer(Trainer):
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn.Module,
            scaler: torch.cuda.amp.GradScaler,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            distributed: bool,
        ):
        super().__init__(
            model,
            optimizer,
            criterion,
            scaler,
            scheduler,
            distributed,
        )

    def forward(self, x, args):
        # Needed since multiple samples are retrieved per file in the batch
        x = x.view(-1,x.shape[2],x.shape[3])
        x = x.cuda()

        with autocast('cuda', enabled=args.mixed_precision):
            x_hat, mask_idx = self.model(x, use_sdpa=args.use_sdpa, mask_ratio=args.mask_ratio)
            if type(x_hat) == tuple:
                loss, diagnostics = self.criterion(*x_hat, target=x, patch_mask=mask_idx)
            else:
                loss, diagnostics = self.criterion(x_hat, target=x, patch_mask=mask_idx)
        
        if torch.isnan(loss).any():
            if torch.isnan(x).any():
                print('nan values in target')
            for key in diagnostics.keys():
                print(f'{key}: {diagnostics[key]}')
            print('Nan loss encountered, exiting...')
            sys.exit(1)
        
        # Normalize loss to account for gradient accumulation
        loss = loss / args.grad_accumulation_steps

        # Track loss and diagnostics for logging
        self.running_loss.append(loss.item())
        for key in diagnostics.keys():
            self.running_diagnostics[key].append(
                diagnostics[key]/args.grad_accumulation_steps
            )
        return loss