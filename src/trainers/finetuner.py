import sys
import torch
from torch.amp import autocast
from src.trainers.trainer import Trainer
from collections import defaultdict

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    cohen_kappa_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    r2_score,
)

class Finetuner(Trainer):
    def __init__(
            self,
            model,
            optimizer,
            criterion,
            scaler,
            scheduler,
            distributed,
            label_map,
            ignore_index,
        ):
        super().__init__(
            model, optimizer, criterion, scaler, scheduler, distributed
        )
        self.validation_metrics = defaultdict(list)
        self.label_map = label_map
        self.ignore_index = ignore_index
        self.running_scores = []
        self.running_targets = []

    def forward(self, x, args):
        x,y = x
        y = y.cuda()
        x = x.cuda()
        with autocast('cuda', enabled=args.mixed_precision):
            if hasattr(args, 'use_sdpa'):
                y_hat = self.model(x, use_sdpa=args.use_sdpa)
            else:
                y_hat = self.model(x)
            loss = self.criterion(y_hat.transpose(1,2).squeeze(), y.squeeze())

        if torch.isnan(loss).any():
            print('Nan loss encountered, exiting...')
            print('labels',y)
            print('preds',y_hat)
            sys.exit(1)
        
        # Normalize loss to account for gradient accumulation
        loss = loss / args.grad_accumulation_steps

        # Track loss and diagnostics for logging
        self.running_loss.append(loss.item())
        self.running_scores.append(y_hat.detach().cpu())
        self.running_targets.append(y.cpu())

        return loss

    def log_losses(self, mode = 'train'):
        targets = torch.cat(self.running_targets).flatten()
        scores = torch.cat(self.running_scores)
        scores = scores.view(-1, scores.shape[-1])
        scores = scores.float().softmax(dim = -1)

        # Remove missing labels
        scores = scores[targets != self.ignore_index]
        targets = targets[targets != self.ignore_index]

        self.calc_metrics(targets, scores)

        self.running_scores = []
        self.running_targets = []
        return super().log_losses(mode=mode)

    def calc_metrics(self, targets, scores):
        preds = scores.argmax(dim=-1)
        labels = list(self.label_map.keys())
        average = 'macro'

        self.running_diagnostics['macro_f1'].append(
            f1_score(targets, preds, average=average, labels=labels)
        )
        self.running_diagnostics['macro_precision'].append(
            precision_score(targets, preds, average=average, labels=labels)
        )
        self.running_diagnostics['macro_recall'].append(
            recall_score(targets, preds, average=average, labels=labels)
        )
        self.running_diagnostics['balanced_accuracy'].append(
            balanced_accuracy_score(targets, preds)
        )
        self.running_diagnostics['cohens_kappa'].append(
            cohen_kappa_score(targets, preds)
        )
        if scores.shape[-1] == 2:
            # Handle that sklearn does not allow binary labels with two-dimentional scores
            neg_auc = roc_auc_score(1-targets, scores[:,0])
            pos_auc =  roc_auc_score(targets, scores[:,1])
            cw_auc = [neg_auc, pos_auc]
            self.running_diagnostics['macro_auc'].append(sum(cw_auc)/2)
        else:
            self.running_diagnostics['macro_auc'].append(
                roc_auc_score(
                    targets, scores, average=average, multi_class='ovo', labels=labels,
                )
            )
            cw_auc = roc_auc_score(
                targets, scores, average=None, multi_class='ovr', labels=labels
            )
        # Compute class wise metric
        cw_f1 = f1_score(targets, preds, average=None, labels=labels)
        cw_prec = precision_score(targets, preds, average=None, labels=labels)
        cw_rec = recall_score(targets, preds, average=None, labels=labels)

        for k in labels:
            self.running_diagnostics[f'{self.label_map[k]}_auc'].append(cw_auc[k])
            self.running_diagnostics[f'{self.label_map[k]}_f1'].append(cw_f1[k])
            self.running_diagnostics[f'{self.label_map[k]}_precision'].append(cw_prec[k])
            self.running_diagnostics[f'{self.label_map[k]}_recall'].append(cw_rec[k])

    def validate(self, val_loader, args):
        self.model.eval()
        self.running_loss = []
        self.running_scores = []
        self.running_targets = []
        self.running_diagnostics = defaultdict(list)

        with torch.no_grad():
            for x in val_loader:
                loss = self.forward(x, args)

        avg_loss = torch.tensor(self.running_loss).mean().item()
        print(f'Validation loss: {avg_loss}')
        if avg_loss < self.smallest_val_loss:
            self.smallest_val_loss = avg_loss

        # Gather losses and diagnostics from all GPUs
        self.log_losses('validation')
        # Add validation metrics to validation_metrics
        for key, value in self.log.items():
            if key == 'macro_f1':
                max_f1 = torch.tensor(self.validation_metrics[key]).max().item()
                if value > max_f1:
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            self.validation_metrics[key].append(value)
        self.log_wandb()

class FinetuneRegression(Trainer):
    def __init__(
            self,
            model,
            optimizer,
            criterion,
            scaler,
            scheduler,
            distributed,
            target_scale: float = 1.0,
        ):
        super().__init__(
            model, optimizer, criterion, scaler, scheduler, distributed
        )
        self.target_scale = target_scale
        self.validation_metrics = defaultdict(list)
        self.running_preds = []
        self.running_targets = []

    def forward(self, batch, args):
        x, y = batch
        y = y / self.target_scale
        y = y.cuda()
        x = x.cuda()
        with autocast('cuda', enabled=args.mixed_precision):
            if hasattr(args, 'use_sdpa'):
                y_hat = self.model(x, use_sdpa=args.use_sdpa)
            else:
                y_hat = self.model(x)
            
            if torch.isnan(y_hat).any():
                print('Nan prediction encountered, dropping it...')
                print('labels',y)
                print('preds',y_hat)
                y_hat = y_hat[~torch.isnan(y_hat)]
                y = y[~torch.isnan(y_hat)]
                print('labels',y)
                print('preds',y_hat)

            loss = self.criterion(y_hat, y)

        if torch.isnan(loss).any():
            print('Nan loss encountered, exiting...')
            print('labels',y)
            print('preds',y_hat)
            sys.exit(1)
        
        # Normalize loss to account for gradient accumulation
        loss = loss / args.grad_accumulation_steps

        # Track loss and diagnostics for logging
        self.running_loss.append(loss.item())
        self.running_preds.append(y_hat.detach().cpu())
        self.running_targets.append(y.cpu())

        return loss

    def log_losses(self, mode = 'train'):
        targets = torch.cat(self.running_targets).flatten()
        preds = torch.cat(self.running_preds).flatten()

        self.calc_metrics(targets, preds)

        self.running_preds = []
        self.running_targets = []
        return super().log_losses(mode=mode)

    def calc_metrics(self, targets, preds):
        preds = preds.float() * self.target_scale
        targets = targets.float() * self.target_scale
        # Report target and prediction distributions
        self.running_diagnostics['targets_median'].append(
            torch.median(targets)
        )
        self.running_diagnostics['targets_q25'].append(
            targets.quantile(0.25)
        )
        self.running_diagnostics['targets_q75'].append(
            targets.quantile(0.75)
        )
        self.running_diagnostics['preds_median'].append(
            torch.median(preds)
        )
        self.running_diagnostics['preds_q25'].append(
            preds.quantile(0.25)
        )
        self.running_diagnostics['preds_q75'].append(
            preds.quantile(0.75)
        )
        
        # Calculate metrics
        self.running_diagnostics['MAE'].append(
            mean_absolute_error(targets, preds)
        )
        # Get standard deviation of residuals
        residuals = preds - targets
        self.running_diagnostics['residuals_std'].append(
            residuals.std()
        )
        self.running_diagnostics['R2'].append(
            r2_score(targets, preds)
        )


    def validate(self, val_loader, args):
        self.model.eval()
        self.running_loss = []
        self.running_preds = []
        self.running_targets = []
        self.running_diagnostics = defaultdict(list)

        with torch.no_grad():
            for x in val_loader:
                loss = self.forward(x, args)

        avg_loss = torch.tensor(self.running_loss).mean().item()
        print(f'Validation loss: {avg_loss}')
        if avg_loss < self.smallest_val_loss:
            self.smallest_val_loss = avg_loss

        # Gather losses and diagnostics from all GPUs
        self.log_losses('validation')
        # Add validation metrics to validation_metrics
        for key, value in self.log.items():
            if key == 'R2':
                max_r2 = torch.tensor(self.validation_metrics[key]).max().item()
                if value > max_r2:
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            self.validation_metrics[key].append(value)
        self.log_wandb()
