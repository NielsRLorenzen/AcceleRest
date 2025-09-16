import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.loss.spectrogram_transform import BandAmplification
from src.loss.fft_utils import patchwise_fft

class BandAmplificationLoss(nn.Module):
    def __init__(
        self,
        patch_size: int,
        sample_freq: float,
        weighted_downsampling_scheme: list[tuple[float, int, tuple[float, float]]],
        log_transform: bool = False,
        loss_norm: str = 'L2',
        std_cutoff: float = None,
        std_cutoff_type: str = None,
        invert_cutoff: bool = False,
        weight_clamp: float = 0.0,
        patchwise_fft_kwargs: dict = {},
        reduction: str = 'mean',
    ):
        '''
        Combines patch-wise FFT magnitude loss with custom 
        frequency downsampling and weighting.

        Args:
            patch_size (int): Length of each patch in time.
            weighted_downsampling_scheme (list[tuple[float, int, tuple[float, float]]]): 
                Downsampling plan.
            loss_norm (str): 'L1' or 'L2'.
            std_cutoff (float): Cutoff in std between stationary and 
                non-stationary windows. 
            std_cutoff_type (str): How to apply the std cutoff to weigh the
                patch-wise losses. One of 'soft', 'hard' or 'decay'.
                    'soft': sigmoid(10/(2*std_cutoff) * -1*std - 10)
                    'hard': std < std_cutoff
                    'decay': min(1, exp(max_std / (std_cutoff/math.log(2))))
            invert_cutoff (bool): Weight patches with high std rather than low
            patchwise_fft_kwargs (dict): keyword arguments passed to patchwise_fft
        '''
        super().__init__()
        self.patch_size = patch_size
        self.log_transform = log_transform
        self.loss_norm = loss_norm
        self.std_cutoff = std_cutoff
        self.std_cutoff_type = std_cutoff_type
        self.invert_cutoff = invert_cutoff
        self.weight_clamp = weight_clamp
        self.patchwise_fft_kwargs = patchwise_fft_kwargs
        self.reduction = reduction

        nbins = patch_size // 2 + 1
        self.target_downsampler = BandAmplification(
            nbins = nbins,
            sample_freq = sample_freq,
            weighted_downsampling_scheme = weighted_downsampling_scheme,
            leftover_handling = 'none',
        )
        unweighted_downsampling_scheme = [
            [1, ds, band] 
            for (weight, ds, band) 
            in weighted_downsampling_scheme
        ]
        self.input_downsampler = BandAmplification(
            nbins = nbins,
            sample_freq = sample_freq,
            weighted_downsampling_scheme = unweighted_downsampling_scheme,
            leftover_handling = 'none',
        )

    def patchwise_magnitudes(self, x):
        # Compute patch-wise FFT magnitude
        x_fft = patchwise_fft(x, self.patch_size, **self.patchwise_fft_kwargs)
        
        # Compute the frequency magnitudes
        x_mag = torch.sqrt(
            torch.clamp(x_fft.real**2 + x_fft.imag**2, min=1e-8)
        )
        return x_mag
        
    def apply_loss_norm(self, input, target):
        if self.loss_norm == 'L1':
            loss = F.l1_loss(input, target, reduction='none')
        elif self.loss_norm == 'L2':
            loss = F.mse_loss(input, target, reduction='none')
        else:
            raise ValueError(f'Invalid loss_norm: {self.loss_norm}')
        return loss

    def get_std_scaling(self, target):
        target_patches = target.unfold(-1, size=self.patch_size, step=self.patch_size)
        # Added nan_to_num (NaN -> 0) to avoid std error with constant input)
        max_std = target_patches.std(dim=-1, keepdim=True).nan_to_num().max(dim=1, keepdim=True)[0]
        # print('max_std shape \n', max_std.shape)
        H = self.std_cutoff
        if self.std_cutoff_type == 'soft':
            # Flipped sigmoid with weight ~1 at max_std < H, 0.5 at 2H, 0 at 3H
            K = 10
            scaling = 1 / (1 + torch.exp(K/(2*H) * max_std - K))
        elif self.std_cutoff_type == 'decay':
            # Exponential decay with 1 at max_std < H, 0.5 at 2H, and 0 ~ 10H
            scaling = torch.clamp(
                torch.exp((max_std - H) / (H/math.log(2))), max=1
            )
        elif self.std_cutoff_type == 'hard':
            # Binary cutoff
            scaling = (max_std < H)
        else:
            raise ValueError(f'std_cutoff_type must be one of "soft", "decay", or "hard", got {self.std_cutoff_type}')
        if self.invert_cutoff:
            scaling = 1 - scaling.to(target.dtype)
        # Optionally clamp scaling for stability when loss is sparse
        scaling = torch.clamp(scaling, min=self.weight_clamp)
        return scaling

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        patch_mask: torch.Tensor = None, 
    ) -> tuple[torch.Tensor, dict]:
        '''
        Args:
            input (torch.Tensor): Input signal, shape (B, C, T)
            target (torch.Tensor): Target signal, shape (B, C, T)
            patch_mask (torch.Tensor, optional): Bool mask (B, num_patches)
        Returns:
            loss (torch.Tensor) 
        '''
        diagnostics = {}

        input_mag = self.patchwise_magnitudes(input)
        target_mag = self.patchwise_magnitudes(target)

        if self.log_transform:
            input_mag = torch.log10(torch.clamp(input_mag, min=1e-4))
            target_mag = torch.log10(torch.clamp(target_mag, min=1e-4))

        input_mag = self.input_downsampler(input_mag)
        target_mag = self.target_downsampler(target_mag)

        loss = self.apply_loss_norm(input_mag, target_mag)

        # Apply weighting and downsampling
        mean_divisor = torch.ones_like(loss)

        if patch_mask is not None:
            # Reshape mask (B, num_patches) -> (B, 1, num_patches, 1)
            patch_mask = patch_mask.to(dtype=loss.dtype).unsqueeze(1).unsqueeze(-1)
            loss *= patch_mask
            mean_divisor *= patch_mask
           
        if self.std_cutoff is not None:
            scaling = self.get_std_scaling(target).to(dtype=loss.dtype)
            diagnostics.update({
                'scaling_min': scaling.min().item(),
                'scaling_max': scaling.max().item(),
                'scaling_mean': scaling.mean().item(),
                # 'max_std': max_std.max().item(),
                # 'min_std': max_std.min().item(),
            })
            # print('scaling shape \n', scaling.shape)
            # print('mean scaling \n', scaling.mean())
            # print('num std < H / num patches \n', (max_std < H).sum()/torch.numel(max_std))
            loss *= scaling
            mean_divisor *= scaling
            
        # Report loss for each bin
        # bin_wise_loss = torch.sum(loss, dim = (0,1,2)).detach() / (mean_divisor.sum() + 1e-6)
        # diagnostics.update({
        #     f'bin_{i}': bin_wise_loss[i].item() 
        #     for i 
        #     in range(len(bin_wise_loss))
        # })
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.sum() / (mean_divisor.sum() + 1e-6)
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(f'Invalid reduction {reduction}')

        return loss, diagnostics