import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedMSELoss(nn.Module):
    '''This class implements a patch-wise mse loss that only considers 
    patches that are marked as True in the patch_mask.

    Note:
    Only supports "mean" or "sum" reduction.
    Masking flattens the tensor along the batch and channel dimensions
    so it would need to be reshaped to (B, C, n_masked, window_samples) 
    to be returned without taking the mean or sum.
    '''
    def __init__(
        self, 
        window_samples: int,
        reduction='mean',
    ) -> None:
        super().__init__()
        self.window_samples = window_samples
        self.reduction = reduction

    def forward(self, input, target, patch_mask) -> torch.Tensor:
        return masked_mse_loss(
            input,
            target,
            patch_mask,
            self.window_samples, 
            self.reduction
        )

def masked_mse_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        patch_mask: torch.Tensor,
        window_samples: int = 300,
        reduction: str = 'mean',
    ) -> torch.Tensor:
    '''This function implements a patch-wise mse loss that only considers
    patches that are marked as True in the patch_mask.

    Args:
        input (torch.Tensor): The input tensor (B,C,L).
        target (torch.Tensor): The target tensor (B,C,L).
        patch_mask (torch.Tensor): A boolean tensor that, marks which 
            patches to consider (B, num_patches).
        window_samples (int): The size of the patches.
        reduction (str): The reduction type. Can be "mean" or "sum". 
            Default is "mean".
    '''

    # Compute the mse loss and unfold it to get the patch-wise loss
    mse = F.mse_loss(input, target, reduction='none') 

    # (B,C,L) -> (B,C,num_patches,window_samples)
    unfolded_loss = mse.unfold(-1, window_samples, window_samples) 

    # Expand the mask to match the channel dimension
    patch_mask = patch_mask.unsqueeze(1).expand(-1,input.shape[1],-1)

    # Apply the mask
    masked_loss = unfolded_loss[patch_mask]
    
    if reduction == 'mean':
        return torch.mean(masked_loss), {}
    elif reduction == 'sum':
        return torch.sum(masked_loss), {}
    else:
        raise ValueError(
            ('Invalid reduction type. '
            f'Expected "mean" or "sum", got {reduction}')
        )