import torch
import torch.nn as nn

class PatchEmbedding1D(nn.Module):
    def __init__(self, patch_size: int, num_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Linear(patch_size * num_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Patchify the input tensor x along the last dimension and 
        project each patch to the embedding space.

        E.g., if x has shape: 
            (batch_size, channels, n_sampels),
        the patched tensor will have shape: 
            (batch_size, num_patches, channels, patch_size).
        and the output will have shape:
            (batch_size, num_patches, embed_dim).
        
        If n_samples % patch_size != 0, the last patch will be discarded.

        Parameters:
            x (torch.Tensor): 
                Input tensor of shape: (batch_size, channels, nsamples).
            
        Returns:
            torch.Tensor: 
                Patchified tensor of shape:
                (batch_size, num_patches, embed_dim).
        '''
        # Shape to (batch_size, channels, num_patches, patch_size)
        x  = x.unfold(-1, self.patch_size, self.patch_size)

        # Shape to (batch_size, num_patches, channels, patch_size)
        x = x.permute(0, 2, 1, 3)

        # Shape to (batch_size, num_patches, channels * patch_size)
        x = x.flatten(-2, -1)

        # Shape to (batch_size, num_patches, embed_dim)
        x = self.embed(x)

        return x

class ConvPatchEmbedding1D(nn.Module):
    def __init__(
            self, 
            patch_size: int,
            num_channels: int,
            embed_dim: int,
            kernel_size: int,
            num_depthwise: int = 1,
            ):       
        super().__init__()
        self.patch_size = patch_size
        # Depthwise convolution (Temporal/channel-wise)
        self.dwconv = nn.Conv1d(
            in_channels = num_channels,
            out_channels = num_channels * num_depthwise,
            kernel_size = kernel_size,
            stride = kernel_size // 2,
            padding = kernel_size // 2,
            groups = num_channels,
        )
        Lout = int(((patch_size + kernel_size - 1) // stride) + 1)
        Lout2 = int(((Lout + kernel_size - 1) // stride) + 1)
        # Normalization        
        norm = nn.RMSNorm(dim)
        # Pointwise convolution
        self.pwconv = nn.Conv2d(
            in_channels = num_channels * num_depthwise,
            out_channels = num_channels * num_depthwise,
            kernel_size = 1,
        )

        self.dwconv2 = nn.Conv1d(
            in_channels = num_channels * num_depthwise,
            out_channels = num_channels * num_depthwise,
            kernel_size = kernel_size,
            stride = kernel_size // 2,
            padding = kernel_size // 2,
            groups = num_channels * num_depthwise,
        )

        self.pwconv2 = nn.Conv2d(
            in_channels = num_channels * num_depthwise,
            out_channels = num_channels * num_depthwise,
            kernel_size = 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Patchify the input tensor x along the last dimension and 
        project each patch to the embedding space.

        E.g., if x has shape: 
            (batch_size, channels, n_sampels),
        the patched tensor will have shape: 
            (batch_size, num_patches, channels, patch_size).
        and the output will have shape:
            (batch_size, num_patches, embed_dim).
        
        If n_samples % patch_size != 0, the last patch will be discarded.

        Parameters:
            x (torch.Tensor): 
                Input tensor of shape: (batch_size, channels, nsamples).
            
        Returns:
            torch.Tensor: 
                Patchified tensor of shape:
                (batch_size, num_patches, embed_dim).
        '''
        # Shape (batch_size, channels, num_patches, patch_size)
        x  = x.unfold(-1, self.patch_size, self.patch_size)

        # Shape (batch_size, num_patches, channels, patch_size)
        x = x.permute(0, 2, 1, 3)

        # Shape (batch_size * num_patches, channels, patch_size)
        x = x.flatten(0, 1)

        # Shape (batch_size * num_patches, channels, patch_size // (kernel_size//2))
        x = self.dwconv(x)
        x = self.norm(x)

        # Shape (batch_size, num_patches, embed_dim)

        return x

class InversePatchEmbedding1D(nn.Module):
    def __init__(self, patch_size: int, num_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.linear = nn.Linear(embed_dim, patch_size * num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Inverse operation of PatchEmbedding1D. It takes the 
        embedded patches and reconstructs the original input tensor.

        Parameters:
            x (torch.Tensor): 
                Input tensor of shape: (batch_size, num_patches, embed_dim).
            
        Returns:
            torch.Tensor: 
                Reconstructed tensor of shape: (batch_size, channels, n_samples).
        '''
        # Shape (batch_size, num_patches, patch_size * num_channels)
        x = self.linear(x)

        # Shape (batch_size, num_patches, num_channels, patch_size)
        x = x.view(x.size(0), x.size(1), self.num_channels, self.patch_size)

        # Shape (batch_size, num_channels, num_patches, patch_size)
        x = x.permute(0, 2, 1, 3)

        # Shape (batch_size, num_channels, n_samples)
        x = x.contiguous().view(x.size(0), self.num_channels, -1)

        return x