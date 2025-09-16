import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def patchwise_fft(
    x: torch.Tensor,
    patch_size: int,
    overlap: int = 0,
    norm: bool = False,
    subtract_mean: bool = False,
    window: bool = False,
    fft_size: int = None,
) -> torch.Tensor:
    '''
    Computes FFT on each patch of the input signal with optional overlap,
    mean subtraction, and windowing (Hann).

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, T)
        patch_size (int): Length of each patch (not including overlap)
        fft_size (int, optional): Length of FFT (default: patch + 2 * overlap)
        overlap (int): Number of overlapping samples to include from adjacent patches
        norm (bool): Whether to normalize the FFT magnitudes to signal length.
        subtract_mean (bool): Subtract patch mean before FFT
        window (bool): Apply Hann window to each patch before FFT

    Returns:
        Tensor of shape (B, C, num_patches, fft_size//2 + 1), complex-valued
    '''
    if overlap < 0 or type(overlap) != int:
        raise ValueError(f"overlap should be 0 or a postive integer got {overlap}")

    B, C, T = x.shape
    stride = patch_size
    full_patch_size = patch_size + 2 * overlap

    # Pad for edge patches
    x_padded = F.pad(x, (overlap, overlap), mode='constant')  # (B, C, T + 2*overlap)

    # Extract patches: shape (B, C, num_patches, full_patch_size)
    patches = x_padded.unfold(-1, size=full_patch_size, step=stride)

    # Apply Hann window
    if window:
        win = torch.hann_window(
            full_patch_size,
            periodic=True,
            device=patches.device,
            dtype=patches.dtype,
        )
        patches = patches * win.view(1, 1, 1, -1)

    if fft_size is None:
        fft_size = full_patch_size

    # Optional zero-padding if fft_size > full_patch_size
    if fft_size > full_patch_size:
        pad_len = fft_size - full_patch_size
        patches = F.pad(patches, (pad_len // 2, pad_len - pad_len // 2))

    # Reshape to (B*C*num_patches, time)
    patches = patches.contiguous().view(-1, patches.size(-1))

    if subtract_mean:
        patch_mean = patches.mean(dim=-1, keepdim=True)
        patches = patches - patch_mean
        if not norm:
            patch_mean *= fft_size 
    else:
        patch_mean = None

    # FFT
    with torch.autocast("cuda", enabled=False):
        fft_out = torch.fft.rfft(
            patches.float(), 
            n=fft_size, 
            norm='forward' if norm else 'backward',
        )

    # Replace DC bin with mean if requested
    if subtract_mean:
        fft_out = torch.cat((patch_mean.to(fft_out.dtype), fft_out), dim=-1)

    # Reshape back to (B, C, num_patches, freq_bins)
    num_patches = (T + 2 * overlap - full_patch_size) // stride + 1
    fft_out = fft_out.view(B, C, num_patches, -1)

    return fft_out