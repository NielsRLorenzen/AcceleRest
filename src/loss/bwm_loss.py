import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss.spectrogram_transform import BandAmplification

class BWMLoss(nn.Module):
    def __init__(
        self,
        window_samples: int,
        sample_freq: float,
        std_cutoff: float = 0.015,
        breathing_band: tuple = (0.1, 0.6),
        reference_band: tuple = None,#(0, 0),
        amplification: float = 1e3,
        downsample: int = 1,
    ) -> None:
        super().__init__()

        self.breathing_band = breathing_band
        self.sample_freq = sample_freq
        self.amplification = amplification
        self.std_cutoff = std_cutoff

        self.window_samples = window_samples
        self.nbins = self.window_samples//2 + 1
        self.center_freqs = torch.linspace(0, sample_freq/2, self.nbins)
        
        ## FFT projections
        weighted_downsampling_scheme = []
        if reference_band is not None:
            weighted_downsampling_scheme.append([1, 1, [reference_band[0], reference_band[1]]])
        weighted_downsampling_scheme.append(
            [
                amplification,
                downsample,
                [breathing_band[0], breathing_band[1]]
            ]
        )

        self.target_filter = BandAmplification(
            nbins = self.nbins,
            sample_freq = sample_freq,
            weighted_downsampling_scheme = weighted_downsampling_scheme,
            leftover_handling = 'none',
        )
        unweighted_downsampling_scheme = [
            [1, ds, band] 
            for (weight, ds, band) 
            in weighted_downsampling_scheme
        ]
        self.input_filter = BandAmplification(
            nbins = self.nbins,
            sample_freq = sample_freq,
            weighted_downsampling_scheme = unweighted_downsampling_scheme,
            leftover_handling = 'none',
        )

    def patchwise_magnitudes(self, x):
        # Undfold signal into windows
        x_unfold = x.unfold(-1, self.window_samples, self.window_samples)
        B, C, W, S = x_unfold.shape # Batch, Channels, Windows, Samples

        # Collapse windows and channels into batch for FFT
        x_unfold = x_unfold.contiguous().view(-1, S)

        with torch.autocast(device_type="cuda", enabled=False):
            x_fft = torch.fft.rfft(x_unfold.float(), norm = 'ortho')

        x_fft = x_fft.view(B, C, W, self.nbins)

        x_mag = torch.sqrt(
            torch.clamp(x_fft.real**2 + x_fft.imag**2, min=1e-8)
        )

        return x_mag
      
    def get_std_filter(self, target):
        target_patches = target.unfold(
            -1,
            size=self.window_samples,
            step=self.window_samples,
        )

        # Added nan_to_num (NaN -> 0) to avoid error with constant input)
        max_std = target_patches.std(
            dim=-1,
            keepdim=True,
        ).nan_to_num().max(dim=1, keepdim=True)[0]

        std_filter = (max_std < self.std_cutoff)
        return std_filter

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        patch_mask: torch.Tensor = None, 
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, dict]:
        '''
        Args:
            breathing_hat (torch.Tensor): Reconstructed breathing signal, shape (B, C, S)
            target (torch.Tensor): Target signal to contstruct breathing from,
            shape (B, C, S)
            patch_mask (torch.Tensor, optional): Bool mask (B, num_patches)
        Returns:
            loss (torch.Tensor) 
        '''
        # Perform windowing and fft
        input_mag = self.patchwise_magnitudes(input)
        target_mag = self.patchwise_magnitudes(target)

        # Downsample and amplify
        input_mag = self.input_filter(input_mag)
        target_mag = self.target_filter(target_mag)

        # Get frequency loss
        freq_loss = F.mse_loss(input_mag, target_mag, reduction = 'none')

        # Initialize mean divisor for mask tracking 
        mean_divisor = torch.ones_like(freq_loss)

        # Get SD filter
        loss_filter = self.get_std_filter(target)

        # Initialize diagnostics dict
        diagnostics = {}

        # Report proportion of patches kept by std filter
        diagnostics['std_pcnt_kept'] = loss_filter.half().mean().item()

        if patch_mask is not None:
            # Reshape mask (B, num_patches) -> (B, 1, num_patches, 1)
            patch_mask = patch_mask.to(dtype=freq_loss.dtype).unsqueeze(1).unsqueeze(-1)
            loss_filter = loss_filter * patch_mask

        # Report proportion of patches kept by std filter and mask
        diagnostics['sd_&_mask_kept'] = loss_filter.half().mean().item()

        # Zero loss for unmasked and patches with high SD
        freq_loss *= loss_filter
        mean_divisor *= loss_filter

        # Report loss for each bin
        bin_wise_loss = torch.sum(freq_loss, dim = (0,1,2)).detach() / (mean_divisor.sum() + eps)
        diagnostics.update({
            f'bin_{i}': bin_wise_loss[i].item() 
            for i 
            in range(len(bin_wise_loss))
        })

        # Calculate average loss
        freq_loss = freq_loss.sum() / (mean_divisor.sum() + eps)

        return freq_loss, diagnostics



class BreathingLossV3(nn.Module):
    def __init__(
        self,
        window_samples: int,
        sample_freq: float,
        std_cutoff: float = 0.011,
        breathing_band: tuple = (0.13, 0.6),
        amplification: float = 1000,
        entropy_weight: float = 1,
    ) -> None:
        super().__init__()

        self.breathing_band = breathing_band
        self.sample_freq = sample_freq
        self.amplification = amplification
        self.std_cutoff = std_cutoff
        self.time_domain_weight = time_domain_weight

        self.window_samples = window_samples
        self.nbins = self.window_samples//2 + 1
        self.center_freqs = torch.linspace(0, sample_freq/2, self.nbins)
        
        ## FFT filters
        weighted_downsampling_scheme = [
            [1, 1, [0, 0]], # DC band
            [amplification, 1, [breathing_band[0], breathing_band[1]]]
        ]
        self.target_filter = BandAmplification(
            nbins = self.nbins,
            sample_freq = sample_freq,
            weighted_downsampling_scheme = weighted_downsampling_scheme,
            leftover_handling = 'none',
        )
        unweighted_downsampling_scheme = [
            [1, ds, band] 
            for (weight, ds, band) 
            in weighted_downsampling_scheme
        ]
        self.input_filter = BandAmplification(
            nbins = self.nbins,
            sample_freq = sample_freq,
            weighted_downsampling_scheme = unweighted_downsampling_scheme,
            leftover_handling = 'none',
        )

        self.entropy_weight = entropy_weight
        if entropy_weight > 0:
            # Get number of bins produced by weighting and downsampling
            num_out_bins = self.target_filter.down_proj.out_features - 1

            # Calculate bound on entropy with a uniform distribution
            unif = torch.ones(num_out_bins)/num_out_bins
            self.h_unif = -torch.sum(unif * torch.log(unif + 1e-8))
    
    def patchwise_magnitudes(self, x):
        # Undfold signal into windows
        x_unfold = x.unfold(-1, self.window_samples, self.window_samples)
        B, C, W, S = x_unfold.shape # Batch, Channels, Windows, Samples

        # Collapse windows and channels into batch for FFT
        x_unfold = x_unfold.contiguous().view(-1, S)

        # Apply breathing filter
        x_fft = torch.fft.rfft(x_unfold, norm = 'forward')

        x_fft = x_fft.view(B, C, W, self.nbins)

        x_mag = torch.sqrt(
            torch.clamp(x_fft.real**2 + x_fft.imag**2, min=1e-8)
        )

        return x_mag
      
    def get_std_filter(self, target):
        target_patches = target.unfold(
            -1,
            size=self.window_samples,
            step=self.window_samples,
        )

        # Added nan_to_num (NaN -> 0) to avoid error with constant input)
        max_std = target_patches.std(
            dim=-1,
            keepdim=True,
        ).nan_to_num().max(dim=1, keepdim=True)[0]

        return (max_std < self.std_cutoff)

    def get_entropy_weights(self, mag, std_filter):
        #  Due to the max smoothing applied high std patches need to be masked prior to calculation
        mag_filt = mag * std_filter
        # Sum over channel dim and calc power
        # Doing this after applying the target filter is essential so the DC bin does not dominate
        # Taking the square-of-sums rather than the sum-of-squares was found to give a cleaner entropy signal
        target_power = mag_filt.sum(dim=1, keepdim=True)**2

        # Calculate spectral entropy
        p = torch.softmax(target_power, dim=-1)
        h = -torch.sum(p * torch.log(p + 1e-8), dim=-1)

        # Varies between 1 and self.entropy_weight + 1 with low-entropy pathces having higher weights
        h_w = (h - self.h_unif).abs()/self.h_unif
        h_w *= self.entropy_weight
        h_w += 1
        h_w = F.max_pool1d(h_w, kernel_size=3, stride=1, padding=1)
        h_w = h_w.unsqueeze(-1)
        return h_w

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        patch_mask: torch.Tensor = None, 
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, dict]:
        '''
        Args:
            breathing_hat (torch.Tensor): Reconstructed breathing signal, shape (B, C, S)
            target (torch.Tensor): Target signal to contstruct breathing from,
            shape (B, C, S)
            patch_mask (torch.Tensor, optional): Bool mask (B, num_patches)
        Returns:
            loss (torch.Tensor) 
        '''
        # Perform windowing and fft
        input_mag = self.patchwise_magnitudes(input)
        target_mag = self.patchwise_magnitudes(target)

        # Downsample and amplify
        input_mag = self.input_filter(input_mag)
        target_mag = self.target_filter(target_mag)
        
        # Get frequency loss
        freq_loss = F.mse_loss(input_mag, target_mag, reduction = 'none')

        # Initialize mean divisor for freq_loss
        mean_divisor = torch.ones_like(freq_loss)

        if patch_mask is not None:
            # Reshape mask (B, num_patches) -> (B, 1, num_patches, 1)
            patch_mask = patch_mask.to(dtype=freq_loss.dtype).unsqueeze(1).unsqueeze(-1)
            freq_loss *= patch_mask
            mean_divisor *= patch_mask

        # Filter out periods with high SD
        std_filter = self.get_std_filter(target)
        freq_loss *= std_filter
        mean_divisor *= std_filter

        # Initialize diagnostics dict
        diagnostics = {}

        # Report proportion of pathces kept by std filter
        diagnostics['std_pcnt_kept'] = std_filter.half().mean().item()

        if self.entropy_weight > 0:
            # Varies between 1 and self.entropy_weight + 1 with low-entropy pathces having higher weights
            h_w = self.get_entropy_weights(target_mag, std_filter)
            freq_loss *= h_w

        # Report loss for each bin
        bin_wise_loss = torch.sum(freq_loss, dim = (0,1,2)).detach() / (mean_divisor.sum() + eps)
        diagnostics.update({
            f'bin_{i}': bin_wise_loss[i].item() 
            for i 
            in range(len(bin_wise_loss))
        })

        # Calculate average loss
        freq_loss = freq_loss.sum() / (mean_divisor.sum() + eps)

        return freq_loss, diagnostics

class BreathingLossV2(nn.Module):
    def __init__(
        self,
        window_samples: int,
        sample_freq: float,
        std_cutoff: float = 0.011,
        breathing_band: tuple = (0.13, 0.6),
        amplification: float = 1000,
        time_domain_weight: float = 0.0,
    ) -> None:
        super().__init__()

        self.breathing_band = breathing_band
        self.sample_freq = sample_freq
        self.amplification = amplification
        self.std_cutoff = std_cutoff
        self.time_domain_weight = time_domain_weight

        self.window_samples = window_samples
        self.nbins = self.window_samples//2 + 1
        self.center_freqs = torch.linspace(0, sample_freq/2, self.nbins)
        
        ## FFT filters
        weighted_downsampling_scheme = [
            [1, 1, [0, 0]], # DC band
            [amplification, 1, [breathing_band[0], breathing_band[1]]]
        ]
        self.target_filter = BandAmplification(
            nbins = self.nbins,
            sample_freq = sample_freq,
            weighted_downsampling_scheme = weighted_downsampling_scheme,
            leftover_handling = 'none',
        )
        unweighted_downsampling_scheme = [
            [1, ds, band] 
            for (weight, ds, band) 
            in weighted_downsampling_scheme
        ]
        self.input_filter = BandAmplification(
            nbins = self.nbins,
            sample_freq = sample_freq,
            weighted_downsampling_scheme = unweighted_downsampling_scheme,
            leftover_handling = 'none',
        )
    
    def patchwise_magnitudes(self, x):
        # Undfold signal into windows
        x_unfold = x.unfold(-1, self.window_samples, self.window_samples)
        B, C, W, S = x_unfold.shape # Batch, Channels, Windows, Samples

        # Collapse windows and channels into batch for FFT
        x_unfold = x_unfold.contiguous().view(-1, S)

        # Apply breathing filter
        x_fft = torch.fft.rfft(x_unfold, norm = 'forward')

        x_fft = x_fft.view(B, C, W, self.nbins)

        x_mag = torch.sqrt(
            torch.clamp(x_fft.real**2 + x_fft.imag**2, min=1e-8)
        )

        return x_mag
      
    def get_std_filter(self, target):
        target_patches = target.unfold(
            -1,
            size=self.window_samples,
            step=self.window_samples,
        )

        # Added nan_to_num (NaN -> 0) to avoid error with constant input)
        max_std = target_patches.std(
            dim=-1,
            keepdim=True,
        ).nan_to_num().max(dim=1, keepdim=True)[0]

        return (max_std < self.std_cutoff)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        patch_mask: torch.Tensor = None, 
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, dict]:
        '''
        Args:
            breathing_hat (torch.Tensor): Reconstructed breathing signal, shape (B, C, S)
            target (torch.Tensor): Target signal to contstruct breathing from,
            shape (B, C, S)
            patch_mask (torch.Tensor, optional): Bool mask (B, num_patches)
        Returns:
            loss (torch.Tensor) 
        '''
        # Perform windowing and fft
        input_mag = self.patchwise_magnitudes(input)
        target_mag = self.patchwise_magnitudes(target)

        # Downsample and amplify
        input_mag = self.input_filter(input_mag)
        target_mag = self.target_filter(target_mag)
        
        # Get frequency loss
        freq_loss = F.mse_loss(input_mag, target_mag, reduction = 'none')

        # Initialize mean divisor for freq_loss
        mean_divisor = torch.ones_like(freq_loss)

        if patch_mask is not None:
            # Reshape mask (B, num_patches) -> (B, 1, num_patches, 1)
            patch_mask = patch_mask.to(dtype=freq_loss.dtype).unsqueeze(1).unsqueeze(-1)
            freq_loss *= patch_mask
            mean_divisor *= patch_mask

        # Filter out periods with high SD
        std_filter = self.get_std_filter(target)
        freq_loss *= std_filter
        mean_divisor *= std_filter

        # Initialize diagnostics dict
        diagnostics = {}

        # Report proportion of pathces kept by std filter
        diagnostics['std_pcnt_kept'] = std_filter.half().mean().item()

        # Report loss for each bin
        bin_wise_loss = torch.sum(freq_loss, dim = (0,1,2)).detach() / (mean_divisor.sum() + eps)
        diagnostics.update({
            f'bin_{i}': bin_wise_loss[i].item() 
            for i 
            in range(len(bin_wise_loss))
        })

        # Calculate average loss
        freq_loss = freq_loss.sum() / (mean_divisor.sum() + eps)

        return freq_loss, diagnostics


class BreathingLossV1(nn.Module):
    def __init__(
        self,
        window_samples: int,
        sample_freq: float,
        std_cutoff: float = 0.011,
        breathing_band: tuple = (0.13, 0.6),
        amplification: float = 1000,
        time_domain_weight: float = 0.0,
    ) -> None:
        super().__init__()

        self.breathing_band = breathing_band
        self.sample_freq = sample_freq
        self.amplification = amplification
        self.std_cutoff = std_cutoff
        self.time_domain_weight = time_domain_weight

        self.window_samples = window_samples
        self.nbins = self.window_samples//2 + 1
        self.center_freqs = torch.linspace(0, sample_freq/2, self.nbins)
        
        # Get FFT filter
        breath_idx = self.get_bin_idx(*breathing_band, self.center_freqs)
        breathing_band_filter = self.get_filter_vec(*breath_idx, self.nbins)

        self.register_buffer('breathing_band_filter', breathing_band_filter)

    def get_bin_idx(self, low_cut, high_cut, center_freqs):
        lower = torch.argmin((center_freqs - low_cut).abs())
        upper = torch.argmin((center_freqs - high_cut).abs())
        return lower, upper
    
    def get_filter_vec(self, lower, upper, nbins):
        filter_vec = torch.zeros(nbins)
        filter_vec[lower: upper] = 1 * self.amplification
        # keep dc band
        filter_vec[0] = 1
        return filter_vec

    def breath_target(self, target):
        # Undfold signal into windows
        target_unfold = target.unfold(-1, self.window_samples, self.window_samples)
        B, C, W, S = target_unfold.shape # Batch, Channels, Windows, Samples

        # Collapse windows and channels into batch for FFT
        target_unfold = target_unfold.contiguous().view(-1, S)

        # Apply breathing filter
        breathing_fft = torch.fft.rfft(
            target_unfold,
            norm = 'forward',
        ) * self.breathing_band_filter

        breathing_magnitude_spectrum = breathing_fft.abs().view(B, C, W, self.nbins)

        # Reconstitute signal
        breathing = torch.fft.irfft(breathing_fft, norm = 'forward')
        breathing = breathing.view(B, C, W, self.window_samples)
        
        return breathing_magnitude_spectrum, breathing
      
    def breath_input(self, breathing_hat):
        breathing_hat = breathing_hat.unfold(-1, self.window_samples, self.window_samples)
        B, C, W, S = breathing_hat.shape # Batch, Windows, Samples
        breathing_hat = breathing_hat.view(-1, S)

        breathing_hat_magnitude_spectrum = torch.fft.rfft(breathing_hat, norm = 'forward').abs()
        breathing_hat_magnitude_spectrum = breathing_hat_magnitude_spectrum.view(B, C, W, self.nbins)
        breathing_hat = breathing_hat.view(B, C, W, S)

        return breathing_hat_magnitude_spectrum, breathing_hat
    
    def get_std_filter(self, target):
        target_patches = target.unfold(
            -1,
            size=self.window_samples,
            step=self.window_samples,
        )

        # Added nan_to_num (NaN -> 0) to avoid error with constant input)
        max_std = target_patches.std(
            dim=-1,
            keepdim=True,
        ).nan_to_num().max(dim=1, keepdim=True)[0]

        return (max_std < self.std_cutoff)

    def forward(
        self,
        breathing_hat: torch.Tensor,
        target: torch.Tensor,
        patch_mask: torch.Tensor = None, 
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, dict]:
        '''
        Args:
            breathing_hat (torch.Tensor): Reconstructed breathing signal, shape (B, C, S)
            target (torch.Tensor): Target signal to contstruct breathing from,
            shape (B, C, S)
            patch_mask (torch.Tensor, optional): Bool mask (B, num_patches)
        Returns:
            loss (torch.Tensor) 
        '''
        # Perform windowing and fft on breathing_hat
        breathing_hat_magnitude_spectrum, breathing_hat = self.breath_input(breathing_hat)

        # Extract seismocardiogram from target
        breathing_magnitude_spectrum, breathing = self.breath_target(target)

        # Separate and sum bins lower and higher than breathing band
        lower, upper = self.get_bin_idx(*self.breathing_band, self.center_freqs)

        body_pos_hat = breathing_hat_magnitude_spectrum[..., :1]
        body_pos = breathing_magnitude_spectrum[..., :1]

        breathing_hat_magnitude_spectrum = breathing_hat_magnitude_spectrum[..., lower: upper]
        breathing_magnitude_spectrum = breathing_magnitude_spectrum[..., lower: upper]

        breathing_hat_magnitude_spectrum = torch.cat(
            [body_pos_hat, breathing_hat_magnitude_spectrum], dim=-1
        )
        breathing_magnitude_spectrum = torch.cat(
            [body_pos, breathing_magnitude_spectrum], dim=-1
        )
        
        # Get frequency loss
        freq_loss = F.mse_loss(
            breathing_hat_magnitude_spectrum,
            breathing_magnitude_spectrum,
            reduction = 'none',
        )

        # Initialize mean divisor for freq_loss
        mean_divisor = torch.ones_like(freq_loss)

        if patch_mask is not None:
            # Reshape mask (B, num_patches) -> (B, 1, num_patches, 1)
            patch_mask = patch_mask.to(dtype=freq_loss.dtype).unsqueeze(1).unsqueeze(-1)
            freq_loss *= patch_mask
            mean_divisor *= patch_mask

        # Filter out periods with high SD
        std_filter = self.get_std_filter(target)
        freq_loss *= std_filter
        mean_divisor *= std_filter

        # Initialize diagnostics dict
        diagnostics = {}

        # Report proportion of pathces kept by std filter
        diagnostics['std_pcnt_kept'] = std_filter.half().mean().item()

        # Report loss for each bin
        bin_wise_loss = torch.sum(freq_loss, dim = (0,1,2)).detach() / (mean_divisor.sum() + eps)
        diagnostics.update({
            f'bin_{i}': bin_wise_loss[i].item() 
            for i 
            in range(len(bin_wise_loss))
        })

        # Calculate average loss
        freq_loss = freq_loss.sum() / (mean_divisor.sum() + eps)

        # Loss on time domain signals
        if self.time_domain_weight > 0.0:
            time_domain_loss = F.mse_loss(breathing_hat, breathing, reduction='none')
            mean_divisor = torch.ones_like(time_domain_loss)

            if patch_mask is not None:
                mean_divisor *= patch_mask
                time_domain_loss *= patch_mask

            mean_divisor *= std_filter    
            time_domain_loss *= std_filter

            time_domain_loss = time_domain_loss.sum() / (mean_divisor.sum() + eps) 
            combined_loss = freq_loss + time_domain_loss * self.time_domain_weight
            diagnostics.update({
                'freq_loss': freq_loss.item(),
                'td_loss': time_domain_loss.item(),
                })
            return combined_loss, diagnostics

        else:
            return freq_loss, diagnostics