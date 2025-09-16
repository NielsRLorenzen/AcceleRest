import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss.spectrogram_transform import BandAmplification

class SCGLoss(nn.Module): # V3
    def __init__(
        self,
        window_samples: int,
        sample_freq: float,
        std_cutoff: float = 0.015,
        jerk_band: tuple = (4, 14),
        pulse_band: tuple = (0.6, 1.77),
        reference_band: tuple = None,
        amplification: float = 1e3,
        downsample: int = 3,
        mode: str = 'jerks',
    ) -> None:
        super().__init__()

        self.pulse_band = pulse_band
        self.sample_freq = sample_freq
        self.amplification = amplification
        self.std_cutoff = std_cutoff
        self.mode = mode

        self.window_samples = window_samples
        self.nbins = self.window_samples//2 + 1
        self.center_freqs = torch.linspace(0, sample_freq/2, self.nbins)

        # FFT projections
        weighted_downsampling_scheme = []
        if reference_band is not None:
            weighted_downsampling_scheme.append([1, 1, [reference_band[0], reference_band[1]]])
        weighted_downsampling_scheme.append([amplification, downsample, [pulse_band[0], pulse_band[1]]])
        
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

        # Jerk filter vector
        lower = torch.argmin((self.center_freqs - jerk_band[0]).abs())
        upper = torch.argmin((self.center_freqs - jerk_band[1]).abs())
        jerk_filter = torch.zeros(self.nbins)
        jerk_filter[lower: upper + 1] = 1
        self.register_buffer('jerk_filter', jerk_filter)

    def patchwise_magnitudes(self, x):
        # Collapse windows into batch for FFT
        B, W, S = x.shape
        x = x.contiguous().view(-1, S)

        with torch.autocast(device_type="cuda", enabled=False):
            x_fft = torch.fft.rfft(x.float(), norm = 'ortho')

        x_fft = x_fft.view(B, W, self.nbins)

        x_mag = torch.sqrt(
            torch.clamp(x_fft.real**2 + x_fft.imag**2, min=1e-8)
        )

        return x_mag
    
    def get_jerks(self, x):
        # Undfold signal into windows
        x = x.unfold(-1, self.window_samples, self.window_samples)
        B, C, W, S = x.shape # Batch, Channels, Windows, Samples

        # Collapse windows and channels into batch for FFT
        x = x.contiguous().view(-1, S)

        # Apply jerk filter in freq space
        with torch.autocast(device_type="cuda", enabled=False):
            jerks = torch.fft.irfft(torch.fft.rfft(x.float()) * self.jerk_filter.float())
        jerks = jerks.view(B, C, W, S)

        return jerks
 
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
        ).nan_to_num().max(dim=1)[0]

        return (max_std < self.std_cutoff)

    def get_peak_loss(self, input_pwr, target_pwr):
        # Sum magnitudes across channel dim and square
        smooth_max_in = input_pwr.logsumexp(dim=-1)
        smooth_max_tgt = target_pwr.logsumexp(dim=-1)

        idx = torch.arange(input_pwr.shape[-1], device=input_pwr.device, dtype=input_pwr.dtype)
        smooth_argmax_in = (input_pwr.softmax(dim=-1) * idx.view(1, 1, input_pwr.shape[-1])).sum(dim=-1)
        smooth_argmax_tgt = (target_pwr.softmax(dim=-1) * idx.view(1, 1, input_pwr.shape[-1])).sum(dim=-1)

        # Calculate reggresion losses for the peak frequency
        peak_loss = F.mse_loss(smooth_max_in, smooth_max_tgt)
        peak_distance_loss = F.mse_loss(smooth_argmax_in, smooth_argmax_tgt)

        peak_strengt = target_pwr.max(dim=-1)[0] / target_pwr.quantile(0.8, dim=-1)
        peak_distance_loss = peak_distance_loss * peak_strengt
        return

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
        # Initialize diagnostics dict
        diagnostics = {}

        input = input.unfold(-1, self.window_samples, self.window_samples)
        if self.mode == 'jerks':
            # Calculate magnitudes over channel dim
            scg_input = torch.linalg.vector_norm(input, dim=1)
        elif self.mode == 'direct':
            scg_input = input.mean(dim=1)

        jerks_target = self.get_jerks(target)
        scg_target = torch.linalg.vector_norm(jerks_target, dim=1)
        
        # FFT power spectra
        input_mag = self.patchwise_magnitudes(scg_input)
        target_mag = self.patchwise_magnitudes(scg_target)

        # Downsample and amplify
        input_mag = self.input_filter(input_mag)
        target_mag = self.target_filter(target_mag)

        # Get frequency loss
        freq_loss = F.mse_loss(input_mag, target_mag, reduction = 'none')

        # Initialize mean divisor for freq_loss
        mean_divisor = torch.ones_like(freq_loss)

        # Get SD filter
        loss_filter = self.get_std_filter(target)

        # Report proportion of patches kept by std filter
        diagnostics['std_pcnt_kept'] = loss_filter.half().mean().item()

        if patch_mask is not None:
            # Reshape mask (B, num_patches) -> (B, num_patches, 1)
            patch_mask = patch_mask.to(dtype=freq_loss.dtype).unsqueeze(-1)
            loss_filter = loss_filter * patch_mask

        # Report proportion of patches kept by std filter and mask
        diagnostics['sd_&_mask_kept'] = loss_filter.half().mean().item()
        
        # Zero loss for unmasked and patches with high SD
        freq_loss *= loss_filter
        mean_divisor *= loss_filter

        # Report loss for each bin
        bin_wise_loss = torch.sum(freq_loss, dim = (0,1)).detach() / (mean_divisor[..., 0].sum() + eps)
        diagnostics.update({
            f'bin_{i}': bin_wise_loss[i].item() 
            for i 
            in range(len(bin_wise_loss))
        })

        # Calculate average loss
        freq_loss = freq_loss.sum() / (mean_divisor.sum() + eps)

        return freq_loss, diagnostics

class SCGLossV2(nn.Module):
    def __init__(
        self,
        window_samples: int,
        sample_freq: float,
        std_cutoff: float = 0.011,
        jerk_band: tuple = (4, 14),
        pulse_band: tuple = (0.6, 1.8),
        breathing_band: tuple = (0.13, 0.57),
        amplification: float = 1000,
        time_domain_weight: float = 0.0,
    ) -> None:
        super().__init__()

        self.pulse_band = pulse_band
        self.sample_freq = sample_freq
        self.amplification = amplification
        self.std_cutoff = std_cutoff
        self.time_domain_weight = time_domain_weight

        self.window_samples = window_samples
        self.nbins = self.window_samples//2 + 1
        self.center_freqs = torch.linspace(0, sample_freq/2, self.nbins)
        self.jerk_band = jerk_band

        ## FFT filters
        # lower = torch.argmin((self.center_freqs - 0.13).abs())
        # upper = torch.argmin((self.center_freqs - 0.6).abs())
        # nbins_breath = upper - lower + 1
        weighted_downsampling_scheme = [
            # [1, 1, [0, 0]], # DC band
            [1, 1, [breathing_band[0], breathing_band[1]]],
            [amplification, 1, [pulse_band[0], pulse_band[1]]]
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

        # Jerk filter vector
        lower = torch.argmin((self.center_freqs - self.jerk_band[0]).abs())
        upper = torch.argmin((self.center_freqs - self.jerk_band[1]).abs())
        jerk_filter = torch.zeros(self.nbins)
        jerk_filter[lower: upper] = 1
        self.register_buffer('jerk_filter', jerk_filter)

    def patchwise_magnitudes(self, x):
        # Collapse windows into batch for FFT
        B, W, S = x.shape
        x = x.contiguous().view(-1, S)

        x_fft = torch.fft.rfft(x, norm = 'forward')

        x_fft = x_fft.view(B, W, self.nbins)

        x_mag = torch.sqrt(
            torch.clamp(x_fft.real**2 + x_fft.imag**2, min=1e-8)
        )

        return x_mag
    
    def get_jerks(self, x):
        # Undfold signal into windows
        x = x.unfold(-1, self.window_samples, self.window_samples)
        B, C, W, S = x.shape # Batch, Channels, Windows, Samples

        # Collapse windows and channels into batch for FFT
        x = x.contiguous().view(-1, S)

        # Apply jerk filter in freq space
        jerks = torch.fft.irfft(torch.fft.rfft(x) * self.jerk_filter)
        jerks = jerks.view(B, C, W, S)

        return jerks
 
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
        ).nan_to_num().max(dim=1)[0]

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
        # Calculate magnitudes over channel dim
        input = input.unfold(-1, self.window_samples, self.window_samples)
        scg_input = torch.linalg.vector_norm(input, dim=1)
        jerks_target = self.get_jerks(target)
        scg_target = torch.linalg.vector_norm(jerks_target, dim=1)
        
        # Perform windowing and fft
        input_pwr = self.patchwise_magnitudes(scg_input).square()
        target_pwr = self.patchwise_magnitudes(scg_target).square()

        # Downsample and amplify
        input_mag = self.input_filter(input_mag)
        target_mag = self.target_filter(target_mag)

        # Get frequency loss
        freq_loss = F.mse_loss(input_mag, target_mag, reduction = 'none')

        # Initialize mean divisor for freq_loss
        mean_divisor = torch.ones_like(freq_loss)

        if patch_mask is not None:
            # Reshape mask (B, num_patches) -> (B, num_patches, 1)
            patch_mask = patch_mask.to(dtype=freq_loss.dtype).unsqueeze(-1)
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
        bin_wise_loss = torch.sum(freq_loss, dim = (0,1)).detach() / (mean_divisor.sum() + eps)
        diagnostics.update({
            f'bin_{i}': bin_wise_loss[i].item() 
            for i 
            in range(len(bin_wise_loss))
        })

        # Calculate average loss
        freq_loss = freq_loss.sum() / (mean_divisor.sum() + eps)

        return freq_loss, diagnostics

class SCGLossV1(nn.Module):
    def __init__(
        self,
        window_samples: int,
        sample_freq: float,
        std_cutoff: float = 0.011,
        jerk_band: tuple = (4, 14),
        pulse_band: tuple = (0.6, 1.7),
        amplification: float = 1000,
        time_domain_weight: float = 0.0,
        half_res: bool = False,
    ) -> None:
        super().__init__()

        self.pulse_band = pulse_band
        self.sample_freq = sample_freq
        self.amplification = amplification
        self.std_cutoff = std_cutoff
        self.time_domain_weight = time_domain_weight
        self.half_res = half_res

        self.window_samples = window_samples
        self.nbins = self.window_samples//2 + 1
        self.center_freqs = torch.linspace(0, sample_freq/2, self.nbins)
        
        # Get FFT filters
        jerk_idx = self.get_bin_idx(*jerk_band, self.center_freqs)
        jerk_band_filter = self.get_filter_vec(*jerk_idx, self.nbins)

        pulse_idx = self.get_bin_idx(*pulse_band, self.center_freqs)
        pulse_band_filter = self.get_filter_vec(*pulse_idx, self.nbins)
        pulse_band_filter *= self.amplification

        self.register_buffer('jerk_band_filter', jerk_band_filter)
        self.register_buffer('pulse_band_filter', pulse_band_filter)

    def get_bin_idx(self, low_cut, high_cut, center_freqs):
        lower = torch.argmin((center_freqs - low_cut).abs())
        upper = torch.argmin((center_freqs - high_cut).abs())
        return lower, upper
    
    def get_filter_vec(self, lower, upper, nbins):
        filter_vec = torch.zeros(nbins)
        filter_vec[lower: upper] = 1
        return filter_vec

    def fft_filter(self, input, filter_vec):
        return torch.fft.irfft(torch.fft.rfft(input) * filter_vec)

    def scg_target(self, target):
        # Unfold signal into windows
        target_unfold = target.unfold(-1, self.window_samples, self.window_samples)
        B, C, W, S = target_unfold.shape # Batch, Channels, Windows, Samples
        target_unfold = target_unfold.transpose(1, 2) # Swap channels and windows

        # Collapse windows and channels into batch for FFT
        target_unfold = target_unfold.contiguous().view(-1, S)

        # Apply jerk filter in freq sapce
        jerks = self.fft_filter(target_unfold, self.jerk_band_filter)
        jerks = jerks.view(B * W, C, S)

        # Calculate magnitudes over channel dim
        jerks = torch.linalg.vector_norm(jerks, dim=1)

        # Apply pulse filter and amplify
        scg_fft = torch.fft.rfft(
            jerks,
            norm = 'forward',
        ) * self.pulse_band_filter

        # Calculate spectrum
        scg_magnitude_spectrum = scg_fft.abs().view(B, W, self.nbins)

        # Reconstitute signal
        scg = torch.fft.irfft(scg_fft, norm = 'forward')
        scg = scg.view(B, W, self.window_samples)
        
        return scg_magnitude_spectrum, scg
      
    def scg_input(self, scg_hat):
        scg_hat = scg_hat.unfold(-1, self.window_samples, self.window_samples)
        B, W, S = scg_hat.shape # Batch, Windows, Samples
        scg_hat = scg_hat.view(-1, S)

        scg_hat_magnitude_spectrum = torch.fft.rfft(scg_hat, norm = 'forward').abs()
        scg_hat_magnitude_spectrum = scg_hat_magnitude_spectrum.view(B, W, self.nbins)
        scg_hat = scg_hat.view(B, W, S)

        return scg_hat_magnitude_spectrum, scg_hat
    
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
        ).nan_to_num().max(dim=1)[0]

        return (max_std < self.std_cutoff)

    def downsample(self, spectrum):
        if (spectrum.shape[-1] % 2) != 0:
            spectrum = torch.cat(
                [spectrum.unfold(-1,2,2).mean(-1), 
                spectrum[...,-1:]],
                dim=-1,
            )
        else:
            spectrum = spectrum.unfold(-1,2,2).mean(-1)
        return spectrum

    def forward(
        self,
        scg_hat: torch.Tensor,
        target: torch.Tensor,
        patch_mask: torch.Tensor = None, 
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, dict]:
        '''
        Args:
            scg_hat (torch.Tensor): Reconstructed scg signal, shape (B, (1,) S)
            target (torch.Tensor): Target signal to contstruct scg from,
            shape (B, C, S)
            patch_mask (torch.Tensor, optional): Bool mask (B, num_patches)
        Returns:
            loss (torch.Tensor) 
        '''
        # Squeeze redundant channel dim
        scg_hat = scg_hat.squeeze()
        
        # Perform windowing and fft on scg_hat
        scg_hat_magnitude_spectrum, scg_hat = self.scg_input(scg_hat)

        # Extract seismocardiogram from target
        scg_magnitude_spectrum, scg = self.scg_target(target)

        # Separate and sum bins lower and higher than pulse band
        lower, upper = self.get_bin_idx(*self.pulse_band, self.center_freqs)
        
        low_sum = scg_hat_magnitude_spectrum[:, :, :lower].sum(-1, keepdim=True)
        high_sum = scg_hat_magnitude_spectrum[:, :, upper:].sum(-1, keepdim=True)

        # Bins above low and high cutoffs should hit 0
        # low_loss = F.mse_loss(low_sum, torch.zeros_like(low_sum), reduction='none')
        # high_loss = F.mse_loss(high_sum, torch.zeros_like(high_sum), reduction='none')
        
        scg_hat_magnitude_spectrum = scg_hat_magnitude_spectrum[:, :, lower: upper]
        scg_magnitude_spectrum = scg_magnitude_spectrum[:, :, lower: upper]
  
        if self.half_res:
            scg_hat_magnitude_spectrum = self.downsample(scg_hat_magnitude_spectrum)
            scg_magnitude_spectrum = self.downsample(scg_magnitude_spectrum)

        # Get loss for the pulse band
        freq_loss = F.mse_loss(
            scg_hat_magnitude_spectrum,
            scg_magnitude_spectrum,
            reduction = 'none',
        )

        # # # Combine frequency losses
        # freq_loss = torch.cat([low_loss, spectral_loss, high_loss], dim=-1)

        # Initialize mean divisor for freq_loss
        mean_divisor = torch.ones_like(freq_loss)

        if patch_mask is not None:
            # Reshape mask (B, num_patches) -> (B, num_patches, 1)
            patch_mask = patch_mask.to(dtype=freq_loss.dtype).unsqueeze(-1)
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
        bin_wise_loss = torch.sum(freq_loss, dim = (0,1)).detach() / (mean_divisor.sum() + eps)
        diagnostics.update({
            f'bin_{i}': bin_wise_loss[i].item() 
            for i 
            in range(len(bin_wise_loss))
        })

        # Calculate average loss
        freq_loss = freq_loss.sum() / (mean_divisor.sum() + eps)

        # Loss on time domain signals
        if self.time_domain_weight > 0.0:
            time_domain_loss = F.mse_loss(scg_hat, scg, reduction='none')
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