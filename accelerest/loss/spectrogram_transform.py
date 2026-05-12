import torch
import torch.nn as nn
import math

class BandAmplification(nn.Module):
    def __init__(
        self,
        nbins: int,
        sample_freq: float,
        weighted_downsampling_scheme: list[tuple[float, int, tuple[float, float]]],
        leftover_handling: str = 'none',
    ):
        '''
        This class applies a downsampling transform with a customisable per
        band weighting and downsampling with all schemes specified in Hz bands.

        Args:
            nbins (int): Number of input frequency bins (e.g., from rFFT).

            sample_freq (float): Sampling frequency in Hz.

            weighted_downsampling_scheme (list[tuple[weight, int, (float, float)]]):
                List of (weight, downsample_factor, (start_hz, stop_hz)).
                Where weight is the weight on the output bins of that band,
                so a weight of 1 is a mean, and a weight of downsample_factor is
                a sum, and downsample_factor is a positive integer => 1 which
                specifies the output frequency resolution of that band so that: 

                new_freq_res = old_freq_res*downsample_factor

                A band where start_hz == stop_hz will specify a single
                bin centered as closely as possible to that value.

            leftover_handling (str):
                Specify how to handle bands where downsample_factor is
                not divisble by the number of original bins. Options
                are:
                    'none': Throw an error if downsample_factor is not
                    divisible by the number of bins between start_hz and
                    stop_hz.

                    'extend': Include more bins beyond stop_hz so that the 
                    effective stop_hz becomes larger.

                    'keep': Remaining bins are simply downsampled to a
                    single bin. This ensures stop_hz remains the same but
                    increases the effective resolution of that output bin 
                    and relative weight on each of the original bins.
                    
                    'discard': Discard remaining bins so that the effective
                    stop_hz becomes smaller.

                    'merge': Include remaining bins in the otherwise
                    second-to-last output bin. This ensures stop_hz
                    remains the same but reduces the effective resolution
                    of that output bin and relative weight on each of 
                    the original bins.

        '''
        super().__init__()
        self.nbins = nbins
        self.sample_freq = sample_freq

        self.center_freqs = torch.linspace(0, sample_freq/2, nbins)
        self.freq_res = sample_freq / (2 * (nbins - 1))
        if leftover_handling not in ['none', 'extend', 'keep', 'discard', 'merge']:
            raise ValueError("leftover_handling must be one of 'none', 'extend', 'keep', 'discard', or 'merge'")
        self.leftover_handling = leftover_handling

        # Build downsampling matrix
        if weighted_downsampling_scheme is not None:
            self.down_mat = self._get_downsample_matrix(weighted_downsampling_scheme)
        else:
            self.down_mat = torch.eye(nbins)

        # Remove rows with all zero weights
        nonzero_rows = self.down_mat.abs().sum(dim=-1) != 0
        self.down_mat = self.down_mat[nonzero_rows]

        self.out_bins = self.down_mat.shape[0]
        self.down_proj = nn.Linear(nbins, self.out_bins, bias=False)
        self.down_proj.weight = nn.Parameter(self.down_mat, requires_grad=False)

    def _hz_to_bin(self, hz_value: float, mode: str = 'start') -> int:
        if hz_value > self.center_freqs.max() + self.freq_res:
            raise ValueError(
                f'Supplied hz_value {hz_value} is more than freq_res '
                f'{self.freq_res} above nyquist {self.center_freqs.max()}'
            )

        bin_index = torch.argmin((self.center_freqs - hz_value).abs())
        return bin_index

    def _get_downsample_matrix(
        self,
        scheme: list[tuple[float, int, tuple[float, float]]]
    ) -> torch.Tensor:
        '''
        Converts a downsampling scheme into a downsampling matrix.

        Args:
            scheme: list[tuple[float, int, tuple[float, float]]]
        '''
        weight_rows = []
        for weight, factor, (start_hz, stop_hz) in scheme:
            start_bin = self._hz_to_bin(start_hz)
            end_bin = self._hz_to_bin(stop_hz) + 1
            if start_bin >= end_bin:
                raise ValueError(f'start bin >= end bin for band {start_hz}-{stop_hz} Hz')

            for i in range(start_bin, end_bin, factor):
                row = torch.zeros(self.nbins)
                k = i + factor
                k_next = k + factor
                if k > end_bin:
                    if self.leftover_handling == 'none':
                        raise ValueError(
                            f'Number of bins between {start_hz} and {stop_hz} '
                            f'({end_bin-start_bin}) is not divisible by {factor}'
                        )
                    elif self.leftover_handling == 'extend':
                        # Keep k as is and include the extra bins in the output bin
                        pass
                    elif self.leftover_handling == 'keep':
                        # Shorten the current output bin to end_bin
                        k = end_bin
                    elif self.leftover_handling == 'discard':
                        # Drop remaining bins
                        break
                    elif self.leftover_handling == 'merge':
                        # Handled on previous iteration
                        break
                elif (k_next > end_bin) and (self.leftover_handling == 'merge'):
                    # Include remaining bins in current output bin
                    k = end_bin

                row[i: k] = weight/(k-i)
                weight_rows.append(row)
        
        return torch.stack(weight_rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): Magnitude spectrum of shape (..., nbins)

        Returns:
            torch.Tensor: Projected and optionally weighted spectrum (..., out_bins)
        '''
        return self.down_proj(x)