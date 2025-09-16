import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from src.models.roformer import RoFormerClassifier

class AccDecomp(nn.Module):
    def __init__(
        self,
        sample_freq: float,
        fft_window_samples: int,
        fft_window_step: int = None,
        breathing_band: tuple = (0.1, 0.6),
        pulse_band: tuple = (0.6, 1.8),
        jerk_band: tuple = (4, 14),
    ) -> None:
        super().__init__()
        self.breathing_band = breathing_band
        self.pulse_band = pulse_band
        self.jerk_band = jerk_band

        self.sample_freq = sample_freq
        self.window_samples = fft_window_samples
        self.window_step = fft_window_samples if fft_window_step is None else fft_window_step
        self.overlap = self.window_samples - self.window_step
        self.nbins = self.window_samples//2 + 1
        self.center_freqs = torch.linspace(0, sample_freq/2, self.nbins)
        
        # Get FFT filters
        breath_idx = self._get_bin_idx(*breathing_band, self.center_freqs)
        breath_band_filter = self._get_filter_vec(*breath_idx, self.nbins)

        jerk_idx = self._get_bin_idx(*jerk_band, self.center_freqs)
        jerk_band_filter = self._get_filter_vec(*jerk_idx, self.nbins)

        pulse_idx = self._get_bin_idx(*pulse_band, self.center_freqs)
        pulse_band_filter = self._get_filter_vec(*pulse_idx, self.nbins)

        self.register_buffer('jerk_band_filter', jerk_band_filter)
        self.register_buffer('pulse_band_filter', pulse_band_filter)
        self.register_buffer('breath_band_filter', breath_band_filter)

    def _get_bin_idx(self, low_cut, high_cut, center_freqs):
        lower = torch.argmin((center_freqs - low_cut).abs())
        upper = torch.argmin((center_freqs - high_cut).abs())
        return lower, upper

    def _get_filter_vec(self, lower, upper, nbins):
        filter_vec = torch.zeros(nbins)
        filter_vec[lower: upper] = 1
        return filter_vec
    
    def get_jerks(self, acc):
        acc = acc.unfold(-1, self.window_samples, self.window_step)
        B, C, W, S = acc.shape

        # Stack windows into first dim for FFT
        acc = acc.contiguous().view(-1, S)
        
        # Apply jerk band filter in freq space
        jerks = torch.fft.irfft(torch.fft.rfft(acc) * self.jerk_band_filter)
        jerks = jerks.view(B, C, W, S)
        return jerks

    def scg_fft(self, acc):
        # Get unfolded jerk signal
        jerks = self.get_jerks(acc)
        
        # Get Euclidean norm across channels for raw scg
        scg = torch.linalg.vector_norm(jerks, dim=1)
        B, W, S = scg.shape

        # Apply pulse band filter to raw scg
        scg_fft = torch.fft.rfft(
            scg.view(-1, S), norm='forward'
        ) * self.pulse_band_filter
        
        # Return signal and power spectrum
        scg = torch.fft.irfft(scg_fft, norm='forward').view(B, W, S)
        # scg = self.reconstitute_signal(scg, self.window_step)
        scg_pwr = scg_fft.abs().view(B, W, self.nbins).square()
        return scg, scg_pwr

    def breath_fft(self, acc):
        acc = acc.unfold(-1, self.window_samples, self.window_step)
        B, C, W, S = acc.shape

        # Stack windows into first dim for FFT
        acc = acc.contiguous().view(-1, S)

        breath_fft = torch.fft.rfft(
            acc, norm='forward'
        ) * self.breath_band_filter

        breathing = torch.fft.irfft(breath_fft, norm='forward').view(B, C, W, S)
        # breathing = self.reconstitute_signal(breathing, self.window_step)
        breath_pwr = breath_fft.abs().view(B, C, W, self.nbins).sum(dim=1).square()
        return breathing, breath_pwr

    def reconstitute_signal(self, windowed_signal):
        *BC, W, S = windowed_signal.shape
        signal = windowed_signal[..., self.overlap//2: (S-self.overlap//2)].reshape(*BC, -1)
        return signal
    
    def spectral_flatness(self, power):
        rel_bins = power.sum(dim=(0,1)) > 0
        num_out_bins = rel_bins.sum()

        # Get spectral flatness
        power = power[..., rel_bins]
        geo_m = power.log().mean(dim=-1).exp()
        arith_m = power.mean(dim=-1)
        sfm = geo_m/arith_m
        return sfm

    def forward(self, acc):
        activity = (torch.linalg.vector_norm(acc, dim=1) - 1).abs()
        
        acc_pad = F.pad(acc, (self.overlap, self.overlap), mode = 'reflect')
        
        scg, scg_pwr = self.scg_fft(acc_pad)
        scg = self.reconstitute_signal(scg)

        breathing, breath_pwr = self.breath_fft(acc_pad)
        breathing = self.reconstitute_signal(breathing)

        signals = torch.cat([
            acc,
            activity.unsqueeze(1),
            scg.unsqueeze(1),
            breathing,
        ], dim = 1)
        
        return signals

class DWSeparableConv(nn.Module):
    def __init__(
        self,
        num_in_channels: int,
        num_out_channels: int,
        kernel_size: int,
        num_depthwise: int = 1,
    ):
        super().__init__() 

        # Depthwise convolution
        self.dwconv = nn.Conv1d(
            in_channels=num_in_channels,
            out_channels=num_in_channels * num_depthwise,
            kernel_size=kernel_size,
            padding='same',
            groups=num_in_channels,
        )

        # Pointwise convolution
        self.pwconv = nn.Conv1d(
            in_channels=num_in_channels * num_depthwise,
            out_channels=num_out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        return x

class SeparableResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_depthwise: int = 1,
        dropout: float = 0.0,
    ):
        '''
            bn-relu-dwsconv-bn-relu-dwsconv
           /                              \
        --x-------------------------------(+)-->
        '''
        super().__init__()

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = DWSeparableConv(out_channels, out_channels, kernel_size, num_depthwise)
        
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DWSeparableConv(out_channels, out_channels, kernel_size, num_depthwise)

        self.dropout = nn.Dropout(dropout)    

    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        return x + identity

class SeparableResNet(nn.Module):
    def __init__(
        self,
        layer_cfg: list[tuple],
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential()
        for i, params in enumerate(layer_cfg):
            self.feature_extractor.add_module(
                f"layer{i}",
                SeparableResNet.make_layer(*params)
            )

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        kernel_size,
        num_depthwise,
        n_blocks,
        downsample_factor,
    ) -> nn.Sequential:
        '''   bn-relu-dwsconv-bn-relu-dwsconv
             /                              \
        x--conv ----------------------------(+)->(..^n)->-bn-relu->
            ^
        (Project + downsample)
        '''
        modules = []
        # Projection        
        modules.append(
            nn.Conv1d(
                in_channels, out_channels, 
                kernel_size = downsample_factor,
                stride = downsample_factor,
            )
        )
        for _ in range(n_blocks):
            modules.append(
                SeparableResBlock(
                    in_channels = out_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    num_depthwise = num_depthwise,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels)) 
        modules.append(nn.ReLU(inplace=True))
        return nn.Sequential(*modules)
    
    def forward(self, x):
        return self.feature_extractor(x)

class AcceleroNet(nn.Module):
    def __init__(
        self,
        patch_size: int,
        sample_freq: float,
        num_classes: int,
        embed_dim: int,
        acc_decomp: bool = True,
        kernel_size: int = 5,
        num_lstm_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = 8 if acc_decomp else 3
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        if acc_decomp:
            self.decomp = AccDecomp(
                sample_freq = sample_freq,
                fft_window_samples = patch_size,
                fft_window_step = 2*patch_size//3,
                breathing_band = (0.1, 0.6),
                pulse_band = (0.6, 1.8),
                jerk_band = (4, 14),
            )

        # [(in_ch, out_ch, kernel_size, numdepthwise, nblocks, downsamplefactor)]
        layer_cfg = [
                (self.in_channels, 64, kernel_size, 3, 2, 1),
                (64, 128, kernel_size, 1, 1, 3),
                (128, 256, kernel_size, 1, 1, 3),
                (256, 512, kernel_size, 1, 1, 5),
                (512, 256, kernel_size, 1, 1, 5),
                (256, embed_dim, None, None, 0, 4),
            ]

        self.feature_extractor = SeparableResNet(layer_cfg)

        self.lstm = nn.LSTM(
            embed_dim,
            embed_dim,
            num_lstm_layers,
            dropout = dropout,
            batch_first = True,
            bidirectional = True,
        )
        
        self.lstm_norm = nn.RMSNorm(embed_dim*2)
        # self.classifier = nn.Linear(embed_dim*2, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim*4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim*4, embed_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim*2, num_classes),
        )

    def forward(self, x):
        # 0. Add acc derived signals [acc, activity, scg, breathing]
        if hasattr(self, 'decomp'):
            x = self.decomp(x)
        B, C, L = x.shape
        # 1. Patchify
        # (B, C, L) -> (B, S, C, patch_size)
        x = x.unfold(-1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3)
        B, S = x.shape[:2]
        # Flatten sequence into batch dim
        x = x.reshape(-1, C, self.patch_size)
        # 2. Feature extraction
        # (B*S, in_channels, patch_size) -> (B, S, embed_dim)
        x = self.feature_extractor(x)
        x = x.view(B, S, self.embed_dim)
        # 3. Sequence modelling
        x, _ = self.lstm(x)
        y_hat = self.classifier(x)
        return y_hat

class AccelFormer(RoFormerClassifier):
    def __init__(
        self,
        mode: str,
        sample_freq: float,
        acc_decomp: bool,
        num_classes: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        num_layers: int,
        num_lstm_layers: int,
        max_seq_len: int,
        encoder_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ):
        in_channels = 8 if acc_decomp else 3
        super().__init__(
            mode, num_classes, patch_size, in_channels, embed_dim, num_heads, mlp_ratio,
            num_layers, num_lstm_layers, max_seq_len, encoder_dropout, head_dropout,
        )
        if acc_decomp:
            self.decomp = AccDecomp(
                sample_freq = sample_freq,
                fft_window_samples = patch_size,
                fft_window_step = 2*patch_size//3,
                breathing_band = (0.1, 0.6),
                pulse_band = (0.6, 1.8),
                jerk_band = (4, 14),
            )

    def forward(self, x, use_sdpa = False):
        if hasattr(self, 'decomp'):
            with torch.no_grad():
                x = self.decomp(x)
        return super().forward(x, use_sdpa)


if __name__ == '__main__':
    patch_size = 900
    inp = torch.randn((5, 3, patch_size*10))
    # model = AcceleroNet(
    #     patch_size = patch_size,
    #     sample_freq = 30,
    #     num_classes = 3,
    #     acc_decomp=False,
    #     embed_dim = 256,
    #     kernel_size = 5,
    #     num_lstm_layers = 2,
    #     dropout = 0.1,
    # )
    model = AccelFormer(
        sample_freq = 30,
        acc_decomp = True,
        num_classes = 3,
        patch_size = 900,
        embed_dim = 128,
        num_heads = 4,
        mlp_ratio = 2,
        num_layers = 6,
        num_lstm_layers = 0,
        max_seq_len = 256,
        encoder_dropout = 0.0,
        head_dropout = 0.0,
    )
    out = model(inp)
    print(out.shape)
    # for mname, module in model.named_parameters():
    #     print(mname)
        # for pname, param in module.named_parameters():
        #     print(f'{mname}.{pname}: requires_grad={param.requires_grad}')
    num_params = sum(
            [p.numel() for p in model.parameters() if p.requires_grad]
    )
    print(num_params)
    # out = model(inp)
    # # print(math.exp(math.log(900)/n_blocks))
    # print(out.shape)