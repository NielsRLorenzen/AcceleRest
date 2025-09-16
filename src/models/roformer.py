import torch
import torch.nn as nn

from src.models.utils import compute_rotations
from src.models.layers import (
    Attention,
    FeedForwardSwiGLU,
    PatchEmbedding1D,
    InversePatchEmbedding1D,
    AttentionPool,
    GlobalPool,
)

class RoFormerEncoderBlock(nn.Module):
    '''A single encoder block consisting of a multi-head self-attention
    layer with rotary embeddings followed by a feed-forward network. 
    Implemented with pre-layer normalization.'''
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            mlp_ratio: float,
            attn_dropout: float,
            ffnn_dropout: float,
            bias: bool = True,
            eps: float = 1e-5,
    ):
        super().__init__()
        self.attention = Attention(
            embed_dim,
            num_heads,
            dropout = attn_dropout,
            bias = bias,
        )
        self.ffnn = FeedForwardSwiGLU(
            embed_dim,
            mlp_ratio,
            dropout = ffnn_dropout,
            bias = bias,
        )

        self.attention_norm = nn.RMSNorm(embed_dim, eps=eps)
        self.atten_dropout = nn.Dropout(attn_dropout)
        self.ffnn_norm = nn.RMSNorm(embed_dim, eps=eps)
        self.ffnn_dropout = nn.Dropout(ffnn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        rotations: torch.Tensor,
        use_sdpa: bool = False,
    ) -> torch.Tensor:
        attn_output = x + self.attention(self.attention_norm(x), rotations, sdpa=use_sdpa)
        attn_output = self.atten_dropout(attn_output)
        ffnn_output = attn_output + self.ffnn(self.ffnn_norm(attn_output))
        ffnn_output = self.ffnn_dropout(ffnn_output)
        return ffnn_output

class RoFormerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        num_layers: int,
        max_seq_len: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        # Cache rotations for rotary embeddings
        rotations = compute_rotations(embed_dim // num_heads, max_seq_len=max_seq_len)
        self.register_buffer("rotations", rotations, persistent=False)
        self.encoder_blocks = nn.ModuleList(
            [
                RoFormerEncoderBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    attn_dropout = dropout,
                    ffnn_dropout = dropout,
                    bias = bias,
                ) 
                for layer_index
                in range(num_layers)
            ]
        )

    def forward(self, x, use_sdpa=False) -> torch.Tensor:
        for block in self.encoder_blocks:
            x = block(x, self.rotations, use_sdpa=use_sdpa)
        return x

class RoFormerMaskedAutoEncoder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        num_layers: int,
        max_seq_len: int,
        dropout: float = 0.0,
        out_channels: int = None,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding1D(
            patch_size,
            in_channels,
            embed_dim,
        )
        self.mask_token = nn.Parameter(
            data=torch.randn(1,1,embed_dim),
            requires_grad = True,
        )
        self.encoder = RoFormerEncoder(
            embed_dim,
            num_heads,
            mlp_ratio,
            num_layers,
            max_seq_len,
            dropout,
        )
        self.norm = nn.RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        out_channels = in_channels if out_channels is None else out_channels
        self.reconstruction_head = InversePatchEmbedding1D(
            patch_size,
            out_channels,
            embed_dim,
        )

    def apply_mask_token(self, x: torch.Tensor, mask_ratio: float = 0.5):
        '''Generates random indeces to mask for each sample in the 
        batch by ranking an thresholding random noise, then replaces 
        the input tokens with the mask token.
        '''
        B, S, D = x.shape
        num_not_to_mask = int(S * (1 - mask_ratio))
        uniform_noise = torch.rand(B, S, device=x.device)
        ranked_noise = torch.argsort(uniform_noise, dim=1)
        mask_idx = ranked_noise >= num_not_to_mask
        x[mask_idx] = self.mask_token.type(x.dtype)
        return x, mask_idx

    def forward(
        self,
        x: torch.Tensor,
        use_sdpa: bool = False,
        mask_ratio: float = 0.0
    ) -> torch.Tensor:
        x = self.patch_embedding(x)
        x, mask_idx = self.apply_mask_token(x, mask_ratio = mask_ratio)
        x = self.encoder(x, use_sdpa=use_sdpa)
        x = self.dropout(self.norm(x))
        x = self.reconstruction_head(x)
        return x, mask_idx

class MultitaskRoFormerMaskedAutoEncoder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        num_layers: int,
        max_seq_len: int,
        dropout: float = 0.0,
        num_tasks: int = 1,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding1D(
            patch_size,
            in_channels,
            embed_dim,
        )
        self.mask_token = nn.Parameter(
            data=torch.randn(1,1,embed_dim),
            requires_grad = True,
        )
        self.encoder = RoFormerEncoder(
            embed_dim,
            num_heads,
            mlp_ratio,
            num_layers,
            max_seq_len,
            dropout,
        )
        self.norm = nn.RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.reconstruction_head = InversePatchEmbedding1D(
            patch_size,
            int(in_channels * num_tasks),
            embed_dim,
        )
        self.num_tasks = num_tasks

    def apply_mask_token(self, x: torch.Tensor, mask_ratio: float = 0.5):
        '''Generates random indeces to mask for each sample in the 
        batch by ranking an thresholding random noise, then replaces 
        the input tokens with the mask token.
        '''
        B, S, D = x.shape
        num_not_to_mask = int(S * (1 - mask_ratio))
        uniform_noise = torch.rand(B, S, device=x.device)
        ranked_noise = torch.argsort(uniform_noise, dim=1)
        mask_idx = ranked_noise >= num_not_to_mask
        x[mask_idx] = self.mask_token.type(x.dtype)
        return x, mask_idx

    def forward(
        self,
        x: torch.Tensor,
        use_sdpa: bool = False,
        mask_ratio: float = 0.0
    ) -> tuple[tuple[torch.Tensor],torch.Tensor]:
        x = self.patch_embedding(x)
        x, mask_idx = self.apply_mask_token(x, mask_ratio = mask_ratio)
        x = self.encoder(x, use_sdpa=use_sdpa)
        x = self.dropout(self.norm(x))
        x = self.reconstruction_head(x)
        x = torch.chunk(x, chunks=self.num_tasks, dim=-2)
        return x, mask_idx

class RoFormerClassifier(nn.Module):
    def __init__(
        self,
        mode: str,
        num_classes: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        num_layers: int,
        num_lstm_layers: int,
        max_seq_len: int,
        encoder_dropout: float = 0.0,
        head_dropout: float = 0.0,
        head: str = 'linear',
        lstm_dim: int = None,
    ) -> None:
        super().__init__()
        assert (mode in ['attention_pool', 'global_pool', 'token_wise']
        ), f"Invalid mode: {mode}. Choose from ['attention_pool', 'global_pool', 'token_wise']"
        self.mode = mode
        self.patch_embedding = PatchEmbedding1D(
            patch_size,
            in_channels,
            embed_dim,
        )
        self.encoder = RoFormerEncoder(
            embed_dim,
            num_heads,
            mlp_ratio,
            num_layers,
            max_seq_len,
            encoder_dropout,
        )
        self.norm = nn.RMSNorm(embed_dim)
        self.num_lstm_layers = num_lstm_layers
        lstm_dropout = head_dropout if num_lstm_layers > 1 else 0.0
        if num_lstm_layers > 0:
            lstm_dim = embed_dim if lstm_dim is None else lstm_dim
            self.pre_lstm = nn.Linear(embed_dim, lstm_dim)
            self.lstm = nn.LSTM(
                input_size = lstm_dim,
                hidden_size = lstm_dim,
                num_layers = num_lstm_layers,
                dropout = lstm_dropout,
                batch_first = True,
                bidirectional = True,
            )
            head_dim = int(lstm_dim * 2)
        else:
            head_dim = embed_dim

        self.head_dropout = nn.Dropout(head_dropout)
        if self.mode == 'attention_pool':
            self.pool = AttentionPool(head_dim)
        elif self.mode == 'global_pool':
            self.pool = GlobalPool()
        elif self.mode == 'token_wise':
            self.pool = nn.Identity()

        if head == 'linear':
            self.classification_head = nn.Linear(head_dim, num_classes)
        elif head == 'mlp':
            hidden_dim = int(head_dim * mlp_ratio)
            self.classification_head = nn.Sequential(
                nn.Linear(head_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim, head_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
                nn.Linear(head_dim, num_classes),
            )

    def forward(self, x, use_sdpa=False) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = self.encoder(x, use_sdpa=use_sdpa)
        x = self.head_dropout(self.norm(x))
        if self.num_lstm_layers > 0:
            x = self.pre_lstm(x)
            x, _ = self.lstm(x)
        x = self.head_dropout(self.pool(x))
        y_hat = self.classification_head(x)
        return y_hat

class RoFormerRegression(nn.Module):
    def __init__(
        self,
        mode: str,
        num_targets: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        num_layers: int,
        num_lstm_layers: int,
        max_seq_len: int,
        encoder_dropout: float = 0.0,
        head_dropout: float = 0.0,
        head: str = 'linear',
        lstm_dim: int = None,
    ) -> None:
        super().__init__()
        assert (mode in ['attention_pool', 'global_pool', 'token_wise']
        ), f"Invalid mode: {mode}. Choose from ['attention_pool', 'global_pool', 'token_wise']"
        self.mode = mode
        self.patch_embedding = PatchEmbedding1D(
            patch_size,
            in_channels,
            embed_dim,
        )
        self.encoder = RoFormerEncoder(
            embed_dim,
            num_heads,
            mlp_ratio,
            num_layers,
            max_seq_len,
            encoder_dropout,
        )
        self.norm = nn.RMSNorm(embed_dim)
        self.num_lstm_layers = num_lstm_layers
        lstm_dropout = head_dropout if num_lstm_layers > 1 else 0.0
        if num_lstm_layers > 0:
            lstm_dim = embed_dim if lstm_dim is None else lstm_dim
            self.pre_lstm = nn.Linear(embed_dim, lstm_dim)
            self.lstm = nn.LSTM(
                input_size = lstm_dim,
                hidden_size = lstm_dim,
                num_layers = num_lstm_layers,
                dropout = lstm_dropout,
                batch_first = True,
                bidirectional = True,
            )
            head_dim = int(lstm_dim * 2)
        else:
            head_dim = embed_dim

        self.head_dropout = nn.Dropout(head_dropout)
        if self.mode == 'attention_pool':
            self.pool = AttentionPool(head_dim, num_targets)
            self.head_dim = int(head_dim * num_targets)
        elif self.mode == 'global_pool':
            self.pool = GlobalPool()
        elif self.mode == 'token_wise':
            self.pool = nn.Identity()

        if head == 'linear':
            self.prediction_head = nn.Linear(head_dim, num_targets)
        elif head == 'mlp':
            hidden_dim = int(head_dim * mlp_ratio)
            self.prediction_head = nn.Sequential(
                nn.Linear(head_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim, head_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(head_dropout),
                nn.Linear(head_dim, num_targets),
            )

    def forward(self, x, use_sdpa=False) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = self.encoder(x, use_sdpa=use_sdpa)
        x = self.head_dropout(self.norm(x))
        if self.num_lstm_layers > 0:
            x = self.pre_lstm(x)
            x, _ = self.lstm(x)
        x = self.head_dropout(self.pool(x))
        y_hat = self.prediction_head(x)
        return y_hat