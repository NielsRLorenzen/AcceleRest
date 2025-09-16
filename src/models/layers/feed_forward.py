import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardSwiGLU(nn.Module):
    '''Feed Forward Network with SiGLU activation function corresponding
    to SwiGLU with beta = 1 as in LLaMA. The output is computed as:
        output = w2(x) * (sigmoid(w1(x)) * w3(x))'''
    def __init__(
            self,
            embed_dim: int,
            mlp_ratio: float,
            dropout: float = 0.0,
            bias: bool = False
        ) -> None:
        '''Initialize the FeedForwardSwiGLU module.
        Parameters:
            embed_dim: 
                Dimension of the input embeddings.
            mlp_ratio: 
                Ratio of the feed-forward hidden dimension to the 
                embedding dimension.
            dropout: 
                Dropout probability in the hidden layer.
            bias: 
                Whether to use bias in the linear layers.
        '''
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.w_gate = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout1(F.silu(self.w1(x)) * self.w_gate(x))
        return self.w2(hidden)