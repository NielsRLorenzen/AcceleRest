import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils.rotary_embeddings import apply_rotary_emb

class Attention(nn.Module):
    '''Multihead Self Attention Layer with Rotary Embeddings.
    Only supports same dimension for query, key, and value.
    Does not support grouped query attention.'''
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            bias: bool = True,
        ):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.w_qkv = nn.Linear(embed_dim, int(3 * embed_dim), bias=bias)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=bias)
        
    def forward(
            self,
            input: torch.Tensor,
            rotations: torch.Tensor,
            sdpa: bool = False,
        ):
        B, S, D = input.shape
        assert (D == self.embed_dim
        ), f"Expected input of shape (B, S, {self.embed_dim}), but got {D}"

        q, k, v = self.w_qkv(input).chunk(3, dim=-1)
        
        # Reshape to (B, S, num_heads, head_dim) and 
        # transpose to (B, num_heads, S, head_dim)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2) 
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, rotations)
        k = apply_rotary_emb(k, rotations)

        if not sdpa:
            q_scaled = q * math.sqrt(1.0 / float(self.head_dim))
            attn_weights = torch.matmul(q_scaled, k.transpose(-2, -1))
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
            if self.dropout > 0.0:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            attn_output = torch.matmul(attn_weights, v)

        if sdpa:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask = None, 
                dropout_p = self.dropout,
            )

        # Reshape back to (B, S, embed_dim)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        )
        return self.w_o(attn_output)