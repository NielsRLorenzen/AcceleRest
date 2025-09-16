import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPool(nn.Module):
    '''Layer to compute a weighted average of input features using 
    attention weights output by linear layer.'''
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.w = nn.Linear(embed_dim, num_heads, bias=False)

    def forward(self, x):
        '''Compute the attention weighted average of input features.

        Parameters:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
        
        Returns:
            output: Tensor of shape (batch_size, num_heads * embed_dim)
        '''
        attn_weights = F.softmax(self.w(x), dim=1)
        output = torch.bmm(attn_weights.transpose(1, 2), x).view(-1, int(self.num_heads * self.embed_dim))
        return output

class GlobalPool(nn.Module):
    '''Layer to compute the global average of input features.'''
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        '''Compute the global average of input features.

        Parameters:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
        
        Returns:
            output: Tensor of shape (batch_size, embed_dim)
        '''
        return x.mean(dim=1)