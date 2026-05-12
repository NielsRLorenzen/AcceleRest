import torch

def compute_rotations(
        head_dim: int,
        max_seq_len: int = 512,
        base: float = 10000.0,
    ) -> torch.Tensor:
    '''A simple implementation of rotary embeddings based on the LLAMA
    implementation. This should be used to precompute and cache
    the rotations at the transformer level.

    Rotation angles (theta) are computed using the formula:
        theta = 1.0 / (base ** (dim_idx / head_dim))
    where 
        dim_idx = [0, 2, 4, ..., head_dim//2] 
    so that one theta is computed for each pair of dimensions in 
    the embedding.

    The theta vector for each pair of positions in the sequence 
    are then computed as the out product:
        seq_theta = seq_idx[:, None] * theta[None, :] 
    where seq_idx is the vector of indices in the sequence:
        [0, 1, 2, ..., max_seq_len-1].
    The resulting matrix has shape:
        (max_seq_len, head_dim//2)
    with rows corresponding to the sequence:
        [0*theta, 1*theta, 2*theta, ..., (max_seq_len-1)*theta]

    The final rotations are the cosine and sine of these angles.
    The rotations have shape:
        (max_seq_len, head_dim//2, 2)
    where the last dimension corresponds to the cosine and sine values.

    Parameters:
        head_dim (int): The dimensionality of the q and k embeddings.
        max_seq_len (int): The maximum sequence length.
        base (float): The base for the rotation angles.
    
    Returns:
        torch.Tensor (max_seq_len, head_dim // 2, 2): The rotations.
    '''

    dim_idx = torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() 
    theta = 1.0 / (base ** (dim_idx / head_dim))
    seq_idx = torch.arange(
        max_seq_len, dtype=torch.float32, device=theta.device
    )
    seq_theta = torch.outer(seq_idx, theta)
    rotations = torch.stack([seq_theta.cos(), seq_theta.sin()], dim=-1)
    return rotations

def apply_rotary_emb(
        x: torch.Tensor,
        rotations: torch.Tensor,
    ) -> torch.Tensor:
    '''Apply rotary embeddings to the input tensor x using
    precomputed rotations. This function should be used inside an
    attention mechanism to apply rotary embeddings to the query and
    key tensors.
    
    Args:
        x (torch.Tensor): 
            Input tensor of shape (B, num_heads, S, head_dim).
        rotations (torch.Tensor):
            Tensor of shape (S, D//2, 2) containing the rotations.

    Returns:
        x_rope (torch.Tensor):
            Output tensor with rotary embeddings applied.

    Shape notation:
            B: batch size
            num_heads: number of heads
            S: sequence length
            head_dim: dimension of each head

    '''
    B, num_heads, S, head_dim = x.shape
    x = x.unflatten(-1, (head_dim//2, 2))
    rotations = rotations[:S].view(1, 1, S, head_dim//2, 2)
    cos_mtheta, sin_mtheta = rotations[..., 0], rotations[..., 1]
    x_rope = torch.stack(
        [
            x[..., 0] * cos_mtheta - x[..., 1] * sin_mtheta,
            x[..., 1] * cos_mtheta + x[..., 0] * sin_mtheta,
        ],
        dim=-1,
    )
    x_rope = x_rope.flatten(-2)
    return x_rope
