import torch

def get_grad_norm(parameters):
    grads = [param.grad.detach().flatten() for param in parameters if param.grad is not None]
    norm = torch.cat(grads).norm()
    return norm