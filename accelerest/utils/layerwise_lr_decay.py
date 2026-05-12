import torch
import torch.nn as nn

def get_layerwise_lr_groups(
    model: torch.nn.Module,
    base_lr: float,
    lr_decay: float,
    ) -> list[dict]:
    '''Adapted from:
    https://gist.github.com/gautierdag/3bd64f33470cb11f4323ce7fa86524a9
    
    Parameters:
        model: 
            Model with a model.encoder module that has a list of 
            model.encoder.encoder_blocks and a patch_embedding layer.
        base_lr:
            Base learning rate for the model that will be decayed for
            each layer.
        lr_decay:
            Factor by which the learning rate will be decayed for each
            layer.

    Returns (List[Dict]):
        List of dictionaries with the parameters and learning rates for
        each layer of the model to be given to an optimzer.

    Example:
        model = RoFormerClassifier(args)
        lr_groups = get_layerwise_lr_groups(model, 1e-4, 0.9)
        optimizer = torch.optim.AdamW(lr_groups)


    '''
    n_layers = len(model.encoder.encoder_blocks) + 1 # +1 for the encoder embedding layer
    patch_embedding_lr = base_lr * (lr_decay ** (n_layers + 1))
    grouped_parameters = []
    grouped_parameters.append({
        'params': model.patch_embedding.parameters(),
        'lr': patch_embedding_lr,
    })
    for i in range(n_layers):
        lr = base_lr * (lr_decay ** (n_layers - i))
        grouped_parameters.append({
            'params': model.encoder.encoder_blocks[i-1].parameters(),
            'lr': lr
        })

    return grouped_parameters

def get_llrd_params(
    model,
    base_lr,
    classifier_lr,
    weight_decay=0.01,
    lr_decay=0.95
):
    """
    Create parameter groups with layerwise learning rate decay.

    Args:
        model: Full model with encoder and classifier.
        base_lr: LR for the last encoder layer.
        classifier_lr: LR for classifier head.
        weight_decay: Weight decay value.
        lr_decay: Learning rate decay factor per layer.
    Returns:
        A list of param groups to pass to the optimizer.
    """
    param_groups = []

    # Classifier components
    classifier_modules = [
        model.lstm if model.num_lstm_layers > 0 else None,
        model.norm,
        model.pool if isinstance(model.pool, nn.Module) else None,
        model.classification_head,
    ]

    for module in classifier_modules:
        if module is not None:
            param_groups.append({
                "params": module.parameters(),
                "lr": classifier_lr,
                "weight_decay": weight_decay,
            })

    # Layer-wise encoder block LRs (top-down)
    encoder_blocks = list(model.encoder.encoder_blocks)
    num_blocks = len(encoder_blocks)

    for i, block in enumerate(reversed(encoder_blocks)):
        lr = base_lr * (lr_decay ** i)  # last layer gets base_lr, earlier layers get less
        param_groups.append({
            "params": block.parameters(),
            "lr": lr,
            "weight_decay": weight_decay,
        })

    # Add lr for patch_embedding layer (tokenizer)
    param_groups.append({
        "params": model.patch_embedding.parameters(),
        "lr": base_lr * (lr_decay ** (num_blocks + 1)),  # Decay more
        "weight_decay": weight_decay,
    })

    return param_groups