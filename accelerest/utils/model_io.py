import torch

def save_checkpoint(
        output_dir: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        scaler: torch.cuda.amp.grad_scaler.GradScaler,
        epoch: int,
        step: int,
        patience: int,
    ) -> None: 
    '''Save the model, optimizer and scheduler to a checkpoint file'''

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'step': step,
        'patience': patience,
    }
    
    if scheduler is not None: 
        checkpoint['scheduler'] = scheduler.state_dict()
        
    torch.save(checkpoint,output_dir+f'checkpoint_epoch{epoch}.pt')

def load_checkpoint(
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        scaler: torch.cuda.amp.grad_scaler.GradScaler,
    ) -> tuple[int,int,int]:
    '''Load the model, optimizer and scheduler state dicts 
    from a checkpoint file'''

    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=True)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    patience = checkpoint['patience']

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return epoch, step, patience