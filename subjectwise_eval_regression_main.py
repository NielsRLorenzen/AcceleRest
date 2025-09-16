import argparse
import json
import os
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader

from src.models.sleepnet import sleepnet
from src.datasets.sleep_dataset import SubjectEvaluationDataset
from src.models.roformer import RoFormerRegression

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cv_folds_path', type=str, required=True,
    #                     help='Path to cross validation fold json files')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file for a finetuning run.')
    # parser.add_argument('--ss_base_dir', type=str,
    #                     help = 'Path to sleep staging cv model checkpoints.',
    #                     default = None)
    # parser.add_argument('--apnea_base_dir', type=str,
    #                     help = 'Path to apnea cv model checkpoints.',
    #                     default = None)
    # parser.add_argument('--ft_epoch', type=int, required=True,
    #                     help='Finetuning epoch to load from base dirs.')
    # parser.add_argument('--results_path', type=str, required=True,
    #                     help='Output directory subject wise dirs will be here.')
    return parser.parse_args()

def load_config(args):
    with open(args.config,'r') as f:
        config = yaml.safe_load(f)
    # Update the args with the config file
    args.__dict__.update(config)
    return args

def get_roformer(args, pretrained_model):
    model = RoFormerRegression(
            mode = "attention_pool",
            num_targets = args.num_targets,
            patch_size = args.patch_size,
            in_channels = args.in_channels,
            embed_dim = args.embed_dim,
            num_heads = args.num_heads,
            mlp_ratio = args.mlp_ratio,
            num_layers = args.num_layers,
            num_lstm_layers = args.num_lstm_layers,
            max_seq_len = args.max_seq_len,
            lstm_dim = args.lstm_dim,
            head = args.head,
    ) 
  
    print(f'Loading {pretrained_model}')
    checkpoint = torch.load(pretrained_model, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True) # Load finetuned classifier parameters
    return model

def eval(args, pretrained_model, device, fold = None):
    model = get_roformer(args, pretrained_model)
    model.to(device)
    model.eval()

    with open(args.val_set_json,'r') as f:
        val_files = json.load(f)

    for i, file in enumerate(val_files):
        dataset = os.path.basename(os.path.dirname(file))
        dataset_results_path = os.path.join(args.results_path, dataset)
        if not os.path.exists(dataset_results_path):
            os.makedirs(dataset_results_path)
            
        output_dir = os.path.join(dataset_results_path, os.path.basename(file)[:-3])
        os.mkdir(output_dir)
        eval_single(file, model, device, output_dir, args)

def eval_single(file, model, device, output_dir, args):
    subject_set = SubjectEvaluationDataset(
        file = file,
        patch_size_samples = args.patch_size,
        context_window_patches = args.max_seq_len,
        step_patches = 1,
        labels = args.labels_name,
        label_map = args.label_map,
        label_priority = 'event_count',
        patches_per_label = 256,
        ignore_index = -9,
    )

    with torch.no_grad():
        # Get Sleep Stage Model output
        x, y = subject_set[:len(subject_set)]
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
            
        y_hat = y_hat.squeeze().cpu()
        y = y.squeeze().cpu()

        # Save preds and labels
        np.save(os.path.join(output_dir,'targets.npy'), y.numpy())
        np.save(os.path.join(output_dir,'preds.npy'), y_hat.cpu().numpy())

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    if args.n_folds is not None:
        output_dir = args.results_path
        for fold in range(args.n_folds):    
        
            args.val_set_json = f'{args.cv_folds_path}/fold_{fold}_validation_files.json'
            args.results_path = os.path.join(output_dir, f'fold_{fold}/')
            os.mkdir(args.results_path)

            if args.model_base_dir == 'sleepnet':
                pretrained_model = args.model_base_dir
            else:
                pretrained_model = os.path.join(args.model_base_dir, f'fold_{fold}/checkpoint_epoch{args.ft_epoch}.pt')
            
            eval(args, pretrained_model, device, fold)

    else:
        pretrained_model = os.path.join(args.model_base_dir, f'checkpoint_epoch{args.ft_epoch}.pt')
        eval(args, pretrained_model, device)

if __name__ == '__main__':
    args = parse_args()
    args = load_config(args)
    main(args)