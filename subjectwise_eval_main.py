import argparse
import json
import os
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader

from src.models.sleepnet import sleepnet
from src.datasets.sleep_dataset import SubjectEvaluationDataset
from src.models.roformer import RoFormerClassifier

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

def get_sleepnet():
    model = sleepnet(
        pretrained=True,
        my_device="cpu",
        num_classes=5,
        lstm_nn_size=1024,
        dropout_p=0.0,
        bi_lstm=True,
        lstm_layer=2,
        local_weight_path="",
    )
    return model

def get_roformer(args, pretrained_model):
    model = RoFormerClassifier(
        mode = "token_wise",
        num_classes = args.n_classes,
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
        head_dropout = 0,
    )
    print(f'Loading {pretrained_model}')
    checkpoint = torch.load(pretrained_model, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True) # Load finetuned classifier parameters
    return model

def eval(args, pretrained_model, device, fold = None):
    if pretrained_model == 'sleepnet':
        model = get_sleepnet()
    else:
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
        if pretrained_model == 'sleepnet':
            eval_sleepnet_single(file, model, device, output_dir, args)
        else:
            eval_single(file, model, device, output_dir, args)

def eval_single(file, model, device, output_dir, args):
    subject_set = SubjectEvaluationDataset(
        file = file,
        patch_size_samples = args.patch_size,
        context_window_patches = args.max_seq_len,
        step_patches = 1,
        labels = args.labels_name,
        label_map = args.label_map,
        label_priority = False,
        patches_per_label = 1,
        ignore_index = -9,
    )

    with torch.no_grad():
        # Get Sleep Stage Model output
        x, y = subject_set[:len(subject_set)]
        x, y = x.to(device), y.to(device)
        y_hat = model(x)

        # y_hat = y_hat.softmax(dim=-1)
        if args.sequence_weights is not None:
            sequence_weights = get_sequence_weights(args.sequence_weights).to(device) + 1e-3
            y_hat = y_hat * sequence_weights.unsqueeze(0).unsqueeze(-1)
            
        # Aggregate model ouputs for each patch
        W, S, K = y_hat.shape
        y_hat_aligned = torch.zeros((W, W + S - 1, K)) * torch.nan
        for w in range(W):
            y_hat_aligned[w, w: S+w, :] = y_hat[w]
        nan_input = y_hat_aligned.isnan().all(dim=0).all(dim=-1).numpy()

        soft_preds = y_hat_aligned.nanmean(dim=0).softmax(dim=-1)
        subject_targets = subject_set.labels_sequence

        # Save preds and labels
        np.save(os.path.join(output_dir,'targets.npy'), subject_targets.numpy())
        np.save(os.path.join(output_dir,'y_hat_aligned.npy'), y_hat_aligned.cpu().numpy())
        np.save(os.path.join(output_dir,'soft_preds.npy'), soft_preds.cpu().numpy())

def get_sequence_weights(kind):
    # Get path to this file's directory
    _module_dir = os.path.dirname(__file__)

    # Load the .npy file
    if kind == 'norm_auc':
        sequence_weights = np.load(os.path.join(_module_dir, "weights/norm_auc_weights.npy"))
    elif kind == 'triangle':
        sequence_weights = np.load(os.path.join(_module_dir, "weights/triangle_weights.npy"))

    sequence_weights = torch.from_numpy(sequence_weights)
    return sequence_weights

def eval_sleepnet_single(file, model, device, output_dir, args):
    subject_set = SubjectEvaluationDataset(
        file = file,
        patch_size_samples = 900,
        context_window_patches = -1, # Full recording
        step_patches = 1,
        labels = args.labels_name,
        label_map = args.label_map,
        label_priority = False,
        patches_per_label = 1,
        ignore_index = -9,
    )

    with torch.no_grad():
        # Get Sleep Stage Model output
        x, y = subject_set[0]
        x, y = x.to(device), y.to(device)
        y_hat = model(x)

        # Aggregate model ouputs for each patch
        W, S, K = y_hat.shape
    
        soft_preds = y_hat.squeeze(0).softmax(dim=-1)
        subject_targets = subject_set.labels_sequence

        # Save preds and labels
        np.save(os.path.join(output_dir,'targets.npy'), subject_targets.numpy())
        np.save(os.path.join(output_dir,'soft_preds.npy'), soft_preds.cpu().numpy())
        
def get_embeddings():
    # # Get embbeddings (Use pretrained model + norm)
    # patch_embeddings = mae_model.patch_embedding(x)
    # x_embed = mae_model.encoder(patch_embeddings, use_sdpa=True)
    # x_embed = mae_model.norm(x_embed)
    # W, S, D = x_embed.shape
    # x_embed_aligned = torch.zeros((W, W + S - 1, D)) * torch.nan
    # for w in range(W):
    #     x_embed_aligned[w, w: S+w, :] = x_embed[w]
    pass

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
        if args.model_base_dir == 'sleepnet':
            pretrained_model = args.model_base_dir
        else:
            pretrained_model = os.path.join(args.model_base_dir, f'checkpoint_epoch{args.ft_epoch}.pt')
        print(args.model_base_dir)
        print(pretrained_model)
        eval(args, pretrained_model, device)

if __name__ == '__main__':
    args = parse_args()
    args = load_config(args)
    main(args)