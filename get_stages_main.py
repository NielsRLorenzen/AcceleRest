import argparse
import json
import os
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader, SequentialSampler

from src.models.sleepnet import sleepnet
from src.datasets.sleep_dataset import SubjectDataset
from src.models.roformer import RoFormerClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file for a finetuning run.')
    return parser.parse_args()

def load_config(args):
    with open(args.config,'r') as f:
        config = yaml.safe_load(f)
    # Update the args with the config file
    args.__dict__.update(config)
    return args

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
        os.makedirs(dataset_results_path, exist_ok = True)
            
        output_dir = os.path.join(
            dataset_results_path,
            os.path.splitext(os.path.basename(file))[0]
        )
        os.makedirs(output_dir, exist_ok=True)

        # Ensure all output files exist
        soft_preds_exists = os.path.isfile(
            os.path.join(output_dir,'soft_preds.npy')
        )
        
        if getattr(args, "window_wise_predictions", False):
            window_wise_preds_exists = os.path.isfile(
                os.path.join(output_dir, "window_wise_logits.dat")
            )
            window_wise_preds_meta_exists = os.path.isfile(
                os.path.join(output_dir, "window_wise_logits_meta.npy")
            )
            all_exist = (
                soft_preds_exists and 
                window_wise_preds_exists and 
                window_wise_preds_meta_exists
            )
        else:
            all_exist = soft_preds_exists

        if all_exist:
            continue
        
        eval_single(file, model, device, output_dir, args)

def eval_single(file, model, device, output_dir, args):
    subject_set = SubjectDataset(
        file = file,
        patch_size_samples = args.patch_size,
        context_window_patches = args.max_seq_len,
        step_patches = 1,
    )

    loader = DataLoader(
        subject_set,
        batch_size=args.max_batch_size,
        sampler=SequentialSampler(subject_set),  # ensures sequential order
        shuffle=False,
        drop_last=False,
        pin_memory=(device == "cuda"),
    )

    W_all = len(subject_set)
    S = args.max_seq_len
    K = args.n_classes

    sum_logits = torch.zeros((W_all + S - 1, K))
    position_count = torch.zeros((W_all + S - 1, 1))

    # Optional: save predictions for each context window (single file)
    window_wise_memmap = None
    window_wise_path = os.path.join(output_dir, "window_wise_logits.dat")
    if getattr(args, "window_wise_predictions", False):
        window_wise_memmap = np.memmap(
            window_wise_path,
            mode="w+",
            dtype=np.float32,
            shape=(W_all, S, K),
        )

    with torch.no_grad():
        for batch_idx, x in enumerate(loader):
            # # Get Sleep Stage Model output
            x = x.to(device)
            y_hat = model(x).cpu()

            batch_start_idx = (batch_idx * args.max_batch_size)

            # Aggregate model ouputs for each patch
            # W: num_windows (batch_size), S: seq_len, K: n_classes
            W, S, K = y_hat.shape

            # Optional: Save per-window predictions sequentially into the memmap
            if window_wise_memmap is not None:
                window_wise_memmap[batch_start_idx : batch_start_idx + W, :, :] = y_hat.numpy()
           
            for w in range(W):
                window_start = batch_start_idx + w
                # Sum logits across context windows
                sum_logits[window_start : window_start + S] += y_hat[w]
                
                # Count the number of windows that cover each position
                position_count[window_start : window_start + S] += 1

        # Finalize memmap to ensure data is written
        if window_wise_memmap is not None:
            window_wise_memmap.flush()
            # Save a metadata file with shape and dtype
            meta = dict(shape=[W_all, S, K], dtype="float32")
            np.save(os.path.join(output_dir, "window_wise_logits_meta.npy"), meta)

        # How to load memmap output
        # meta = np.load(os.path.join(output_dir, "window_wise_logits_meta.npy"), allow_pickle=True).item()
        # mm = np.memmap(
        #     os.path.join(output_dir, "window_wise_logits.dat"),
        #     mode="r", dtype=meta["dtype"], shape=tuple(meta["shape"])
        # )

        # Save predictions
        avg_logits = sum_logits.div(position_count.clamp_min(1))  
        soft_preds = avg_logits.softmax(dim=-1)
        np.save(os.path.join(output_dir,'soft_preds.npy'), soft_preds.cpu().numpy())


        ## Used previously to track predictions across context windows             
            # Initialize NaN tensor 
            # y_hat_aligned = torch.zeros((W, W + S - 1, K)) * torch.nan

            # Fill in logits for each context window
            # for w in range(W):
                # y_hat_aligned[w, w: S+w, :] = y_hat[w]

            # # Locate patches with NaN across all context windows
            # nan_input = y_hat_aligned.isnan().all(dim=0).all(dim=-1).numpy()

            # # Make soft predictions from the mean across context windows
            # soft_preds = y_hat_aligned.nanmean(dim=0).softmax(dim=-1)

        # Save context window-wise prediction
        # np.save(os.path.join(output_dir,'y_hat_aligned.npy'), y_hat_aligned.cpu().numpy())

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

    pretrained_model = os.path.join(args.model_base_dir, f'checkpoint_epoch{args.ft_epoch}.pt')

    print(args.model_base_dir)
    print(pretrained_model)
    eval(args, pretrained_model, device)

if __name__ == '__main__':
    args = parse_args()
    args = load_config(args)
    main(args)