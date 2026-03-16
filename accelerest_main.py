import argparse
import os, glob
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader, SequentialSampler

from src.datasets.sleep_dataset import SubjectDataset

def parse_args():
    parser = argparse.ArgumentParser()
    # IO parameters
    parser.add_argument('--data_file_dir', type=str, required = True,
                        help='Path to folder with h5 files to run AcceleRest on.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to folder to save accelerest outputs in.'
                             'Defaults to <data_file_dir>/accelerest_outputs')
                             
    # Select which predictions to return
    parser.add_argument('--lstm_sleepstages', action='store_true',
                        help='Return sleep stages predicted with an LSTM-C head.')
    parser.add_argument('--linear_sleepstages', action='store_true',
                        help='Return sleep stages predicted with a linear head.')
    parser.add_argument('--linear_resp_events', action='store_true',
                        help='Return respiratory events predicted with a linear head.')

    # Processing parameters
    parser.add_argument('--context_window_shift', type=int, default=1,
                        help='The number of 30 sec patches to shift consecutive windows.')
    parser.add_argument('--max_batch_size', type=int, default=32,
                        help='Batch size for inference.')

    # Select intermediate outputs to save
    parser.add_argument('--get_embeddings', action='store_true',
                        help='Return model embeddings.')
    parser.add_argument('--window_wise_predictions', action='store_true',
                        help='Return predictions for each context window.')
    args = parser.parse_args()

    # Dynamically set default output dir
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_file_dir, "accelerest_outputs")

    return args

def outputs_exist(output_dir: str, prefixes: list, args):
    prefix_preds_exist = []
    for prefix in prefixes:
        soft_preds_exists = os.path.isfile(
            os.path.join(output_dir, f'{prefix}_soft_preds.npy')
        )

        if args.window_wise_predictions:
            window_wise_preds_exists = os.path.isfile(
                os.path.join(output_dir, f"{prefix}_window_wise_logits.dat")
            )
            window_wise_preds_meta_exists = os.path.isfile(
                os.path.join(output_dir, f"{prefix}_window_wise_logits_meta.npy")
            )
            prefix_preds_exist.append(
                soft_preds_exists and
                window_wise_preds_exists and
                window_wise_preds_meta_exists
            )
        
        else:
            prefix_preds_exist.append(soft_preds_exists)

    return all(prefix_preds_exist)

def eval(args, device):
    model = torch.hub.load(
        'NielsRLorenzen/AcceleRest:multihead',
        'accelerest_multihead',
        linear_sleepstage = args.linear_sleepstages,
        lstm_sleepstage = args.lstm_sleepstages,
        linear_respevent = args.linear_resp_events,
        trust_repo='check',
        force_reload = True,
    )

    model.to(device)
    model.eval()

    # Get files
    files = glob.glob(os.path.join(args.data_file_dir, '*.h5'))

    # Make overall output dir
    os.makedirs(args.output_dir, exist_ok = True)

    for i, file in enumerate(files):
        # Make individual output dirs if they don't exist
        individual_output_dir = os.path.join(
            args.output_dir,
            os.path.splitext(os.path.basename(file))[0]
        )
        if os.path.exists(individual_output_dir):
            # Check if all output files exist
            if outputs_exist(individual_output_dir, model.names, args):
                continue
        
        else:
            os.makedirs(individual_output_dir)

        eval_single(file, model, device, individual_output_dir, args)


def init_storage(output_dir, prefix, num_windows, window_size, num_classes, args):
    storage = {
        "sum_logits": torch.zeros((num_windows + window_size - 1, num_classes), dtype=torch.float32),
        "position_count": torch.zeros((num_windows + window_size - 1, 1), dtype=torch.float32),
        "memmap": None,
        "shape": (num_windows, window_size, num_classes),
    }

    if args.window_wise_predictions:
        storage["memmap"] = np.memmap(
            os.path.join(output_dir, f"{prefix}_window_wise_logits.dat"),
            mode="w+",
            dtype=np.float32,
            shape=(num_windows, window_size, num_classes),
        )

    return storage

def store_outputs(storage: dict, prefix:str, y_hat: torch.Tensor, batch_start_idx:int):
    batch_size, window_size, num_classes = y_hat.shape

    # Fill slice of memmap corresponding to batch (optional)
    if storage["memmap"] is not None:
        storage["memmap"][batch_start_idx: batch_start_idx + batch_size, :, :] = y_hat.numpy()

    for i in range(batch_size):
        window_start = batch_start_idx + i
        # Fill in predictions summing patch logits across context windows
        storage["sum_logits"][window_start: window_start + window_size] += y_hat[i]
        # Store number of context windows for each patch (Used to avg. later)
        storage["position_count"][window_start: window_start + window_size] += 1

def finalize_head_storage(storage, output_dir, prefix):
    # Save memmap and meta data for reading
    if storage["memmap"] is not None:
        storage["memmap"].flush()
        meta = {
            "shape": list(storage["shape"]),
            "dtype": "float32",
        }
        np.save(
            os.path.join(output_dir, f"{prefix}_window_wise_logits_meta.npy"),
            meta,
        )

    # Save the soft predictions based on cross-context window average logits
    avg_logits = storage["sum_logits"].div(
        storage["position_count"].clamp_min(1)
    )
    soft_preds = avg_logits.softmax(dim=-1)
    np.save(
        os.path.join(output_dir, f"{prefix}_soft_preds.npy"),
        soft_preds.numpy(),
    )

def eval_single(file, model, device, output_dir, args):
    subject_set = SubjectDataset(
        file=file,
        patch_size_samples=model.patch_size,
        context_window_patches=model.max_seq_len,
        step_patches=args.context_window_shift,
    )

    loader = DataLoader(
        subject_set,
        batch_size=args.max_batch_size,
        sampler=SequentialSampler(subject_set),
        shuffle=False,
        drop_last=False,
        pin_memory=(device == "cuda"),
    )

    num_windows = len(subject_set)
    window_size = model.max_seq_len

    storage = {}

    with torch.no_grad():
        for batch_idx, x in enumerate(loader):
            batch_start_idx = batch_idx * args.max_batch_size
            
            x = x.to(device)
            outputs = model(x)

            for name, logits in outputs.items():
                if name not in storage.keys():
                    # Initialize output storage for each prediction head
                    _, _, num_classes = logits.shape
                    storage[name] = init_storage(
                        output_dir, name, num_windows, window_size, num_classes, args,
                    )   
                logits = logits.cpu()
                store_outputs(name, logits, batch_start_idx)

    for name in storage.keys():
        finalize_storage(storage[name], output_dir, name)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)
    eval(args, device)

if __name__ == '__main__':
    args = parse_args()
    main(args)