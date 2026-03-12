import argparse
import os, glob
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader, SequentialSampler

from src.datasets.sleep_dataset import SubjectDataset

def parse_args():
    parser = argparse.ArgumentParser()
    # Base parameters
    parser.add_argument('--data_file_dir', type=str, required = True,
                        help='Path to folder with h5 files to run AcceleRest on.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to folder to save accelerest outputs in.'
                             'Defaults to <data_file_dir>/accelerest_outputs')
    parser.add_argument('--context_window_shift', type=int, default=1,
                        help='The number of 30 sec patches to shift consecutive windows.')
    parser.add_argument('--max_batch_size', type=int, default=32,
                        help='Batch size for inference.')

    # Deselect which predictions to return
    parser.add_argument('--no_sleepstages', dest='get_sleepstages', action='store_false',
                        help='Do not return predicted sleep stages.')
    parser.add_argument('--no_respiratory_events', dest='get_respiratory_events', action='store_false',
                        help='Do not return predicted respiratory events.')
    arser.set_defaults(get_sleepstages=True, get_respiratory_events=True)

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

def outputs_exist(output_dir, args):
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

    return all_exist

def outputs_exist(output_dir, args):
    prefixes = []
    if args.get_sleepstages:
        prefixes.append('sleepstage')
    if args.get_respiratory_events:
        prefixes.append('respevents')

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
    # Get the AcceleRest model
    if args.get_sleepstages and args.get_respiratory_events:
        model = torch.hub.load('NielsRLorenzen/AcceleRest', 'accelerest_dualhead')
    
    elif args.get_sleepstages:
        model = torch.hub.load('NielsRLorenzen/AcceleRest', 'accelerest_sleepstage')

    elif args.get_respiratory_events:
        model = torch.hub.load('NielsRLorenzen/AcceleRest', 'accelerest_respevent')
    
    else:
        raise ValueError('Select either --get_sleepstages or --get_respiratory_events or both')

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
            if outputs_exist(individual_output_dir, args):
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

def store_outputs(storage, y_hat, batch_start_idx):
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
    nclass_sleep = 4
    nclass_resp = 2

    # Initialize output storage
    if args.get_sleepstages:
        sleepstage_storage = init_storage(
            output_dir, "sleepstage", num_windows, window_size, nclass_sleep, args,
        )

    if args.get_respiratory_events:
        respevent_storage = init_storage(
            output_dir, "respevent", num_windows, window_size, nclass_resp, args,
        )

    with torch.no_grad():
        for batch_idx, x in enumerate(loader):
            x = x.to(device)
            output = model(x)

            batch_start_idx = batch_idx * args.max_batch_size

            # Get model outputs
            if args.get_sleepstages and args.get_respiratory_events:
                sleep_logits, resp_logits = output
                sleep_logits = sleep_logits.cpu()
                resp_logits = resp_logits.cpu()

                store_outputs(sleepstage_storage, sleep_logits, batch_start_idx)
                store_outputs(respevent_storage, resp_logits, batch_start_idx)

            elif args.get_sleepstages:
                sleep_logits = output.cpu()
                store_outputs(sleepstage_storage, sleep_logits, batch_start_idx)

            elif args.get_respiratory_events:
                resp_logits = output.cpu()
                store_outputs(respevent_storage, resp_logits, batch_start_idx)

    if args.get_sleepstages:
        finalize_head_storage(sleepstage_storage, output_dir, "sleepstage")

    if args.get_respiratory_events:
        finalize_head_storage(respevent_storage, output_dir, "respevent")

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)
    eval(args, device)

if __name__ == '__main__':
    args = parse_args()
    main(args)