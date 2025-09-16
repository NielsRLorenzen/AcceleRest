import os
import glob
import h5py
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta, time

def parse_args():
    argparser = argparse.ArgumentParser(
        description='Extract nights from preprocessed UK biobank accelerometry data'
    )
    argparser.add_argument('--preprocessed_data_dir', type=str, required=True,
                            help='Path to the preprocessed uk biobank actigraphy data folder.')
    argparser.add_argument('--output_dir', type=str, required=True,
                            help='Path to put extracted night files in.')
    argparser.add_argument('--chunk_size', type=int, required=True,
                            help='chunk_size for the output files.')
    return argparser.parse_args()

def extract_nights_all(in_files: list, out_path: str, chunk_size: int):
    done_files = []
    for h5_path in in_files:
        out_h5 = out_path + os.path.basename(h5_path)
        if os.path.exists(out_h5):
            print('skipping', out_h5)
            continue
        extract_nights(h5_path, out_h5, chunk_size)
        done_files.append(out_h5)
    return done_files
    
def extract_nights(h5_path: str, out_h5: str, chunk_size: int,):
    data = defaultdict(dict)
    all_nights = []
    all_nights_time = []
    with h5py.File(h5_path, 'r', rdcc_nbytes=1024**3) as f:
            num_segments = f.attrs['num_segments']
            for segment in range(num_segments):
                data = f[f'data/acc_segment_{segment}'] # shape (n_channels, nsamples)
                nsamples = data.shape[1]
                
                # Get the sampling frequency
                fs = data.attrs['fs']

                # Get the start and end times of the segment
                start_time = datetime.strptime(data.attrs['start_time'], '%Y-%m-%d %H:%M:%S')
                end_time = datetime.strptime(data.attrs['end_time'], '%Y-%m-%d %H:%M:%S')
                
                nights = get_nights(start_time, end_time)
                night_idx = [
                    time_to_idx(start_time, t1, t2, fs)
                    for t1, t2
                     in nights
                ]
                for i, (idx1, idx2) in enumerate(night_idx):
                    all_nights.append(np.array(data[:, idx1:idx2]))
                    all_nights_time.append(nights[i])

    if len(all_nights) == 0:
        print(f'Dropping {h5_path} due to no nights.')
        return

    with h5py.File(out_h5, 'x', rdcc_nbytes=1024**3) as f:
        f.attrs.create('num_nights', len(all_nights))
        f.create_group('data')
        for i in range(len(all_nights)):
            dataset = f['data'].create_dataset(
                f'night_{i}',
                data = all_nights[i],
                chunks=(all_nights[i].shape[0], chunk_size),
            )

            # Write the segment metadata    
            segment_info = {
                'fs': fs,
                'start_time': all_nights_time[i][0].strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': all_nights_time[i][1].strftime('%Y-%m-%d %H:%M:%S'),
            }
            for key, value in segment_info.items():
                dataset.attrs.create(key, value)

def time_to_idx(start_time: datetime, t1: datetime, t2: datetime, fs: float):
    idx1 = int((t1 - start_time).total_seconds() * fs)
    idx2 = int((t2 - start_time).total_seconds() * fs)
    return idx1, idx2

def get_nights(start: datetime, end: datetime):
    """
    Return list of full (9PM, 9AM) night periods within [start, end].

    Args:
        start (datetime): Start datetime
        end (datetime): End datetime

    Returns:
        nights (list of tuples): List of (night_start, night_end) datetime tuples
                                only including full untruncated nights.
    """
    if end <= start:
        return []

    nights = []

    current_date = start.date()

    # Determine first night period
    if start.time() < time(9,0):
        # Start is before 9AM, previous night is ongoing
        night_start = datetime.combine(current_date - timedelta(days=1), time(21,0))
        night_end = datetime.combine(current_date, time(9,0))

    elif start.time() >= time(21,0):
        # Start is after 9PM, current night starts now
        night_start = datetime.combine(current_date, time(21,0))
        night_end = datetime.combine(current_date + timedelta(days=1), time(9,0))
    else:
        # Start is between 9AM and 9PM, next night starts today at 9PM
        night_start = datetime.combine(current_date, time(21,0))
        night_end = datetime.combine(current_date + timedelta(days=1), time(9,0))

    while night_start < end:
        # Check if full night fits within [start, end]
        if night_start >= start and night_end <= end:
            nights.append((night_start, night_end))
        # Move to next night
        night_start += timedelta(days=1)
        night_end += timedelta(days=1)

    return nights

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    files = glob.glob(args.preprocessed_data_dir + '*.h5')
    done_files = extract_nights_all(
        files,
        args.output_dir,
        chunk_size=int(args.chunk_size*30),
    )