import os
import sys
import glob
import argparse
import traceback
import pandas as pd
import numpy as np
import json
import warnings
import actipy

# # Assuming newcastle.utils is in PYTHONPATH or newcastle/utils.py exists
# from newcastle import utils

# # Import shared preprocessing functions
from utils.preprocess_actigraphy import preprocess_actigraphy_df
from utils.write_h5 import write_h5_acc

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--actigraphy_path',
        type = str,
        help ='Path to the directory containing the actigraphy files',
        default = '/oak/stanford/groups/mignot/actigraphy/newcastle2015/dataset_psgnewcastle2015_v1.0/acc/'
    )
    argparser.add_argument(
        '--hypno_path',
        type = str,
        help = 'Path to the directory containing the edf files',
        default = '/oak/stanford/groups/mignot/actigraphy/newcastle2015/dataset_psgnewcastle2015_v1.0/psg/'
    )
    argparser.add_argument(
        '--output_dir',
        type=str,
        help = 'Directory to save the output h5 files',
        default = '/oak/stanford/groups/mignot/projects/actigraphy_fm/data/newcastle/'
    )
    argparser.add_argument(
        '--config_json_path',
        type=str,
        help = 'Path to json config file',
        default = '/oak/stanford/groups/mignot/projects/actigraphy_fm/code/SRL_WASSED/preprocessing/newcastle/preproc_config.json'
    )
    return argparser.parse_args()

def read_config(config_path: str, args):
    with open(config_path, 'r') as f:
        config = json.load(f)
    args.__dict__.update(config)

args = parse_args()
read_config(args.config_json_path, args)

# psg_annot_path = '/oak/stanford/groups/mignot/actigraphy/newcastle2015/dataset_psgnewcastle2015_v1.0/psg/'
# acc_bin_path = '/oak/stanford/groups/mignot/actigraphy/newcastle2015/dataset_psgnewcastle2015_v1.0/acc/'
# demographics = '/oak/stanford/groups/mignot/actigraphy/newcastle2015/dataset_psgnewcastle2015_v1.0/participants_info.csv'
short_recording = []
calib_failed = []

psg_paths = glob.glob(args.hypno_path + '*.txt')
acc_paths = glob.glob(args.actigraphy_path + '*.bin')
psg_paths.sort()
acc_paths.sort()

for annot_file in psg_paths:
    sub_id = os.path.basename(annot_file).split('_')[0].upper()
    print(f'\n Processing {sub_id}')
    # Read header
    header = []
    with open(annot_file) as f:
        for line_num, line in enumerate(f):
            header.append(line)
            if 'Recording Date:' in line:
                date = line.split('\t')[1].strip()

            if 'Scoring Time:' in line:
                header_end = line_num + 1
                break

    annot_df = pd.read_csv(annot_file, sep='\t', skiprows=header_end)
    # Add recording date from header to time
    annot_df['time'] = date + ' ' + annot_df['Time [hh:mm:ss]']
    annot_df['time'] = pd.to_datetime(annot_df['time'], format= '%d/%m/%Y %H:%M:%S')
    annot_df = annot_df.set_index('time')
    annot_df.index = annot_df.index.tz_localize('UTC')
    # Make sure dates are incremented after midnight
    time_diffs = annot_df.index.diff()
    day_increments = (time_diffs < pd.Timedelta(0)).cumsum()
    annot_df.index = annot_df.index + pd.to_timedelta(day_increments, unit='D')
    # Make new dataframe and map labels to format
    hypnogram = pd.DataFrame(index = annot_df.index)
    hypnogram['hypnogram'] = annot_df['Sleep Stage'].map({
        'W': 'wake',
        'N1': 'n1',
        'N2': 'n2',
        'N3': 'n3',
        'R': 'rem',
    })
    hypnogram['sleep_wake'] = annot_df['Sleep Stage'].map({
        'W': 'wake',
        'N1': 'sleep',
        'N2': 'sleep',
        'N3': 'sleep',
        'R': 'sleep',
    })

    hypnogram['light_deep_rem'] = annot_df['Sleep Stage'].map({
        'W': 'wake',
        'N1': 'light',
        'N2': 'light',
        'N3': 'deep',
        'R': 'rem',
    })

    hypnogram['rem_nrem'] = annot_df['Sleep Stage'].map({
        'W': 'wake',
        'N1': 'nrem',
        'N2': 'nrem',
        'N3': 'nrem',
        'R': 'rem',
    })

    sub_acc = {
        'left': None,
        'right': None,
    }
    for path in acc_paths:
        acc_sub, wrist = os.path.basename(path).split('_')[:2]
        if acc_sub  == sub_id:
            if wrist == 'left wrist':
                sub_acc['left'] = path
            elif wrist == 'right wrist':
                sub_acc['right'] = path

    for wrist, path in sub_acc.items():
        if path is not None:
            acc_df, info_dict = actipy.read_device(
                input_file = path,
                lowpass_hz = None,
                calibrate_gravity = False,
                detect_nonwear = False,
                resample_hz = None,
                verbose = False,
            )

            # Determine the input sampling frequency (fs_in)
            fs_in = info_dict.get('sample_rate')
            if fs_in is None:
                header_fs_str = info_dict.get('Measurement Frequency')
                if header_fs_str:
                    try:
                        fs_in = float(header_fs_str)
                    except ValueError:
                        warnings.warn(f"Could not parse 'Measurement Frequency' ('{header_fs_str}') as float.", UserWarning)
                else:
                    print(f'Inferring sample freq for {path}')
                    sample_gaps = acc_df.index.diff().total_seconds()[1:].round(4)
                    print('Identified unique sample spacings (s):', sample_gaps.unique())
                    fs_in = 1/np.mean(sample_gaps)
                    print('Avg Hz:', fs_in, '\n')

            if acc_df.index.tz is None:
                print("    Actigraphy index is timezone-naive. Localizing to UTC.")
                # tz_localize attaches timezone info to a naive index. This is the correct method here.
                acc_df.index = acc_df.index.tz_localize('UTC')

            elif str(acc_df.index.tz).upper() != 'UTC':
                print(f"    Actigraphy index has non-UTC timezone ({data_df.index.tz}). Converting to UTC.")
                acc_df.index = acc_df.index.tz_convert('UTC')
            
            print(f'Accelerometry recording time {acc_df.index[0]} - {acc_df.index[-1]}')
            print(f'Hypnogram recording time {hypnogram.index[0]} - {hypnogram.index[-1]}')

            act_before_edf_start = acc_df.index[0] < hypnogram.index[0]
            act_after_edf_end = acc_df.index[-1] > hypnogram.index[-1]

            start_time = hypnogram.index[0] if act_before_edf_start else acc_df.index[0]
            end_time = hypnogram.index[-1] if act_after_edf_end else acc_df.index[-1]
        
            if (end_time - start_time).total_seconds()/60**2 < args.min_recording_hours:
                print(f'Less than {args.min_recording_hours} of concurrent data for {path}, skipping.')
                short_recording.append(path)
                continue
            
            acc_df_preproc, all_info =  preprocess_actigraphy_df(
                data_df = acc_df,
                x_col = 'x',
                y_col = 'y',
                z_col = 'z',
                fs_in = fs_in,
                fs_out = args.actigraphy_resample_freq,
                gravity_calibration = True,
                calib_cube = args.calib_cube,
                max_iters_calib = 1000,
                verbose = False,
            )

            if all_info['calibration_diagnostics']['CalibOK'] == 0:
                calib_failed.append(subjects_df.loc[i, 'act'])
                print('Skipping file due to failed calibration')
                continue

            # Cut acc to hypnogram
            acc_df_preproc = acc_df_preproc.loc[hypnogram.index[0]:hypnogram.index[-1]]

            # Make new empty df with accelerometry index
            hyp_upsample = pd.DataFrame(index=acc_df_preproc.index)
            hyp_upsample = pd.merge_asof(
                hyp_upsample,
                hypnogram,
                left_index = True,
                right_index = True,
                direction = 'forward',
                tolerance = pd.Timedelta(31, unit='s'),
            )

            # Get sleep stage amounts
            stages_hours = {}
            for stage in hyp_upsample['hypnogram'].unique():
                stage_hours = round(
                    (hyp_upsample['hypnogram'] == stage).sum() 
                    / args.actigraphy_resample_freq / 60**2,
                    3
                )
                stages_hours[stage] = stage_hours

            outfile = args.output_dir + sub_id.lower() + '_' + wrist + '.h5'
            write_h5_acc(
                outfile = outfile,
                accelerometry = acc_df_preproc,
                acc_info = all_info,
                x_col = 'x_preprocessed',
                y_col = 'y_preprocessed',
                z_col = 'z_preprocessed',
                annotations = hyp_upsample.astype(str),
                chunk_size_sec = args.hdf5_chunksize_sec,
                study_start = start_time,
                ahi = None,
                stage_hours = stages_hours,
            )
            print(stages_hours)
            print(f'Saved file {outfile} \n')