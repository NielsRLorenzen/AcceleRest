import os
import h5py
import json
import glob
import actipy
import argparse
import numpy as np
import pandas as pd
from pyedflib.edfreader import EdfReader

from stages import io_utils
from utils.preprocess_actigraphy import preprocess_actigraphy_df
from utils.preprocess_psg import preprocess_psg_dict
from utils.write_h5 import write_h5_acc_psg

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--subject_dirs_path',
        type = str,
        help='Path to the directory containing the subject directories',
    )
    argparser.add_argument(
        '--actigraphy_path',
        type = str,
        help ='Path to the directory containing the actigraphy files',
    )
    argparser.add_argument(
        '--output_dir',
        type=str,
        help = 'Directory to save the output h5 files',
    )
    argparser.add_argument(
        '--config_json_path',
        type=str,
        help = 'Path to json config file',
        default = './preproc_config.json'
    )
    return argparser.parse_args()

def read_config(config_path: str, args):
    with open(config_path, 'r') as f:
        config = json.load(f)
    args.__dict__.update(config)

def main(args):
    act_files = glob.glob(args.actigraphy_path+'*.h5')

    sub_ids = [os.path.basename(sub)[:-3] for sub in act_files]

    subject_dirs = [
        f.path 
        for f 
        in os.scandir(args.subject_dirs_path) 
        if f.is_dir()
    ]

    calib_failed = []
    short_recording = []
    errors = []
    ahis = []
    stages_count = {
        'wake': 0,
        'n1': 0,
        'n2': 0,
        'n3': 0,
        'rem': 0,
        'missing': 0
    }

    for act_h5 in act_files:
        subject_id = os.path.basename(act_h5)[:-3]

        # Skip if the output file already exists
        out_file = os.path.join(args.output_dir, f'{subject_id}.h5')
        if os.path.exists(out_file):
            print(f'{out_file} already exists, skipping\n')
            continue
        
        subject_dir = os.path.join(args.subject_dirs_path, subject_id)

        ## -- PSG --
        # Read EDF
        edf_path = glob.glob(subject_dir+'/**/*.edf', recursive=True)[0]
        print(f'Processing {edf_path}')
        with EdfReader(edf_path) as edf_file:
            study_start = pd.Timestamp(edf_file.getStartdatetime())
            
            labels = edf_file.getSignalLabels()
            l, c = np.unique(labels, return_counts=True)
            if np.any(c > 1):
                print(f'More than one {l[c > 1]} channel found in {edf_path} keeping first')
            
            duplicate_idx = []
            for label in l[c > 1]:
                duplicate_idx.extend([i for i, x in enumerate(labels) if x == label][1:])

            signals = {
                label: edf_file.readSignal(i) 
                for i, label 
                in enumerate(labels) 
                if label in args.psg_montage and i not in duplicate_idx
            }
            
            headers = edf_file.getSignalHeaders()
            
            signals_metadata = {
                header['label']:{
                    'unit': header['dimension'],
                    'physical_range': (header['physical_min'], header['physical_max']),
                    'sample_frequency': header['sample_frequency'],                 
                    'prefilter': header['prefilter'],
                }
                for i, header 
                in enumerate(headers)
                if header['label'] in args.psg_montage and i not in duplicate_idx
            }

        # Resample signals to 100Hz and apply anti-aliasing and standardization
        signals, signals_metadata = preprocess_psg_dict(
            signals,
            signals_metadata,
            fs_out = args.psg_resample_freq,
            filter_order = args.psg_filter_order,
            standardization = args.standardize_psg,
        )

        # Rename psg channels if specified
        signals = {
            (args.psg_rename_map[name]
            if name in args.psg_rename_map.keys()
            else name): signal 
            for name, signal in signals.items()
        }
        signals_metadata = {
            (args.psg_rename_map[name]
            if name in args.psg_rename_map.keys()
            else name): metadata 
            for name, metadata in signals_metadata.items()
        }
        print(list(signals.keys()))

        ## -- Actigraphy --
        print(f'Processing {act_h5}')
        # Read actigraphy h5 and infer timestamps
        acc_df, fs_acc = io_utils.read_amazfit_h5(act_h5, study_start)

        # if subject_id == 'STNF00249':
        #     acc_df.iloc[:fs_acc*60*384,:] = 0
        #     #acc[:fs*60*384,:] = 0
        # elif subject_id == 'STNF00264':
        #     acc_df.iloc[:fs_acc*60*20,:] = 0
        # elif subject_id == 'STNF00486':
        #     acc_df.iloc[:fs_acc*60*72,:] = 0
        # elif subject_id == 'STNF00392':
        #     acc_df.iloc[:fs_acc*60*292,:] = 0
        # elif subject_id == 'STNF00182':
        #     print('Skipping due to missing data\n')
        #     continue
        # elif subject_id == 'STNF00370':
        #     print('Skipping due to missing data\n')
        #     continue
        
        # Currently this drops ppg column
        acc_df_preproc, all_info = preprocess_actigraphy_df(
            data_df = acc_df,
            x_col = 'x',
            y_col = 'y',
            z_col = 'z',
            fs_in = fs_acc,
            fs_out = args.actigraphy_resample_freq,
            gravity_calibration = True,
            calib_cube = args.calib_cube,
            input_unit_divisor = args.input_unit_divisor,
            scale_by_force = args.scale_by_force,
            max_iters_calib = 1000,
            verbose = False,
        )
        
        if all_info['calibration_diagnostics']['CalibOK'] == 0:
            calib_failed.append(subject_id)
            print('Skipping file due to failed calibration\n')
            continue

        save_info = all_info['calibration_diagnostics']
        
        # Drop NA segments
        acc_df_preproc_na = acc_df_preproc.isna().any(axis=1)
        print(f'Dropping {acc_df_preproc_na.sum()} NaN values')
        acc_df_preproc = acc_df_preproc.loc[~acc_df_preproc_na]

        # Check for gaps in data after dropping nans
        unique_fs = acc_df_preproc.index.diff().total_seconds()[1:].round(3).unique()
        if len(unique_fs) > 1:
            raise RuntimeError(f'Found multiple fs after dropping NaNs: {unique_fs}')

        # Calculate start and end time of EDF recording from random signal
        sig = signals[list(signals.keys())[0]]
        fs = signals_metadata[list(signals.keys())[0]]['sample_frequency']
        edf_time = pd.date_range(start=study_start, periods=len(sig), freq=f'{1/fs}s')

        act_before_edf_start = acc_df_preproc.index[0] < edf_time[0]
        act_after_edf_end = acc_df_preproc.index[-1] > edf_time[-1]

        start_time = edf_time[0] if act_before_edf_start else acc_df_preproc.index[0]
        end_time = edf_time[-1] if act_after_edf_end else acc_df_preproc.index[-1]
    
        if (end_time - start_time).total_seconds()/60**2 < args.min_recording_hours:
            print(f'Less than {args.min_recording_hours} hours of concurrent data for {act_h5}, skipping. \n')
            short_recording.append(act_h5)
            continue
        
        # Check for nonwear in remaining data
        std_roll = acc_df_preproc.rolling(window = '10s').std().max(axis=1)
        no_move = std_roll.rolling(window='7200s').max() < 0.015
        no_move_prcnt = no_move.sum() / len(acc_df_preproc)
        if no_move_prcnt > 0.2:
            print(f'More than 20% ({round(no_move_prcnt*100)}%) possible nonwear detected for {act_h5}, skipping. \n')
            errors.append(act_h5)
            continue

        ## -- Annotations --
        # Get paths to annotation and edf files based on the subject id
        annotation_paths = {}
        for annot_file, tag in args.annotation_file_tags.items():
            try:
                tag_path = glob.glob(f'{subject_dir}/**/{annot_file}*.txt', recursive=True)[0]
                annotation_paths[tag] = tag_path
            except IndexError:
                print(f'No file found for {tag} in {subject_dir}')

        annot_df = io_utils.get_annotations(acc_df_preproc.index, annotation_paths, study_start)

        annot_df['apnea'] = annot_df['apnea'].map({
            'none': 'none', 
            'RERA': 'none',
            'Obstructive Apnea': 'oa',
            'Hypopnea': 'ha',
            'Apnea': 'oa',
            'Central Apnea': 'ca',
            'Mixed Apnea': 'oa',
        })
        if annot_df['apnea'].isna().any():
            print(f'Apnea label unaccounted for for {subject_dir}')
            continue

        # Count AHI by apnea/hypopnea (onset+offset)//2
        num_apnea = (annot_df['apnea'] == 'none').diff().sum()//2 
        hours = (annot_df.index[-1] - annot_df.index[0]).total_seconds()/60**2
        ahi = round(num_apnea/hours,3)
        ahis.append(ahi)

        # Get sleep stage amounts
        stages_hours = {}
        for stage in annot_df['hypnogram_'].unique():
            stage_hours = round(
                (annot_df['hypnogram_'] == stage).sum() 
                / args.actigraphy_resample_freq / 60**2,
                3
            )
            stages_count[stage] += stage_hours
            stages_hours[stage] = stage_hours

        write_h5_acc_psg(
            out_file,
            accelerometry = acc_df_preproc,
            acc_info = save_info,
            x_col = 'x_preprocessed',
            y_col = 'y_preprocessed',
            z_col = 'z_preprocessed',
            annotations = annot_df.astype(str),
            chunk_size_sec = args.hdf5_chunksize_sec,
            psg_data = signals,
            psg_metadata = signals_metadata,
            study_start = study_start,
            ahi = ahi,
            stage_hours = stages_hours,
        )
        print(f'Saved {out_file}')
        print(' ')
        
    print(f'{len(calib_failed)} files dropped due to failed calibration.')
    print(f'{len(short_recording)} files dropped due to short recordings.')
    print(f'{len(errors)} files dropped due to other erros.')

    ahi_a = np.array(ahis)
    apnea_dict = {
        'none': (ahi_a < 5).sum(),
        'mild': (ahi_a < 15).sum() - (ahi_a < 5).sum(),
        'moderate': (ahi_a < 30).sum() - (ahi_a < 15).sum(),
        'severe': (ahi_a >= 30).sum()
    }
    for degree, num in apnea_dict.items():
        print(f'Number of individual with {degree} apnea: {num}/{len(ahi_a)}')

    total_time = 0
    for stage, time in stages_count.items():
        total_time += time
    for stage, time in stages_count.items():
        print(f'% time in {stage}: {round(time/total_time,3)}')

if __name__ == '__main__':
    args = parse_args()
    read_config(args.config_json_path, args)
    main(args)