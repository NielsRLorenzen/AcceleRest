import os
import h5py
import json
import glob
import actipy
import argparse
import numpy as np
import pandas as pd
from pyarrow import csv
from collections import defaultdict

from utils.preprocess_actigraphy import preprocess_actigraphy_df
from utils.preprocess_psg import preprocess_psg_dict
from utils.write_h5 import write_h5_acc_psg

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--subject_csv_path',
        type = str,
        help='Path to the directory containing the subject directories',
        default = './dreamt-2.0.0/data_100Hz/',
    )
    argparser.add_argument(
        '--output_dir',
        type=str,
        help = 'Directory to save the output h5 files',
        default = './data/dreamt/'
    )
    argparser.add_argument(
        '--config_json_path',
        type=str,
        help = 'Path to json config file',
        default = './dreamt/preproc_config.json'
    )
    return argparser.parse_args()

def read_config(config_path: str, args):
    with open(config_path, 'r') as f:
        config = json.load(f)
    args.__dict__.update(config)

def main(args):
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

    csv_files = glob.glob(args.subject_csv_path+'*.csv')
    for f in csv_files:
        subject_id = os.path.basename(f)[:-11]

        # Skip if the output file already exists
        out_file = os.path.join(args.output_dir, f'{subject_id}.h5')
        if os.path.exists(out_file):
            print(f'{out_file} already exists, skipping\n')
            continue
    
        print(' ')    
        print(f'Processing {f}')
        
    
        df = csv.read_csv(f).to_pandas().set_index(keys='TIMESTAMP')
        startime_proxy = pd.Timestamp('2025-01-01 23:00:00')
        df.index = (startime_proxy + pd.to_timedelta(df.index, unit='s')).rename('time')

        ## Trim leading flat segment of missing values
        acc_df = df[['ACC_X', 'ACC_Y', 'ACC_Z']]
        # Find when flat segements changes to active
        std_filt = (acc_df.rolling(window = '10s').std().sum(axis=1) > 1e-6).diff()
        # Trim up to the end of first flat segment
        trim_idx = acc_df.index[std_filt.argmax()]
        df = df.loc[trim_idx:]

        if (df.index[-1] - df.index[0]).total_seconds() < 60**2 * args.min_recording_hours:
            print(f'Less than {min_recording_hours} hours of data after trimming file {f}, skipping\n')
            continue
        
        # Get trimmed dataframes
        acc_df = df[['ACC_X', 'ACC_Y', 'ACC_Z']].rename(
            columns={
                'ACC_X': 'x',
                'ACC_Y': 'y',
                'ACC_Z': 'z',
            }
        )
        psg_df = df[args.psg_montage]
        stages = df[['Sleep_Stage']]
        apnea = df[['Obstructive_Apnea', 'Central_Apnea', 'Hypopnea']]
        
        ## -- PSG --
        psg_dict = {signal:psg_df[signal].values for signal in psg_df.columns}
        psg_metadata = defaultdict(dict)
        for signal in psg_dict.keys():
            # Adjust assumed sample freq by a small factor (0.9985) for correct actigraphy aligment
            psg_metadata[signal]['sample_frequency'] = 100 * args.drift_correction_factor

        # Resample signals to 100Hz and apply anti-aliasing and standardization
        signals, signals_metadata = preprocess_psg_dict(
            psg_dict,
            psg_metadata,
            fs_out = args.psg_resample_freq,
            filter_order = args.psg_filter_order,
            standardization = args.standardize_psg,
        )

        # Cut excess time due to drift correction off signals
        cutoff = int(len(psg_df)/(100/args.psg_resample_freq)) + 1
        for channel, signal in signals.items():
            signals[channel] = signal[:cutoff]

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
        
        ## -- Actigraphy
        acc_df_preproc, all_info = preprocess_actigraphy_df(
            data_df = acc_df,
            x_col = 'x',
            y_col = 'y',
            z_col = 'z',
            fs_in = 100,
            fs_out = args.actigraphy_resample_freq,
            gravity_calibration = True,
            calib_cube = args.calib_cube,
            input_unit_divisor = args.input_unit_divisor,
            scale_by_force = args.scale_by_force,
            max_iters_calib=1000,
            verbose = False,
        )

        if all_info['calibration_diagnostics']['CalibOK'] == 0:
            calib_failed.append(subject_id)
            print('Skipping file due to failed calibration\n')
            continue

        save_info = all_info['calibration_diagnostics']

        # Check for nonwear
        std_roll = acc_df_preproc.rolling(window = '10s').std().max(axis=1)
        no_move = std_roll.rolling(window='7200s').max() < 0.015
        no_move_prcnt = no_move.sum() / len(acc_df_preproc)
        if no_move_prcnt > 0.2:
            print(f'More than 20% ({round(no_move_prcnt*100)}%) possible nonwear detected for {subject_id}, skipping. \n')
            errors.append(subject_id)
            continue

        if np.any(acc_df_preproc.isna().any(axis=1)):
            print(f'NaNs detected in the actigrapy signal for {subject_id}.')
            errors.append(subject_id)
            continue

        ## -- Annotations --
        # Apply same drift correction factor to labels as to PSG signals
        duration = len(stages) / (100 * args.drift_correction_factor)
        time_points = np.linspace(0, duration, num=len(stages))
        new_index = (stages.index[0] + pd.to_timedelta(time_points, unit='s')).rename('time')
        apnea = apnea.set_index(new_index)
        stages = stages.set_index(new_index)

        # Resample annotations to actigraphy sample freq
        apnea_rs, info = actipy.processing.resample(apnea, args.actigraphy_resample_freq)
        stages_rs, info = actipy.processing.resample(stages, args.actigraphy_resample_freq)

        # Combine apneas into one series
        annot_df = pd.DataFrame(index=apnea_rs.index, columns=['apnea']).fillna('none')
        annot_df.loc[apnea_rs['Hypopnea'].notna(), 'apnea'] = 'ha'
        annot_df.loc[apnea_rs['Obstructive_Apnea'].notna(), 'apnea'] = 'oa'
        annot_df.loc[apnea_rs['Central_Apnea'].notna(), 'apnea'] = 'ca'

        # Relabel sleep stages
        annot_df['hypnogram'] = stages_rs['Sleep_Stage'].map({
            'W': 'wake',
            'N1': 'n1',
            'N2': 'n2',
            'N3': 'n3',
            'R': 'rem',
            'Missing': 'missing',
        })

        annot_df['sleep_wake'] = stages_rs['Sleep_Stage'].map({
            'W': 'wake',
            'N1': 'sleep',
            'N2': 'sleep',
            'N3': 'sleep',
            'R': 'sleep',
            'Missing': 'missing',
        })

        annot_df['rem_nrem'] = stages_rs['Sleep_Stage'].map({
            'W': 'wake',
            'N1': 'nrem',
            'N2': 'nrem',
            'N3': 'nrem',
            'R': 'rem',
            'Missing': 'missing',
        })

        annot_df['light_deep_rem'] = stages_rs['Sleep_Stage'].map({
            'W': 'wake',
            'N1': 'light',
            'N2': 'light',
            'N3': 'deep',
            'R': 'rem',
            'Missing': 'missing',
        })

        # Cut excess time due to drift correction off annotations
        annot_df = annot_df.loc[:acc_df_preproc.index[-1]]

        # Count AHI by apnea/hypopnea (onset+offset)//2
        num_apnea = (annot_df['apnea'] == 'none').diff().sum()//2 
        hours = (annot_df.index[-1] - annot_df.index[0]).total_seconds()/60**2
        ahi = round(num_apnea/hours,3)
        ahis.append(ahi)

        # Get sleep stage amounts
        stages_hours = {}
        for stage in annot_df['hypnogram'].unique():
            stage_hours = round(
                (annot_df['hypnogram'] == stage).sum() 
                / args.actigraphy_resample_freq / 60**2,
                3
            )
            stages_count[stage] += stage_hours
            stages_hours[stage] = stage_hours

        # Write the data to an h5 file
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
            study_start = acc_df_preproc.index[0],
            ahi = ahi,
            stage_hours = stages_hours,
        )
        print(f'Finished writing {out_file}')
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