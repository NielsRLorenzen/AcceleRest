import pandas as pd
import numpy as np
import edfio
import h5py
import warnings

from pyarrow import csv
from datetime import datetime
from pyedflib.edfreader import EdfReader
from collections import defaultdict

## IO UTILS
def get_subjects_df(
        keep_subject_ids: set,
        act_files: list,
        edf_files: list,
        hypno_files: list,
        event_files: list,
    ) -> pd.DataFrame:
    rows = []
    for subject_id in keep_subject_ids:
        act_file = [file for file in act_files if subject_id in file]
        edf_file = [edf for edf in edf_files if subject_id in edf]
        if (len(act_file) > 1) or (len(edf_file) > 1):
            print('More than one actigraphy or edf files found:')
            print(act_file)
            print(edf_file)
            continue
        else:
            act_file, edf_file = act_file[0], edf_file[0]

            # Get and check paths to annotation files based on the subject id
            hypno_path = [hypno for hypno in hypno_files if subject_id in hypno]
            if len(hypno_path) == 0:
                print(f'No hypno file found for {subject_id}')
            elif len(hypno_path) > 1:
                print(f'Multiple hypno files found for {subject_id}:')
                print(hypno_path)
            else:
                hypno_path = hypno_path[0]
                
            event_path = [event for event in event_files if subject_id in event]
            if len(event_path) == 0:
                print(f'No event file found for {subject_id}')
            elif len(event_path) > 1:
                print(f'Multiple event files found for {subject_id}:')
                print(event_path)
            else:
                event_path = event_path[0]

            rows.append([subject_id, act_file, edf_file, hypno_path, event_path])
    subjects_df = pd.DataFrame(data=rows, columns=['id', 'act', 'edf', 'hypno', 'event'])
    return subjects_df

def _read_actigraphy_header(file_path):
    '''Read the header of a TBI actigraphy csv file.'''
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                line = line.split()
                for j, word in enumerate(line):
                    if word == 'Hz':
                        sample_rate = int(line[j-1])
            if 'Start Time' in line:
                start_time = line.split()[-1].strip()
            if 'Start Date' in line:
                start_date = line.split()[-1].strip()
                start_time = datetime.strptime(
                    start_date + ' ' + start_time, 
                    '%m/%d/%Y %H:%M:%S'
                )
            if 'Timestamp' in line:
                header_end = i
                break
    return start_time, sample_rate, header_end

def read_actigraphy_csv(csv_path: str):
    # Read csv header for metadata and file start
    start_time, sample_rate, header_end = _read_actigraphy_header(csv_path)

    # Read csv file with pyarrow and convert to pandas
    read_options = csv.ReadOptions(skip_rows=header_end)
    df = csv.read_csv(csv_path, read_options = read_options).to_pandas()
    
    # Convert Timestamps to datetime index for actipy compliance
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'],
        format='%m/%d/%Y %H:%M:%S.%f',
    )
    df = df.set_index('Timestamp')
    
    # Check if start time from file header corersponds to first timestamp
    if start_time != df.index[0]:
        print(f'Header start time: {start_time}, does not equal first timestamp: {df.index[0]}')

    # Check if sample rate from header corresponds to timestamps
    file_sample_rate = round(1/df.index.diff().mean().total_seconds(),2)
    if sample_rate != file_sample_rate:
        print(f'Header sample rate {sample_rate}, does not equal file sample rate {file_sample_rate}')
    unique_rates = df.index.diff().total_seconds().unique()
    if unique_rates.notna().sum() > 1:
        print(f'Variable sample rates found: {unique_rates}')
    return df

def change_date(timestamp, study_start) -> pd.Timestamp:
    """Change the date of a timestamp to the date of study_start and
    change the date to the next day if the timestamp crosses 12am.

    args:
        timestamp:
            A pandas Timestamp.
        study_start:
            A pandas Timestamp.

    returns:
        updated_timestamp:
            A pandas Timestamp.
    """
    #if the hour of the timestamp is lower than that of the study start,
    #This indicates that it is a new date, so add one to day
    if timestamp.hour < study_start.hour:
        if study_start.is_year_end:
            timestamp = timestamp.replace(
                year=study_start.year + 1,
                month=1,
                day=1
            )
        elif study_start.is_month_end:
            timestamp = timestamp.replace(
                year=study_start.year,
                month=study_start.month +1,
                day=1
            )
        else:
            timestamp = timestamp.replace(
                year=study_start.year,
                month=study_start.month,
                day=study_start.day+1
            )
    else:
        timestamp = timestamp.replace(
            year=study_start.year,
            month=study_start.month,
            day=study_start.day
        )
    return timestamp

def read_events(event_path: str, study_start: pd.Timestamp) -> pd.DataFrame:
    event_df = pd.read_csv(event_path)
    # Convert the start time to a pandas Timestamp
    event_df['start'] = pd.to_datetime(event_df['Time'], format='%I:%M:%S %p')
    # Change the date of the start time to the date of the study start
    event_df['start'] = event_df['start'].apply(lambda x: change_date(x, study_start))
    event_df['duration'] = event_df['Duration'].apply(lambda x: pd.Timedelta(x, 's'))
    event_df['end'] = event_df['start'] + event_df['duration'] - pd.Timedelta(1, 'ms')
    return event_df

def get_events(
        time_index: pd.DataFrame,
        event_path: str,
        study_start: pd.Timestamp,
    ) -> pd.DataFrame:
    # Get event-based dataframe, ie. start, end, event_type -format
    event_df = read_events(event_path, study_start)

    # Setup sample-wise timeseries dataframe for the hypnogram
    samplewise_df = pd.DataFrame(index = time_index)
    
    # Translate events to sample-wise timeseries
    for i, row in event_df.iterrows():
        samplewise_df.loc[row['start']:row['end'], 'event'] = ( 
            # Only use events recorded during sleep
            row['Type'] 
            if row['Stage'] 
            in ['REM', 'N1', 'N2', 'N3']
            else 'none'
        )
    samplewise_df.fillna('none', inplace=True)
    # Create specific columns for Arousals and Apneas
    arousal_map = defaultdict(lambda: 'none')
    arousal_map['Arousal'] = 'arousal'
    samplewise_df['arousal'] = samplewise_df['event'].map(arousal_map)

    apnea_map = defaultdict(lambda: 'none')
    apnea_map['Hypopnea'] = 'ha'
    apnea_map['Central Apnea'] = 'ca'
    apnea_map['Obstructive Apnea'] = 'oa'
    samplewise_df['apnea'] = samplewise_df['event'].map(apnea_map)
    return samplewise_df

def read_hypnogram(hypno_path: str, study_start: pd.Timestamp) -> pd.DataFrame:
    hypno_df = pd.read_csv(hypno_path)
    hypno_df['start'] = pd.to_datetime(hypno_df['Start Time '], format='%I:%M:%S %p')
    # Change the date of the time to the date of the study start
    hypno_df['start'] = hypno_df['start'].apply(lambda x: x.replace(
        year = study_start.year,
        month = study_start.month,
        day = study_start.day,
    ))

    # Make sure dates increment
    time_diffs = hypno_df['start'].diff()
    day_increments = (time_diffs < pd.Timedelta(0)).cumsum()
    hypno_df['start'] = hypno_df['start'] + pd.to_timedelta(day_increments, unit='D')

    # # Convert the start time to a pandas Timestamp
    # hypno_df['start'] = pd.to_datetime(hypno_df['Start Time '], format='%I:%M:%S %p')
    # # Change the date of the start time to the date of the study start
    # study_start_ = pd.Timestamp(
    #     study_start.date().strftime('%Y-%m-%d') 
    #     + ' ' 
    #     + hypno_df['start'][0].time().strftime('%H:%M:%S')
    # )
    # hypno_df['start'] = hypno_df['start'].apply(lambda x: change_date(x, study_start_))
    # hypno_df['end'] = hypno_df['start'].shift(-1) - pd.Timedelta(1, 'ms')
    return hypno_df

def get_hypnogram(
    time_index: pd.DatetimeIndex,
    hypno_path: str,
    study_start: pd.Timestamp,
) -> pd.DataFrame:
    # Read hypnogram into a start_time: stage df
    event_df = read_hypnogram(hypno_path, study_start)
    hypnogram = event_df.set_index('start')['Sleep Stage']

    # Setup sample-wise timeseries dataframe for the hypnogram
    samplewise_df = pd.DataFrame(index=time_index)

    # Translate event-based hypnogram to sample-wise time-series
    samplewise_df = pd.merge_asof(
        samplewise_df,
        hypnogram,
        left_index = True,
        right_index = True,
        tolerance = pd.Timedelta(31, unit='s'),
    )

    # Fill NANs caused by inference tolerance at the end of signal
    samplewise_df['Sleep Stage'] = samplewise_df['Sleep Stage'].fillna('missing')

    # Combine sleep stages into various detail-levels
    samplewise_df['sleep_wake'] = samplewise_df['Sleep Stage'].map(
        {
        'NS': 'missing',
        'WK': 'wake',
        'N1': 'sleep',
        'N2': 'sleep',
        'N3': 'sleep',
        'REM': 'sleep',
        'missing': 'missing',
        }
    )
    samplewise_df['rem_nrem'] = samplewise_df['Sleep Stage'].map(
        {
        'NS': 'missing',
        'WK': 'wake',
        'N1': 'nrem',
        'N2': 'nrem',
        'N3': 'nrem',
        'REM': 'rem',
        'missing': 'missing',
        }
    )
    samplewise_df['light_deep_rem'] = samplewise_df['Sleep Stage'].map(
        {
        'NS': 'missing',
        'WK': 'wake',
        'N1': 'light',
        'N2': 'light',
        'N3': 'deep',
        'REM': 'rem',
        'missing': 'missing',
        }
    )
    samplewise_df['hypnogram'] = samplewise_df['Sleep Stage'].map(
        {
        'NS': 'missing',
        'WK': 'wake',
        'N1': 'n1',
        'N2': 'n2',
        'N3': 'n3',
        'REM': 'rem',
        'missing': 'missing',
        }
    )
    return samplewise_df

def read_edf(edf_path: str, psg_montage: list):
    try:
        # Might eventually be changed to use edfio
        with EdfReader(edf_path) as edf_file:
            study_start = pd.Timestamp(edf_file.getStartdatetime())
            
            labels = edf_file.getSignalLabels()
            l, c = np.unique(labels, return_counts=True)
            if np.any(c > 1):
                print(f'More than one {l[c > 1]} channel found in {edf_path}')
            
            signals = {
                label: edf_file.readSignal(i) 
                for i, label 
                in enumerate(labels) 
                if label in psg_montage
            }
            
            headers = edf_file.getSignalHeaders()
            
            signals_metadata = {
                header['label']:{
                    'unit': header['dimension'],
                    'physical_range': (header['physical_min'], header['physical_max']),
                    'sample_frequency': header['sample_frequency'],                 
                    'prefilter': header['prefilter'],
                }
                for header 
                in headers
                if header['label'] in psg_montage
            }
    except OSError as e:
        # 5 files have filesizes that are not EDF compliant
        # The edfio package does not enforce this contstraint
        print(f'Error occured when reading {edf_path}')
        print(e)
        print('Trying with edfio')
        edf = edfio.read_edf(edf_path)
        study_start = pd.Timestamp(
            datetime.combine(edf.startdate, edf.starttime)
        )
        
        labels = list(edf.labels)
        l, c = np.unique(labels, return_counts=True)
        if np.any(c > 1):
            warnings.warn(f'More than one {l[c > 1]} channel found in {edf_path}')
        
        signals = {
            signal.label: signal.data 
            for signal 
            in edf.signals
            if signal.label in psg_montage
        }
        
        signals_metadata = {
            signal.label:{
                'unit': signal.physical_dimension,
                'physical_range': signal.physical_range,
                'sample_frequency': signal.sampling_frequency,
                'prefilter': signal.prefiltering,
            }
            for signal
            in edf.signals
            if signal.label in psg_montage
        }
    not_found = set(psg_montage).difference(set(signals.keys()))
    if len(not_found) > 0:
        print(f'Following channels in psg montage not found in {edf_path}:')
        print(not_found)
    return signals, signals_metadata, study_start

def write_h5(
    outfile: str,
    accelerometry: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    acc_info: dict,
    annotations: pd.DataFrame,
    psg_data: dict[np.array],
    psg_metadata: dict[dict],
    study_start: pd.Timestamp,
    chunk_size_sec: int = 600,
    ahi: float = None,
    stage_hours: dict = None,
) -> None:
    '''Write the data to a h5 file'''
    with h5py.File(outfile, 'w') as f:
        f.create_group('annotations')
        f.create_group('data')
        f.attrs.create('start_time', study_start.strftime('%Y-%m-%d %H:%M:%S'))
        if ahi is not None:
            f.attrs.create('ahi', ahi)
        if stage_hours is not None:
            for stage, hours in stage_hours.items():
                f.attrs.create(f'{stage}_hours', hours)

        # Write PSG data
        for label, signal in psg_data.items():
            ch_fs = psg_metadata[label]['sample_frequency']
            dataset = f['data'].create_dataset(
                label,
                data=signal.astype(np.float32),
                chunks=(chunk_size_sec * ch_fs,),
            )
            for attribute, value in psg_metadata[label].items():
                dataset.attrs.create(attribute, value)
        
        # Write the calibrated accelerometry data
        acc_fs = round(1/accelerometry.index.diff().mean().total_seconds(), 2)
        acc = accelerometry[[x_col, y_col, z_col]].values
        dataset = f['data'].create_dataset(
            f'accelerometry',
            data=acc.T.astype(np.float32),
            chunks=(acc.shape[1], chunk_size_sec * acc_fs),
        )
        for key, value in acc_info.items():
            dataset.attrs.create(key, value)
        dataset.attrs.create('sample_frequency', acc_fs)
            
        # Write the annotations
        annot_fs = round(1/annotations.index.diff().mean().total_seconds(), 2)
        for field in annotations.columns:
            annotation = annotations[field]
            dataset = f['annotations'].create_dataset(
                f'{field}',
                data=annotation.to_numpy(),
                dtype=h5py.string_dtype(encoding='utf-8'),
                chunks=(chunk_size_sec * annot_fs,),
            )
            dataset.attrs.create('sample_frequency', annot_fs)