import os
import glob
import h5py
import numpy as np
import pandas as pd

def get_annotations(
        time_index: pd.DatetimeIndex,
        annotation_paths: dict,
        study_start: pd.Timestamp,
    ) -> pd.DataFrame:
    annot_df = pd.DataFrame(index = time_index)
    # Add annotations to the dataframe
    for tag, annot_path in annotation_paths.items():                                
        event_df, tag_type = read_labels(annot_path, study_start)

        # None means no event, missing means missing data
        if tag_type == 'event':
            annot_df[tag] = 'none'
        elif tag_type == 'continuous':
            annot_df[tag] = 'missing'
        
        for i, event in event_df.iterrows():
            # Use event['start'] and event['duration_seconds'] 
            # to find the corresponding rows in annot_df
            start = event['start']
            duration = pd.Timedelta(event['duration_seconds'], unit='s')
            end = start + duration
            annot_df.loc[start:end, tag] = event['event_label']
        annot_df[tag] = annot_df[tag].str.strip()

        if tag == 'hypnogram':
            annot_df.loc[annot_df[tag] == 'A', tag] = 'missing'
            annot_df.loc[annot_df[tag] == 'Artifact', tag] = 'missing'
            annot_df.loc[annot_df[tag] == 'NS', tag] = 'missing'
            
            annot_df['sleep_wake'] = annot_df[tag].map({
                'Wake':'wake',
                'N1':'sleep',
                'N2':'sleep',
                'N3':'sleep',
                'Rem':'sleep',
                'missing':'missing',
            })
            annot_df['rem_nrem'] = annot_df[tag].map({
                'Wake':'wake',
                'N1':'nrem',
                'N2':'nrem',
                'N3':'nrem',
                'Rem':'rem',
                'missing':'missing',
            })
            annot_df['light_deep_rem'] = annot_df[tag].map({
                'Wake':'wake',
                'N1':'light',
                'N2':'light',
                'N3':'deep',
                'Rem':'rem',
                'missing':'missing',
            })
            annot_df['hypnogram_'] = annot_df[tag].map({
                'Wake': 'wake',
                'N1': 'n1',
                'N2': 'n2',
                'N3': 'n3',
                'Rem': 'rem',
                'missing': 'missing',
            })

    return annot_df

def read_labels(label_file_path:str, study_start:pd.Timestamp) -> pd.DataFrame:
    """Read event labels from a clinical annotation file and return a 
    pandas.DataFrame with the event labels, start times, seconds since 
    study start, and duration of each event in seconds. The format of 
    the file is detected from the header of the file and the file is
    read accordingly. 

    Example of an annotation file with 'discrete' format:
    -------------------------------------
    Signal ID: SchlafProfil\\profil
    Start Time: 11/18/2021 8:39:30 PM
    Unit: 
    Signal Type: Discret
    Events list: N3,N2,N1,Rem,Wake,Artifact
    Rate: 30 s

    20:39:30,000; A
    20:40:00,000; Wake
    20:40:30,000; Wake
    .
    .
    .
    06:35:00,000; N2
    06:35:30,000; N2
    06:36:00,000; N2
    -------------------------------------
    Example of an annotation file with 'impulse' format:
    -------------------------------------
    Signal ID: FlowD\\flow
    Start Time: 11/18/2021 8:39:46 PM
    Unit: s
    Signal Type: Impuls

    00:41:31,513-00:41:42,736; 11;Hypopnea
    01:31:15,326-01:31:26,333; 11;Hypopnea
    01:35:52,664-01:36:03,887; 11;Hypopnea
    01:44:01,458-01:44:13,544; 12;Hypopnea
    02:15:54,551-02:16:06,853; 12;Hypopnea
    02:24:31,242-02:25:00,594; 29;Hypopnea
    02:52:37,932-02:52:53,040; 15;Central Apnea
    -------------------------------------

    args:
        label_file_path:
            String path to the file containing event labels.

        study_start: 
            A pandas.Timestamp with the start time of the sleep study.
    Returns:
        events: 
            A pandas.Dataframe with rows for event with columns:
            'event_label': 
                The label for each event,
            'start': 
                The start of the event as a pandas.Timestamp,
            'seconds_since_study_start': 
                The start of the event in seconds from study start.
            'duration_seconds': 
                The duration of the event in seconds.
    """
    # Read the header of the file to determine how to read the file
    signal_type, header_end = read_header(label_file_path)

    # Read file with pandas from header end
    label_file = pd.read_csv(
        label_file_path,
        sep = ';',
        skiprows = header_end,
        header = None
    )

    if signal_type == 'Impuls':
        events = events_from_impulse_labels(label_file,study_start)
        signal_type = 'event'

    elif signal_type == 'Discret' or signal_type == 'Analog':
        events = events_from_discrete_or_analog_labels(label_file,study_start)
        signal_type = 'continuous'

    return events, signal_type

def read_header(label_file_path:str) -> tuple:
    with open(label_file_path) as file:
        for line_num, line in enumerate(file):
            # The signal type determines how to read the file
            if 'Signal Type:' in line:
                signal_type = line[len('Signal Type:'):].strip()    
            # Detect first empty line
            if line.strip() == '':
                header_end = line_num
                break
    return signal_type, header_end

def events_from_impulse_labels(
        label_file: pd.DataFrame,
        study_start: pd.Timestamp,
    ) -> pd.DataFrame:
    # Read the last column as the event label
    events = pd.DataFrame({'event_label':label_file.iloc[:,-1]})

    # Read the timestamp column as an interval
    events[['start','end']] = label_file[0].str.split(
        pat = '-', expand=True
    )
    # Convert timestamps to datetime and adjust date to study start
    for col in ['start','end']:
        events[col] = pd.to_datetime(
            events[col],
            format="%H:%M:%S,%f"
        ).apply(
            lambda x: change_date(x,study_start)
        )
    # Calculate the seconds since study start
    events['seconds_since_study_start'] = (
        events['start']-study_start
    ).dt.total_seconds()

    # Calculate the duration of each event in seconds
    events['duration_seconds'] = (
        events['end']-events['start']
    ).dt.total_seconds()

    # Drop 'end' column to adhere to format
    events = events.drop(columns=['end'])
    return events

def events_from_discrete_or_analog_labels(
        label_file: pd.DataFrame,
        study_start: pd.Timestamp
    ) -> pd.DataFrame:

    # Read the last column as the event label
    events = pd.DataFrame({'event_label':label_file.iloc[:,-1]})

    # Convert timestamps to datetime and adjust date to study start
    events['start'] = pd.to_datetime(
        label_file[0],
        format = "%H:%M:%S,%f"
    ).apply(
        lambda x: change_date(x,study_start)
    )

    # Calculate the seconds since study start
    events['seconds_since_study_start'] = (
        events['start']-study_start
    ).dt.total_seconds()

    # Calculate the duration of each label until the next label
    events['duration_seconds'] = events['seconds_since_study_start'].diff().shift(-1)

    # Find the rows where the label changes
    label_change = (events.event_label != events.event_label.shift())
    
    # Calculate the number of consecutive events with the same label
    events['event_num'] = label_change.cumsum()

    # Calculate the cumulative duration of each event
    cumulative_duration = events.groupby(
        'event_num'
    ).sum('duration_seconds')['duration_seconds']

    # Keep only the last row of each event 
    events = events.loc[label_change].copy().reset_index(drop=True)

    # Add the cumulative duration of each event
    events['duration_seconds'] = cumulative_duration.reset_index(drop=True)

    # Drop 'event_num' column to adhere to format
    events = events.drop(columns=['event_num'])
    return events

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


def read_amazfit_h5(file_path, study_start):
    with h5py.File(file_path) as h5:
        acc_dataset = h5['acc']
        fs_acc = acc_dataset.attrs['fs']
        acc_array = np.array(acc_dataset)

        ppg_dataset = h5['ppg']
        # fs_ppg = ppg_dataset.attrs['fs']
        ppg_array = np.array(ppg_dataset)

        data = pd.DataFrame(acc_array, columns=['x', 'y', 'z'])
        data['ppg'] = ppg_array
        
        # Interpolate time stamps using study start time and sampling rate 
        data['time'] = pd.date_range(
            start = study_start, 
            periods = len(data), 
            freq = f'{1 / fs_acc}s',
        )
        data.set_index('time', inplace=True)

    return data, fs_acc

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
    ) -> None:
    '''Write the data to a h5 file'''
    with h5py.File(outfile, 'w') as f:
        f.create_group('annotations')
        f.create_group('data')
        f.attrs.create('start_time', study_start.strftime('%Y-%m-%d %H:%M:%S'))

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