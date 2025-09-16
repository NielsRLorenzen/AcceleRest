import os
import glob
import h5py
import numpy as np
import pandas as pd

def get_annotations(
        data,
        annotation_paths: dict,
        study_start: pd.Timestamp,
    ) -> pd.DataFrame:
    # Add annotations to the dataframe
    for tag, annot_path in annotation_paths.items():                                
        annot_df, tag_type = read_labels(annot_path, study_start)

        # None means no event, missing means missing data
        if tag_type == 'event':
            data[tag] = 'none'
        elif tag_type == 'continuous':
            data[tag] = 'missing'
        
        for i, event in annot_df.iterrows():
            # Use event['start'] and event['duration_seconds'] 
            # to find the corresponding rows in data
            start = event['start']
            duration = pd.Timedelta(event['duration_seconds'], unit='s')
            end = start + duration
            data.loc[start:end, tag] = event['event_label']
        data[tag] = data[tag].str.strip()

        if tag == 'hypnogram':
            data.loc[data[tag] == 'A', tag] = 'missing'
            data.loc[data[tag] == 'Artifact', tag] = 'missing'

            data['sleep_wake'] = data[tag].map({
                'Wake':'wake',
                'N1':'sleep',
                'N2':'sleep',
                'N3':'sleep',
                'Rem':'sleep',
                'A':'missing',
                'Artifact':'missing',
                'missing':'missing',
            })
            data['rem_nrem'] = data[tag].map({
                'Wake':'wake',
                'N1':'nrem',
                'N2':'nrem',
                'N3':'nrem',
                'Rem':'rem',
                'A':'missing',
                'Artifact':'missing',
                'missing':'missing',
            })
            data['light_deep_rem'] = data[tag].map({
                'Wake':'wake',
                'N1':'light',
                'N2':'light',
                'N3':'deep',
                'Rem':'rem',
                'A':'missing',
                'Artifact':'missing',
                'missing':'missing',
            })

    return data

def read_labels(label_file_path:str, study_start:pd.Timestamp) -> pd.DataFrame:
    """Read event labels from a clinical annotation file and return a 
    pandas.DataFrame with the event labels, start times, seconds since 
    study start, and duration of each event in seconds. The format of 
    the file is detected from the header of the file and the file is
    read accordingly. 

    Example of an annotation file with 'discrete' format:
    -------------------------------------
    Signal ID: SchlafProfil\profil
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

if __name__ == '__main__':
    plm_path = '/oak/stanford/groups/mignot/psg/Amazfit_Health/all/HUA01/HUA 01_analysis/PLM Events.txt'
    hypnogram_path = '/oak/stanford/groups/mignot/psg/Amazfit_Health/all/HUA01/HUA 01_analysis/Sleep profile.txt'
    flow_path = '/oak/stanford/groups/mignot/psg/Amazfit_Health/all/HUA01/HUA 01_analysis/Flow Events.txt'
    edf_path = '/oak/stanford/groups/mignot/psg/Amazfit_Health/all/HUA01/HUA_01.edf'
    acc_path =  '/oak/stanford/groups/mignot/psg/Amazfit_Health/wearable/HUA01.h5'

    # Read labels from file
    edf_file = EdfReader(edf_path)
    study_start = pd.Timestamp(edf_file.getStartdatetime())
    plm_events = read_labels(plm_path, study_start)
    print(plm_events)
    hypnogram = read_labels(hypnogram_path, study_start)
    print(hypnogram)
    flow_events = read_labels(flow_path, study_start)
    print(flow_events)