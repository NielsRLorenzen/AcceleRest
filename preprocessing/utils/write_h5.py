import h5py
import pandas as pd
import numpy as np

def write_h5_acc_psg(
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

def write_h5_acc(
    outfile: str,
    accelerometry: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    acc_info: dict,
    annotations: pd.DataFrame,
    study_start: pd.Timestamp,
    chunk_size_sec: int = 600,
    ahi: float = None,
    stage_hours: dict = None,
) -> None:
    '''Write the data to a h5 file'''
    with h5py.File(outfile, 'w') as f:
        f.create_group('annotations')
        f.create_group('data')
        try:
            f.attrs.create('start_time', study_start.strftime('%Y-%m-%d %H:%M:%S'))
        except AttributeError:
            f.attrs.create('start_time', study_start)
        if ahi is not None:
            f.attrs.create('ahi', ahi)
        if stage_hours is not None:
            for stage, hours in stage_hours.items():
                f.attrs.create(f'{stage}_hours', hours)
        
        # Write the calibrated accelerometry data
        try:
            acc_fs = round(1/accelerometry.index.diff().mean().total_seconds(), 2)
        except:
            acc_fs = round(1/np.nanmean(accelerometry.index.diff()), 2)
        acc = accelerometry[[x_col, y_col, z_col]].values
        dataset = f['data'].create_dataset(
            f'accelerometry',
            data=acc.T.astype(np.float32),
            chunks=(acc.shape[1], chunk_size_sec * acc_fs),
        )
        for key, value in acc_info.items():
            dataset.attrs.create(key, str(value))
        dataset.attrs.create('sample_frequency', acc_fs)
            
        # Write the annotations
        try:
            annot_fs = round(1/annotations.index.diff().mean().total_seconds(), 2)
        except:
            annot_fs = round(1/np.nanmean(annotations.index.diff()), 2) 
        for field in annotations.columns:
            annotation = annotations[field]
            dataset = f['annotations'].create_dataset(
                f'{field}',
                data=annotation.to_numpy(),
                dtype=h5py.string_dtype(encoding='utf-8'),
                chunks=(chunk_size_sec * annot_fs,),
            )
            dataset.attrs.create('sample_frequency', annot_fs)