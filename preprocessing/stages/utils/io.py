import h5py
import glob
import numpy as np
import pandas as pd

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
        wear_segments: list[pd.DataFrame],
        info: dict,
        annotation_fields: list[str],
        fs: int,
        chunk_size_sec: int = 600,
    ) -> None:
    '''Write the data to an h5 file'''
    with h5py.File(outfile, 'w') as f:
        f.create_group('annotations')
        f.create_group('data')

        # Write the file metadata
        for key,value in info.items():
            f.attrs.create(key,value)
        f.attrs.create('num_segments', len(wear_segments))
        
        for i, segment in enumerate(wear_segments):
            # Write the calibrated accelerometry data
            acc = segment[['x','y','z']].values
            dataset = f['data'].create_dataset(
                f'acc_segment_{i}',
                data=acc.T.astype(np.float32),
                chunks=(acc.shape[1], chunk_size_sec*fs),
            )
            # Write the segment metadata    
            segment_info = {
                'fs': fs,
                'start_time': segment.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': segment.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            }
            for key, value in segment_info.items():
                dataset.attrs.create(key, value)
            
            ppg = segment['ppg'].values
            dataset_ppg = f['data'].create_dataset(
                f'ppg_segment_{i}',
                data=ppg.astype(np.float32),
                chunks=(chunk_size_sec*fs,),
            )
            for key, value in segment_info.items():
                dataset_ppg.attrs.create(key, value)
            
            # Write the annotations
            for field in annotation_fields:
                annotation = segment[field]
                dataset = f['annotations'].create_dataset(
                    f'{field}_segment_{i}',
                    data=annotation.to_numpy(),
                    dtype=h5py.string_dtype(encoding='utf-8'),
                    chunks=(chunk_size_sec*fs,),
                )