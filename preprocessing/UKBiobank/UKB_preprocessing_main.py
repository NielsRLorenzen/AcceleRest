import os
import glob
import h5py
import actipy
import argparse
import multiprocessing
import numpy as np
import pandas as pd

argparser = argparse.ArgumentParser(
    description='Preprocess UK biobank accelerometry data'
)
argparser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the uk biobank actigraphy data folder')
argparser.add_argument('--start', type=int, default=0,
                        help='Index of the first file to process')
argparser.add_argument('--end', type=int, default=-1,
                        help='Index of the last file to process')
argparser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory')
argparser.add_argument('--exclusion_dir', type=str, required=True,
                        help='Path to the directory for excluded files')
argparser.add_argument('--resample_freq', type=int, default=30,
                        help='Resample frequency in Hz')
argparser.add_argument('--chunk_size_sec', type=int, default=600,
                        help='Chunk size in seconds')
argparser.add_argument('--nonwear_patience', type=int,default=90,
                        help='Nonwear patience in minutes')
argparser.add_argument('--num_workers', type=int, default=-1,
                        help='Number of workers to use for multiprocessing.')

class UKB_preprocessing():
    """A class to preprocess UK Biobank accelerometry data. Calibration,
    resampling, and nonwear detection are performed using the actipy 
    library. Wear segments separated by non-wear are split into 
    segments with at least 24 hours of weartime. The data is written to
    h5 files with metadata.
    """
    def __init__(
            self,
            data_list: str,
            output_dir: str,
            exclusion_dir: str,
            resample_freq: int,
            chunk_size_sec: int,
            nonwear_patience: int,
            num_workers: int = 1,
        ):
        self.data_list = data_list
        self.output_dir = output_dir
        self.exclusion_dir = exclusion_dir
        self.resample_freq = resample_freq
        self.chunk_size = chunk_size_sec * resample_freq
        self.nonwear_patience = nonwear_patience
        self.num_workers = num_workers

    def preprocess_all(self):
        """Preprocess all files in the data list using multiprocessing 
        if available.
        """ 
        if self.num_workers > 1:
            with multiprocessing.Pool(self.num_workers) as pool:
                future_results = pool.imap_unordered(
                    self.pipeline, 
                    self.data_list,
                    chunksize=5,
                )
                print('Starting multiprocessing')
                for _ in future_results:
                    pass
        else:
            for file in self.data_list:
                self.pipeline(file)
    
    def pipeline(self, file:str):
        filename = os.path.basename(file)
        sub_id = filename.split('_')[0]
        run = filename.split('_')[2]
        outfile = f'{sub_id}_{run}.h5'
        outpath = os.path.join(self.output_dir, outfile)
        exclusion_path = os.path.join(self.exclusion_dir, outfile)

        # Check if the output file already exists and skip if it does
        if (os.path.exists(outpath)
            or os.path.exists(exclusion_path)
        ):
            return

        # Perform filtering, resampling, calibration, and nonwear detection
        try:
            data, info = self.preprocess_file(file)
        except EOFError as e:
            print(f'EOFError while processing {file}: {e}')
            return
        except OverflowError as e:
            Oprint(f'OverflowError while processing {file}: {e}')
        except OSError as e:
            print(f'OSError while processing {file}: {e}')
            return
        except Exception as e:
            print(f'Error while processing {file}: {e}')
            return

        # Exclude if calibration failed
        if info['CalibOK'] == 0:
            outpath = exclusion_path

        # Split the data into segments with at least 24 hours of weartime
        wear_segments = self.get_wear_segments(data, '24h')

        # Exclude if no wear segments > 24 hours
        if len(wear_segments) == 0:
            outpath = exclusion_path

        # Write the data to an h5 file
        self.write_h5(outpath, wear_segments, info)

    def preprocess_file(self, file:str):
        '''Preprocess a single file using the actipy library'''
        print(f'Processing {file}')
        # Use actipy to read the cwa data file
        data, meta = actipy.read_device(
            file,
            lowpass_hz=None,
            calibrate_gravity=False,
            detect_nonwear=False,
            resample_hz=None,
            verbose=False,
        )

        # Lowpass filter
        data, filter_info = actipy.processing.lowpass(
            data,
            meta['SampleRate'],
            cutoff_rate = self.resample_freq//2
        )

        # Resample to resample_freq Hz
        data, resample_info = actipy.processing.resample(
            data, 
            self.resample_freq
        )
        
        # Calibrate gravity
        data, calib_diagnostics = actipy.processing.calibrate_gravity(
            data,
            calib_cube=0.3,
            calib_min_samples=50,
            window='10s',
            stdtol=0.013,
            chunksize=int(1e6),
        )

        # Make detected nonwear periods NaN
        data, nonwear_info = actipy.processing.detect_nonwear(
            data, 
            patience = f'{self.nonwear_patience}m', 
            window = '10s',
            stdtol = 0.013,
        )

        info = {
            **filter_info,
            **resample_info,
            **nonwear_info,
            **calib_diagnostics,
        }
        return data, info
        
    def write_h5(
            self,
            outfile: str,
            wear_segments: list[pd.DataFrame],
            info: dict,
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
                    chunks=(acc.shape[1], self.chunk_size),
                )
                # Write the segment metadata    
                segment_info = {
                    'fs': self.resample_freq,
                    'start_time': segment.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': segment.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                }
                for key, value in segment_info.items():
                    dataset.attrs.create(key, value)

    def get_wear_segments(
            self,
            data: pd.DataFrame,
            min_weartime: str,
        ) -> list[pd.DataFrame]:
        '''Split the data into segments with at least min_weartime 
        of weartime
        '''
        nonwear = data.isna()['x']
        wear_start, wear_end = self.find_contiguous_weartime(
            nonwear,
            data.index,
        )
        # Get the lengths of the wear segments
        segment_lengths = wear_end - wear_start
        keep = segment_lengths >= pd.Timedelta(min_weartime)
        
        # Split the data into segments
        start_times = wear_start[keep]
        end_times = wear_end[keep]
        wear_segments = [
            data.loc[start:end]
            for start, end
            in zip(start_times, end_times)
        ]
        return wear_segments

    def find_contiguous_weartime(
            self,
            nonwear: np.ndarray,
            time: pd.DatetimeIndex,
        ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:

        '''Find contiguous weartime segments in the data'''
        if not nonwear.any():
            # Return a single segment that covers the whole file
            return time[[0]], time[[-1]]
        elif nonwear.all():
            # Return a single segment that covers no time
            return time[[0]], time[[0]]
        else:
            # Find the indices where nonwear starts and ends
            wear_start, wear_end = self.get_wear_change_indices(nonwear)
            # Return the start and end times of the wear segments
            # Subtract 1 from wear_end since time index is inclusive
            return time[wear_start], time[wear_end-1]                        
        
    def get_wear_change_indices(
            self,
            nonwear,
        ) -> tuple[np.ndarray, np.ndarray]:
        '''Find the indices where nonwear starts and ends.
        Each pair of the returned arrays corresponds to a contiguous 
        weartime segment.'''
        nonwear_change = np.diff(nonwear.astype(int))
        # Add a 0 at the beginning to account for the first sample
        nonwear_change = np.insert(nonwear_change, 0, 0)
        # Find the indices where nonwear starts and ends
        wear_end = np.argwhere(nonwear_change == 1).flatten()
        wear_start = np.argwhere(nonwear_change == -1).flatten()

        # Check if the file starts or ends with nonwear
        if len(wear_start) == 0: 
            # There is no start of wear so the first segment is wear
            begins_with_nonwear = False
            ends_with_nonwear = True
        elif len(wear_end) == 0: 
            # There is no end of wear so the last segment is wear
            begins_with_nonwear = True
            ends_with_nonwear = False
        else:
            # First start of wear is before the first end of wear
            begins_with_nonwear = wear_start[0] < wear_end[0] 
            # Last start of wear is before the last end of wear
            ends_with_nonwear = wear_start[-1] < wear_end[-1]

        # Make sure that wear_start[i] < wear_end[i] for all i 
        if not ends_with_nonwear: # ends with wear
            wear_end = np.append(wear_end, len(nonwear)-1)
        if not begins_with_nonwear: # begins with wear
            wear_start = np.insert(wear_start, 0, 0)

        return wear_start, wear_end

if __name__ == '__main__':
    args = argparser.parse_args()

    print(f'Number of available CPUs: {multiprocessing.cpu_count()}')
    args.num_workers = (
        multiprocessing.cpu_count() 
        if args.num_workers == -1 
        else args.num_workers
    )
    print(f'Using {args.num_workers} workers')
    
    data_list = glob.glob(args.data_dir + '*.cwa.gz')
    if len(data_list) == 0:
        raise FileNotFoundError(f'No files found in {args.data_dir}')
    data_list.sort()
    data_list = data_list[args.start:args.end]

    preprocess = UKB_preprocessing(
        data_list,
        args.output_dir,
        args.exclusion_dir,
        args.resample_freq, 
        args.chunk_size_sec,
        args.nonwear_patience,
        args.num_workers,
    )

    preprocess.preprocess_all()    
