import torch
import h5py
import numpy as np
from warnings import warn
from torch.utils.data import Dataset


class ContigDataset(Dataset):
    '''Dataset that loads contiguous possibly overlapping windows of 
    data from indexed files.
    Indeces are file paths. The data is loaded from the 
    'data/acc_segment_{segment_number}' dataset in the file.

    Args:
        files (list): 
            List of file paths to load data from.
        
        windows_per_file (int): 
            Number of windows to load from each file.
        
        window_size_seconds (int): 
            Size of the window in seconds.
        
        fit_overlap (bool):
            If True, calculate the overlap such that windows_per_file
            fit within min_hours_per_file.

        overlap (float):
            Fraction of overlap between windows. Ignored if fit_overlap
            is True.
        
        min_hours_per_file (float): 
            Minimum number of hours contiguous valid data in each file.
        
        labels (str): 
            Name of the labels to load from the files.
        
        seconds_per_label (int): 
            Number of seconds that one label should cover
    '''
    def __init__(
            self, 
            files: list,
            windows_per_file: int,
            window_size_seconds: int = 3000,
            fit_overlap: bool = False,
            overlap: float = 0.0,
            min_hours_per_file: float = None,
            labels: str = None,
            seconds_per_label: int = 10,
        ):
        super().__init__()
        self.files = files
        self.labels = labels
        self.seconds_per_label = seconds_per_label
        if labels is not None:
            if window_size_seconds % seconds_per_label != 0:
                warn(
                    'Window size should be a multiple of seconds_per_label '
                    'for downsampling labels.'
                )
        self.windows_per_file = windows_per_file
        self.window_size_seconds = window_size_seconds
        self.fit_overlap = fit_overlap
        self.overlap = overlap

        # Allow possibility of automatic overlap calculation to fit 
        # windows_per_file within min_hours_per_file
        self.min_hours_per_file = min_hours_per_file
        if min_hours_per_file is not None:
            min_seconds_per_file = min_hours_per_file * 3600
            needed_seconds = windows_per_file * window_size_seconds
            requires_overlap = (needed_seconds > min_seconds_per_file)
            if fit_overlap and requires_overlap:
                # Calculate the overlap such that windows_per_file 
                # fit within min_hours_per_file
                self.overlap = 1 - min_seconds_per_file / float(needed_seconds + 1)
                print(
                    f'Overlap set to {self.overlap} to fit', 
                    f'{windows_per_file} windows in {min_hours_per_file} hours.'
                )

    def __len__(self):
        return len(self.files)

    def load_data(self, file:str):
        '''Load a single segment of file and return it as a tensor of
        shape (n_windows, n_channels, window_size)
        '''
        with h5py.File(file, 'r', rdcc_nbytes=1024**3) as f:
            num_segments = f.attrs['num_segments']
            segment = np.random.randint(0, num_segments)
            data = f[f'data/acc_segment_{segment}'] # shape (n_channels, nsamples)
            nsamples = data.shape[1]
            # Get the sampling frequency
            fs = data.attrs['fs']
            # Calculate the number of samples to load
            load_size = int(
                (self.windows_per_file * (1-self.overlap))
                * self.window_size_seconds * fs
            )
            # This might be assured by the file creation process if 
            # the file is created with a min_hours_per_file parameter
            assert nsamples >= load_size, f'File {file} is too short to load {load_size} samples.'
            
            # Determine the window size and step size for the sliding window
            window_size = self.window_size_seconds * fs
            step_size = window_size - int(window_size * self.overlap)
            
            # Select segments until a segment without NaN values is found
            while True: 
                # Choose random segment of data of load_size
                start = np.random.randint(0, nsamples - load_size)
                data_tensor = torch.from_numpy(
                    np.array(data[:, start:start+load_size])
                )
                # check for NaN values in the data
                if not torch.isnan(data_tensor).any():
                    break
                
            # Unfold the data into windows
            data_tensor = data_tensor.unfold(-1, window_size, step_size)
            # Move new window dimension to the front as batch dimension
            data_tensor = data_tensor.permute(1,0,2)

            if self.labels is not None: 
                # Load the labels
                labels = f[f'annotations/{self.labels}']
                labels_tensor = torch.from_numpy(
                    np.array(labels[start:start+load_size])
                )
                samples_per_label = int(self.seconds_per_label * fs)
                # Downsample labels by factor samples_per_label
                labels_tensor = labels_tensor.unfold(
                    -1,
                    samples_per_label,
                    samples_per_label,
                )
                labels_tensor = torch.mode(labels_tensor, dim=-1)[0]
                labels_tensor = labels_tensor.unfold(
                    -1,
                    window_size//samples_per_label,
                    step_size//samples_per_label,
                )
                return data_tensor, labels_tensor
            
            else:
                return data_tensor

    def __getitem__(self, idx):
        return self.load_data(self.files[idx])