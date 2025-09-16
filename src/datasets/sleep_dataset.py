import torch
import h5py
import numpy as np
from collections import defaultdict
from warnings import warn
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from time import time
from warnings import warn

class ContigDatasetNight(Dataset):
    '''Dataset that loads contiguous possibly overlapping windows of 
    data from indexed files.
    Indeces are file paths. The data is loaded from the 
    'data/acc_segment_{segment_number}' dataset in the file.

    Args:
        files (list): 
            List of file paths to load data from.
        
        windows_per_night (int): 
            Number of windows to load from each file.
        
        window_size_seconds (int): 
            Size of the window in seconds.
        
        hours_per_night (float): 
            Number of hours is each night segment. Default is 12.
        
        fit_overlap (bool):
            If True, calculate the overlap such that windows_per_night
            fit within hours_per_night.

        overlap (float):
            Fraction of overlap between windows. Ignored if fit_overlap
            is True./oak/stanford/groups/mignot/3projects/actigraphy_fm/code/SRL_WASSED/src/models
        
        labels (str): 
            Name of the labels to load from the files.
        
        seconds_per_label (int): 
            Number of seconds that one label should cover.
    '''
    def __init__(
            self, 
            files: list,
            windows_per_night: int,
            window_size_seconds: int,
            # num_nights_per_file: int = 1,
            hours_per_night: float = 12,
            fit_overlap: bool = False,
            overlap: float = 0.0,
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
        self.windows_per_night = windows_per_night
        self.window_size_seconds = window_size_seconds
        self.fit_overlap = fit_overlap
        self.overlap = overlap
        self.hours_per_night = hours_per_night

        # Allow possibility of automatic overlap calculation to fit 
        # windows_per_night within hours_per_night
        seconds_per_night = hours_per_night * 3600
        needed_seconds = windows_per_night * window_size_seconds
        requires_overlap = (needed_seconds > seconds_per_night)
        if fit_overlap and requires_overlap:
            # Calculate the overlap such that windows_per_night fit within hours_per_night
            self.overlap = 1 - seconds_per_night / float(needed_seconds + 1)
            print(
                f'Overlap set to {self.overlap} to fit', 
                f'{windows_per_night} windows in {hours_per_night} hours.'
            )
        # Initalize a dict to keep track of which nights have been visited
        self.night_idx_tracker = {}

    def __len__(self):
        return len(self.files)

    def load_data(self, file:str):
        '''Load a single segment of file and return it as a tensor of
        shape (n_windows, n_channels, window_size)
        '''
        with h5py.File(file, 'r', rdcc_nbytes=1024**3) as f:
            # Read the number of night segments available in the file
            num_nights = f.attrs['num_nights']

            # Keep track of which night to use next time the file is loaded    
            night_idx = self.night_idx_tracker.get(file, 0)
            if night_idx > (num_nights - 1):
                night_idx = 0
            self.night_idx_tracker[file] = night_idx + 1

            # Load data
            data = f[f'data/night_{night_idx}'] # shape (n_channels, nsamples)
            
            # Get the sampling frequency
            fs = data.attrs['fs']

            # Determine the window size and step size for the sliding window
            window_size = self.window_size_seconds * fs
            step_size = window_size - int(window_size * self.overlap)
            
            # Unfold the data into windows
            data_tensor = torch.Tensor(np.array(data))
            data_tensor = data_tensor.unfold(-1, window_size, step_size)

            # Move new window dimension to the front as batch dimension
            data_tensor = data_tensor.permute(1,0,2)

            return data_tensor

    def __getitem__(self, idx):
        return self.load_data(self.files[idx])

# class WindowDataset(Dataset):
#     '''Dataset with indeces that are windows of data from a list of 
#     files. The windows are created by sliding a window of size 
#     window_size_seconds with a specified overlap. 
#     All data is loaded into memory.

#     Args:
#         files (list): 
#             List of file paths to the data files.

#         window_size_seconds (int):
#             Size of the window in seconds.

#         overlap (float):
#             Fraction of overlap between windows. Default is 0.0.

#         labels (str):
#             Name of the label to use for the dataset. 
#             If None no labels are loaded. Default is None.

#         label_map (dict):
#             Dictionary mapping labels to integers. Only used if labels 
#             is not None. Default is None.

#         label_priority (bool):
#             If True, integer values in label_map are assumed to represent
#             the priority of that label. Labels with higher values will
#             be prioritized when downsampling. If False (default) the
#             mode of each label window is used when downsampling.

#         seconds_per_label (int):
#             Number of seconds per label. Only used if labels is not None.
#             Default is 10.

#         ignore_index (int):
#             Index to use for ignored labels. Only used if labels is not 
#             None. Default is -9.

#         num_workers (int):
#                 Number of workers to use for loading data. Default is 1. 
#     '''
#     def __init__(
#         self,
#         files: list,
#         window_size_seconds: int,
#         overlap: float = 0.0,
#         labels: str = None,
#         label_map: dict = None,
#         label_priority: bool = False,
#         seconds_per_label: int = 10,
#         ignore_index: int = -9,
#         class_weights: list = None,
#         downsample_negative: tuple = None,
#         num_workers = 1,
#     ):
#         super().__init__()
#         self.files = files
#         self.window_size_seconds = window_size_seconds
#         self.overlap = overlap
#         self.labels = labels
#         self.label_map = label_map
#         self.label_priority = label_priority
#         self.seconds_per_label = seconds_per_label
#         self.ignore_index = ignore_index
#         self.class_weights = class_weights
#         self.downsample_negative = downsample_negative
#         self.num_workers = num_workers

#         if labels is not None and window_size_seconds % seconds_per_label != 0:
#             warn('Window size should be a multiple of seconds_per_label for downsampling labels.')

#         self.data_windows = []
#         self.label_windows = []
#         for file in self.files:
#             self._get_windows(file)

#         self.data_windows = torch.cat(self.data_windows, dim=0)
#         if labels is not None:
#             self._drop_windows(downsample_negative)

#     def _drop_windows(self, downsample_negative):
#         self.label_windows = torch.cat(self.label_windows, dim=0)  
#         # Remove any windows where all labels are the ignore index
#         ignore_windows = (self.label_windows == self.ignore_index).all(dim=1)
#         print(ignore_windows.sum(), 'of', len(self.data_windows), 'windows dropped due to ignore index')
#         self.data_windows = self.data_windows[~ignore_windows]
#         self.label_windows = self.label_windows[~ignore_windows]
#         print(len(self.data_windows), 'windows remaining after dropping ignore index')
        
#         if downsample_negative is not None:
#             negative, drop_prcnt = downsample_negative
#             # Get indeces of windows with all negative labels
#             all_neg_idx = torch.argwhere((self.label_windows == negative).all(dim=1))
#             num_neg = len(all_neg_idx)
#             # Pick random subset to drop
#             idx_shuffle = all_neg_idx[torch.randperm(num_neg)]
#             idx_drop = idx_shuffle[:int(num_neg * drop_prcnt)]
#             keep_mask = torch.ones(len(self.label_windows), dtype=torch.bool)
#             keep_mask[idx_drop] = False
#             self.label_windows = self.label_windows[keep_mask]
#             self.data_windows = self.data_windows[keep_mask]
#             print(len(self.data_windows), 'windows remaining after downsampling negatives')

#     def _get_windows(self, file: str) -> None:
#         '''Get windows of data from a file.
#         args:
#             file (str):
#                 Path to the file to load data from.

#         returns: None
#         '''
#         with h5py.File(file, 'r', rdcc_nbytes=1024**3) as f:
#             data_tensor_segments = []
#             label_tensor_segments = []
#             # num_segments = f.attrs['num_segments']
#             # for segment in range(num_segments):

#             data = f[f'data/accelerometry']
#             nsamples = data.shape[1]
#             fs = data.attrs['sample_frequency']

#             # Calculate the window length and step size in samples
#             window_size = int(self.window_size_seconds*fs)
#             step_size = int(window_size*max((1-self.overlap),0.01))

#             # Load data into memory
#             data_tensor = torch.from_numpy(np.array(data))

#             # Unfold the data into windows
#             # -> shape (n_channels, n_windows, window_size)
#             data_tensor = data_tensor.unfold(-1, window_size, step_size)

#             # Move new window dimension to the front as batch dimension
#             # -> shape (n_windows, n_channels, window_size)
#             data_tensor = data_tensor.permute(1,0,2)

#             # Remove any windows with NaN values
#             nan_windows = torch.isnan(data_tensor).any(dim=2).any(dim=1)
#             data_tensor = data_tensor[~nan_windows]
#             self.data_windows.append(data_tensor)

#             if self.labels is not None:
#                 # Load the labels
#                 str_labels = np.array(f[f'annotations/{self.labels}'], dtype='S')
#                 str_labels = np.char.decode(str_labels, 'utf-8')

#                 # Convert the labels to integers using the label dictionary
#                 int_labels = np.array(
#                     [
#                         self.label_map[label] 
#                         for label 
#                         in str_labels
#                     ],
#                     dtype=int
#                 )
#                 labels_tensor = torch.from_numpy(int_labels)

#                 # First make labels corresponding to the data windows
#                 # -> shape (n_windows, window_size)
#                 labels_tensor = labels_tensor.unfold(-1,window_size,step_size)
                
#                 # downsample labels by factor seconds_per_label
#                 # -> shape (n_windows, labels_per_window, samples_per_label)
#                 samples_per_label = int(self.seconds_per_label * fs)
#                 labels_tensor = labels_tensor.unfold(
#                     -1,
#                     samples_per_label,
#                     samples_per_label,
#                 )

#                 # -> shape (n_windows, labels_per_window)
#                 if self.label_priority:
#                     # Keep the label with highest priority in each window
#                     labels_tensor = torch.max(labels_tensor, dim=-1)[0]
#                 else:
#                     # Take the mode of the labels in each window
#                     labels_tensor = torch.mode(labels_tensor,dim=-1)[0]
                
#                 # Remove any windows with NaN values
#                 labels_tensor = labels_tensor[~nan_windows]
#                 self.label_windows.append(labels_tensor)

#     def __len__(self):
#         return len(self.data_windows)

#     def __getitem__(self, idx):
#         if self.labels is not None:
#             return self.data_windows[idx], self.label_windows[idx]
#         else:
#             return self.data_windows[idx]

#     def get_class_weights(self):
#         '''Calculate the class weights for the dataset based on the 
#         label distribution.
#         '''
#         if self.labels is None:
#             raise ValueError('No labels are loaded for the dataset.')

#         elif self.class_weights is not None:
#             return torch.Tensor(self.class_weights)

#         else:
#             labels = self.label_windows.flatten()
#             # Remove the ignore index
#             labels = labels[labels != self.ignore_index]
#             # Count the number of samples in each class
#             class_counts = torch.bincount(labels)
#             class_weights = 1/class_counts
#             return class_weights/class_weights.sum()

class WindowDataset(Dataset):
    '''Dataset with indeces that are windows of data from a list of 
    files. The windows are created by sliding a window of size 
    context_window_patches * patch_size_samples by step_patches.
    All data is loaded into memory.

    Args:
        files (list): 
            List of file paths to the data h5 files.

        patch_size_samples (int):
            Number of samples in patch (token).

        context_window_patches (int):
            Number of patches that go into a single data window.

        step_patches (int):
            The number of patches to move the cotext window when
            creating each input window.

        labels (str):
            Name of the label to use for the dataset. 
            If None no labels are loaded. Default is None.

        label_map (dict):
            Dictionary mapping labels to integers. Only used if labels 
            is not None. Default is None.

        label_priority (bool):
            If True, integer values in label_map are assumed to represent
            the priority of that label. Labels with higher values will
            be prioritized when downsampling. If False (default) the
            mode of each label window is used when downsampling.

        patches_per_label (int):
            Number of patches per label. Should be an integer value >= 1
            or a float < 1 where 1/patches_per_label (1/2, 1/3, ...) 
            gives an integer >= 1. In the first case multiple patches
            are assigned one label in the latter case multiple labels
            are assigned to one patch. 
            Only used if labels is not None.
            Default is 1.

        ignore_index (int):
            Index to use for ignored labels. Only used if labels is not 
            None. Default is -9.

        downsample_negative (tuple[int, float]):
            Tuple of (negative_label_idx, drop_percent). If specified,
            drop_percent of windows with only negative_label_idx are
            dropped.
    '''
    def __init__(
        self,
        files: list,
        patch_size_samples: int,
        context_window_patches: int,
        step_patches: int,
        labels: str,
        label_map: dict,
        label_priority: bool = False,
        patches_per_label: int = 1,
        ignore_index: int = -9,
        class_weights: list = None,
        downsample_negative: tuple[int, float] = None,
    ):
        super().__init__()
        self.files = files
        self.patch_samples = patch_size_samples
        self.window_patches = context_window_patches
        self.step_patches = step_patches
        
        self.labels = labels
        self.label_map = label_map
        self.label_priority = label_priority
        self.ignore_index = ignore_index
        self.patches_per_label = patches_per_label

        self.class_weights = class_weights

        # Calculate the window length and step size in samples
        self.window_samples = int(self.window_patches * self.patch_samples)
        self.step_samples = int(self.step_patches * self.patch_samples)
        self.samples_per_label = int(self.patches_per_label * self.patch_samples)

        self.data_windows = []
        self.label_windows = []
        for file in self.files:
            try:
                self._get_windows(file)
            except RuntimeError as e:
                print(e, 'occured for', file)
                continue

        self.data_windows = torch.cat(self.data_windows, dim=0)
        if labels is not None:
            self._drop_windows(downsample_negative)

    def _drop_windows(self, downsample_negative):
        self.label_windows = torch.cat(self.label_windows, dim=0)  
        # Remove any windows where all labels are the ignore index
        ignore_windows = (self.label_windows == self.ignore_index).all(dim=1)
        print(ignore_windows.sum(), 'of', len(self.data_windows), 'windows dropped due to ignore index')
        self.data_windows = self.data_windows[~ignore_windows]
        self.label_windows = self.label_windows[~ignore_windows]
        print(len(self.data_windows), 'windows remaining after dropping ignore index')
        
        if downsample_negative is not None:
            negative, drop_prcnt = downsample_negative
            # Get indeces of windows with all negative labels
            all_neg_idx = torch.argwhere((self.label_windows == negative).all(dim=1))
            num_neg = len(all_neg_idx)
            # Pick random subset to drop
            idx_shuffle = all_neg_idx[torch.randperm(num_neg)]
            idx_drop = idx_shuffle[:int(num_neg * drop_prcnt)]
            keep_mask = torch.ones(len(self.label_windows), dtype=torch.bool)
            keep_mask[idx_drop] = False
            self.label_windows = self.label_windows[keep_mask]
            self.data_windows = self.data_windows[keep_mask]
            print(len(self.data_windows), 'windows remaining after downsampling negatives')

    def _get_windows(self, file: str) -> None:
        '''Get windows of data from a file.
        args:
            file (str):
                Path to the file to load data from.

        returns: None
        '''
        with h5py.File(file, 'r', rdcc_nbytes=1024**3) as f:
            # Read data
            data = f[f'data/accelerometry']
            nsamples = data.shape[1]
            fs = data.attrs['sample_frequency']

            # Load data into memory
            data_tensor = torch.from_numpy(np.array(data))

            # Unfold the data into windows
            # -> shape (n_channels, n_windows, window_samples)
            data_tensor = data_tensor.unfold(-1, self.window_samples, self.step_samples)

            # Move new window dimension to the front as batch dimension
            # -> shape (n_windows, n_channels, window_size)
            data_tensor = data_tensor.permute(1,0,2)

            # Remove any windows with NaN values
            nan_windows = torch.isnan(data_tensor).any(dim=2).any(dim=1)
            data_tensor = data_tensor[~nan_windows]
            self.data_windows.append(data_tensor)

            if self.labels is not None:
                # Load the labels
                str_labels = np.array(f[f'annotations/{self.labels}'], dtype='S')
                str_labels = np.char.decode(str_labels, 'utf-8')

                # Convert the labels to integers using the label dictionary
                int_labels = np.array(
                    [
                        self.label_map[label] 
                        for label 
                        in str_labels
                    ],
                    dtype=int
                )
                labels_tensor = torch.from_numpy(int_labels)

                # First make labels corresponding to the data windows
                # -> shape (n_windows, window_size)
                labels_tensor = labels_tensor.unfold(-1, self.window_samples, self.step_samples)
                
                # -> shape (n_windows, labels_per_window)
                if self.label_priority == 'event_count':
                    # Count number of events as (onsets+offsets)/2
                    labels_tensor = (labels_tensor >= 1).diff(dim=-1).sum(dim=-1,keepdim=True)/2

                else:
                    # downsample labels by factor seconds_per_label
                    # -> shape (n_windows, labels_per_window, samples_per_label)
                    labels_tensor = labels_tensor.unfold(-1, self.samples_per_label, self.samples_per_label)
                    
                    if self.label_priority:
                        # Keep the label with highest priority in each window
                        labels_tensor = torch.max(labels_tensor, dim=-1)[0]
                    else:
                        # Take the mode of the labels in each window
                        labels_tensor = torch.mode(labels_tensor,dim=-1)[0]
                        
                # Remove any windows with NaN values
                labels_tensor = labels_tensor[~nan_windows]
                self.label_windows.append(labels_tensor)

    def __len__(self):
        return len(self.data_windows)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data_windows[idx], self.label_windows[idx]
        else:
            return self.data_windows[idx]

    def get_class_weights(self):
        '''Calculate the class weights for the dataset based on the 
        label distribution.
        '''
        if self.labels is None:
            raise ValueError('No labels are loaded for the dataset.')

        elif self.class_weights is not None:
            return torch.Tensor(self.class_weights)

        else:
            labels = self.label_windows.flatten()
            # Remove the ignore index
            labels = labels[labels != self.ignore_index]
            # Count the number of samples in each class
            class_counts = torch.bincount(labels)
            class_weights = 1/class_counts
            return class_weights/class_weights.sum()

class SubjectEvaluationDataset(Dataset):
    '''Dataset with indeces that are windows of data from a list of 
    files. The windows are created by sliding a window of size 
    window_size_seconds with a specified overlap. 
    All data is loaded into memory.

    Args:
        file (str): 
            Path to the data h5 file.

        patch_size_samples (int):
            Number of samples in patch (token).

        context_window_patches (int):
            Number of patches that go into a single data window.

        step_patches (int):
            The number of patches to move the cotext window when
            creating each input window.

        labels (str):
            Name of the label to use for the dataset. 
            If None no labels are loaded. Default is None.

        label_map (dict):
            Dictionary mapping labels to integers. Only used if labels 
            is not None. Default is None.

        label_priority (bool):
            If True, integer values in label_map are assumed to represent
            the priority of that label. Labels with higher values will
            be prioritized when downsampling. If False (default) the
            mode of each label window is used when downsampling.

        patches_per_label (int):
            Number of patches per label. Should be an integer value >= 1
            or a float < 1 where 1/patches_per_label (1/2, 1/3, ...) 
            gives an integer >= 1. In the first case multiple patches
            are assigned one label in the latter case multiple labels
            are assigned to one patch. 
            Only used if labels is not None.
            Default is 1.

        ignore_index (int):
            Index to use for ignored labels. Only used if labels is not 
            None. Default is -9.
    '''
    def __init__(
            self,
            file: str,
            patch_size_samples: int,
            context_window_patches: int,
            step_patches: int,
            labels: str,
            label_map: dict,
            label_priority: bool = False,
            patches_per_label: int = 1,
            ignore_index: int = -9,
            downsample_negative: tuple = None,
        ):
        super().__init__()
        self.file = file
        self.patch_samples = patch_size_samples
        self.window_patches = context_window_patches
        self.step_patches = step_patches
        
        # Calculate the window length and step size in samples
        self.window_samples = int(self.window_patches * self.patch_samples)
        self.step_samples = int(self.step_patches * self.patch_samples)

        self.labels = labels
        self.label_map = label_map
        self.label_priority = label_priority
        self.ignore_index = ignore_index
        self.patches_per_label = patches_per_label
        
        # Also sets self.data and self.labels_seq with the unwindowed shapes
        self.data_windows, self.label_windows = self.get_windows(file)

    def get_windows(self, file: str):
        '''Get windows of data from a file.

        TODO: Implement a way to take the second most common label 
        if the mode is unlabelled. If the mode was 'NaN' but not all 
        values were missing prioritize the non-missing label. This is 
        mainly important for sub-samples_per_label events that would 
        be lost otherwise.

        args:
            file (str):
                Path to the file to load data from.

        returns:
            if labels is None:
                data_tensor (torch.Tensor):
                    Tensor with the data windows.

            else:    
                tuple(
                    data_tensor (torch.Tensor), 
                    labels_tensor (torch.Tensor)
                ):
                    Tensor with the data windows and the corresponding 
                    labels.

        '''
        with h5py.File(file, 'r', rdcc_nbytes=1024**3) as f:
            if self.labels == 'apnea' and 'ahi' in f.attrs.keys():
                self.ahi = f.attrs['ahi']
            
            # Get data
            data = f[f'data/accelerometry']
            n_samples = data.shape[1]

            # Load data into memory
            data_tensor = torch.from_numpy(np.array(data))

            # Store original data
            self.data = data_tensor
            # n_windows = ((n_samples - self.window_samples) // self.step_samples) + 1
            # last_covered_sample = (n_windows - 1) * self.step_samples + self.window_samples
            # n_dropped_samples = n_samples-last_covered_sample#(((n_samples - self.window_samples) % self.step_samples))

            ## Load the labels
            str_labels = np.array(f[f'annotations/{self.labels}'], dtype='S')
            str_labels = np.char.decode(str_labels, 'utf-8')

            # Convert the labels to integers using the label dictionary
            int_labels = np.array(
                [
                    self.label_map[label] 
                    for label 
                    in str_labels
                ],
                dtype=int
            )
            labels_tensor = torch.from_numpy(int_labels)
            samples_per_label = int(self.patches_per_label * self.patch_samples)
            
            # Get a sequence of labels for each patch for the entire recording
            labels_sequence = labels_tensor.unfold(-1, samples_per_label, samples_per_label)

            if self.window_patches == -1:
                # Return full-length data and labels
                if self.label_priority:
                    # Keep the label with highest priority in each window
                    labels_sequence = torch.max(labels_sequence, dim=-1)[0]
                else:
                    # Take the mode of the labels in each window
                    labels_sequence = torch.mode(labels_sequence, dim=-1)[0]
                
                self.labels_sequence = labels_sequence
                
                # Shorten data tensor to account for samples cut by unfolding
                data_tensor = data_tensor[:, :len(labels_sequence)*samples_per_label]

                return data_tensor.unsqueeze(0), labels_sequence.unsqueeze(0)
            
            # Unfold the data into windows
            # -> shape (n_channels, n_windows, window_size)
            data_tensor = data_tensor.unfold(-1, self.window_samples, self.step_samples)

            # Move new window dimension to the front as batch dimension
            # -> shape (n_windows, n_channels, window_size)
            data_tensor = data_tensor.permute(1,0,2)

            # First make labels corresponding to the data windows
            # -> shape (n_windows, window_size)
            labels_tensor = labels_tensor.unfold(-1, self.window_samples, self.step_samples)
            
            if self.label_priority == 'event_count':
                # Count number of events as (onsets+offsets)/2
                labels_tensor = (labels_tensor >= 1).diff(dim=-1).sum(dim=-1,keepdim=True)/2
            else:
                # downsample labels by factor seconds_per_label
                # -> shape (n_windows, labels_per_window, samples_per_label)
                labels_tensor = labels_tensor.unfold(-1, samples_per_label, samples_per_label)

                # -> shape (n_windows, labels_per_window)
                if self.label_priority:
                    # Keep the label with highest priority in each window
                    labels_tensor = torch.max(labels_tensor, dim=-1)[0]
                    self.labels_sequence = torch.max(labels_sequence, dim=-1)[0]
                else:
                    # Take the mode of the labels in each window
                    labels_tensor = torch.mode(labels_tensor,dim=-1)[0]
                    self.labels_sequence = torch.mode(labels_sequence, dim=-1)[0]

            return data_tensor, labels_tensor 

    def __len__(self):
        return len(self.data_windows)

    def __getitem__(self, idx):
        return self.data_windows[idx], self.label_windows[idx]