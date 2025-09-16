import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import warnings

# Adding this function for easier compatibility with different fs signals
# as these don't necessarily fit in the same dataframe
# We could do more integration and refactoring later

def preprocess_psg_dict(
    signals: dict[np.array],
    signals_metadata: dict[dict],
    fs_out: float = 100.,
    filter_order: int = 4,
    standardization: bool = True,
):
    resampled_signals = {}
    resampled_metadata = {}
    for label, signal in signals.items():
        # Copy old metadata
        resampled_metadata[label] = signals_metadata[label]

        fs_in = signals_metadata[label]['sample_frequency']
        if fs_out == fs_in:
            resampled_signals[label] = signal
            resampled_metadata[label]['resampled'] = False
            resampled_metadata[label]['antialiased'] = False

        else:
            if fs_out <= fs_in:    
                # Filtering with butterworth
                b, a = butter(filter_order, fs_out/fs_in, btype='lowpass')
                filtered_signal = filtfilt(b, a, signal)
                resampled_metadata[label]['antialiased'] = True
            
            elif fs_out >= fs_in:
                filtered_signal = signal
                resampled_metadata[label]['antialiased'] = False
                
            # Resampling by linear interpolation
            duration = len(filtered_signal) / fs_in
            original_time_points = np.linspace(0, duration, num = len(filtered_signal))
            new_time_points = np.linspace(0, duration, num = int(np.round(duration * fs_out)))
            resampled_signal = np.interp(new_time_points, original_time_points, filtered_signal)
            
            # Set new metadata 
            resampled_metadata[label]['sample_frequency'] = fs_out
            resampled_metadata[label]['resampled'] = True

            # Save preprocessed signal
            resampled_signals[label] = resampled_signal
        
        if standardization:
            mean = np.nanmean(resampled_signals[label])
            std = np.max([np.nanstd( resampled_signals[label]), 1e-8])
            resampled_signals[label] = (resampled_signals[label] - mean) / std

            resampled_metadata[label]['standardized'] = True


    return resampled_signals, resampled_metadata