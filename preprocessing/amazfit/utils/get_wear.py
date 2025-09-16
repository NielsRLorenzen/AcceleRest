import numpy as np
import pandas as pd

def get_wear_segments(
            data: pd.DataFrame,
            min_weartime: str,
        ) -> list[pd.DataFrame]:
        '''Split the data into segments with at least min_weartime 
        of weartime
        '''
        nonwear = data.isna()['x']
        wear_start, wear_end = find_contiguous_weartime(
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
        wear_start, wear_end = get_wear_change_indices(nonwear)
        # Return the start and end times of the wear segments
        # Subtract 1 from wear_end since time index is inclusive
        return time[wear_start], time[wear_end-1]                        
    
def get_wear_change_indices(
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