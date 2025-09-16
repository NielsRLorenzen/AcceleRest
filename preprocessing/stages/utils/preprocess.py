import actipy
import numpy as np
import pandas as pd

def preprocess_data(
        data: pd.DataFrame,
        fs: int = 32,
        fs_resample: int = 30,
        calib_cube: float = -1.,
    ) -> pd.DataFrame:

    # Zero magnitude force vector means signal dropout
    force = np.linalg.norm(data[['x', 'y', 'z']].values, axis = 1)
    dropout = force == 0
    force[dropout] = np.nan

    # Scale the signal by the mean of the force vector
    # This provides a better starting point for the calibration
    data[['x', 'y', 'z']] = data[['x', 'y', 'z']] / np.nanmean(force)
    # Set the dropout values to NaN
    data.loc[dropout] = np.nan

    # Preprocessing pipeline
    data, filter_info = actipy.processing.lowpass(
        data,
        fs,
        cutoff_rate = fs_resample // 2,
    )
    data, resample_info = actipy.processing.resample(
        data,
        fs_resample,
    )
    # Calibrate gravity
    data, calib_diagnostics = actipy.processing.calibrate_gravity(
        data,
        calib_cube=calib_cube,
        calib_min_samples=50,
        window='10s',
        stdtol=0.013,
        chunksize=int(1e6),
    )

    info = {
        **filter_info,
        **resample_info,
        **calib_diagnostics,
    }
    return data, info