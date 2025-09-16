# utils/preprocess_actigraphy.py file

import actipy
import numpy as np
import pandas as pd
import scipy.signal as signal
import statsmodels.api as sm
import warnings

def preprocess_actigraphy_df(
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    fs_in: int = 100,
    fs_out: int = 30,
    gravity_calibration: bool = False,
    calib_cube: float = -1.0,
    max_iters_calib: int = 1500,
    output_suffix: str = '_preprocessed',
    input_unit_divisor: float = 1,
    scale_by_force: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Preprocesses 3-axis accelerometer data using actipy.
    This version is ROBUST: It handles both timezone-aware and timezone-naive
    input DataFrames by temporarily making the index naive for actipy processing.
    """

    required_cols = [x_col, y_col, z_col]
    if not all(col in data_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data_df.columns]
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    all_info = {
        'input_fs': fs_in, 'output_fs': fs_out,
        'input_columns': {'x': x_col, 'y': y_col, 'z': z_col},
        'output_suffix': output_suffix, 'input_unit_divisor_used': input_unit_divisor,
        'calibration_requested': gravity_calibration,
    }

    process_df = data_df[[x_col, y_col, z_col]].copy()
    process_df.rename(columns={x_col: 'x', y_col: 'y', z_col: 'z'}, inplace=True)
    
    if not isinstance(process_df.index, pd.DatetimeIndex):
        raise ValueError(f'Dataframe index should be pd.DatetimeIndex got {type(process_df.index)}')

    # --- ROBUSTNESS CHECK: This block makes the function handle both cases ---
    # 1. Check if the input data has a timezone.
    original_tz = process_df.index.tz
    
    # 2. If it has a timezone, remove it TEMPORARILY for actipy.
    if original_tz is not None:
        if verbose:
            print(f"  Temporarily converting index from {original_tz} to timezone-naive for actipy processing.")
        process_df.index = process_df.index.tz_localize(None)
    # If original_tz is None, this block is skipped, and the naive index is used directly.
    # --- END OF ROBUSTNESS CHECK ---
    
    process_df.index.name = 'time'

    # 0. Apply Input Unit Scaling (Convert to 'g')
    if input_unit_divisor != 1.0:
        if input_unit_divisor == 0:
            raise ValueError("input_unit_divisor cannot be zero.")
        if verbose: print(f"  Applying input unit scaling: Dividing x, y, z by {input_unit_divisor}")
        process_df[['x', 'y', 'z']] = process_df[['x', 'y', 'z']].astype(float) / input_unit_divisor
    elif verbose:
            print("  Skipping input unit scaling (divisor is 1.0).")

    # 1. Handle potential signal dropout
    force = np.linalg.norm(process_df[['x', 'y', 'z']].values, axis=1)
    dropout_indices = np.where(force == 0)[0]
    if len(dropout_indices) > 0:
        # This is a significant event, consider warning always or if verbose
        message = f"  Detected {len(dropout_indices)} potential dropout points (zero magnitude) out of {len(force)} total samples. Setting to NaN."
        if verbose: print(message)
        else: warnings.warn(message, UserWarning)
        process_df.iloc[dropout_indices] = np.nan
        force[dropout_indices] = np.nan
        all_info['dropout_points_handled'] = len(dropout_indices)
    else:
        all_info['dropout_points_handled'] = 0

    if np.all(np.isnan(force)):
        warnings.warn("All force vector values are NaN after dropout handling. Scaling cannot be applied.", UserWarning)
        all_info['mean_force_scaling_skipped'] = 'All data NaN after dropout'

    # 2. Scale signal by the mean force (OPTIONAL)
    if scale_by_force:
        mean_force = np.nanmean(force)
        if np.isnan(mean_force) or mean_force == 0:
            message = f"  Skipping scaling by mean force (mean_force is {mean_force})."
            if verbose: print(message)
            else: warnings.warn(message, UserWarning)
            all_info['mean_force_scaling_skipped'] = f'Mean force was {mean_force}'
        else:
            if verbose: print(f"  Scaling data by mean force: {mean_force:.4f}")
            process_df[['x', 'y', 'z']] = process_df[['x', 'y', 'z']] / mean_force
            all_info['mean_force_scale_factor'] = float(mean_force)
        if len(dropout_indices) > 0:
            process_df.iloc[dropout_indices] = np.nan
    elif verbose:
        print("  Skipping scaling by mean force (scale_by_force=False).")
    all_info['mean_force_scale_factor'] = all_info.get('mean_force_scale_factor', None)

    # 3. Low-pass filter
    cutoff_freq = fs_out / 2.0
    if verbose:
        print(f"  Applying low-pass filter with cutoff ~{cutoff_freq:.2f} Hz...")

    process_df, filter_info = actipy.processing.lowpass(
        data=process_df,
        data_sample_rate=fs_in,
        cutoff_rate=cutoff_freq,
    )
    all_info.update(filter_info); all_info['filtering_applied'] = True

    # 4. Resample
    if verbose: 
        print(f"  Resampling data to {fs_out} Hz...")

    process_df, resample_info = actipy.processing.resample(
        data=process_df,
        sample_rate=fs_out,
    )
        
    all_info.update(resample_info); all_info['resampling_applied'] = True

    # 5. Calibrate Gravity (Conditional)
    calib_diagnostics = {'calibration_skipped_reason': 'Not Skipped'}
    calibration_successful = False; calibration_applied = False
    if gravity_calibration:
        if process_df.isnull().all().all():
            warnings.warn("Data is all NaN before calibration. Skipping calibration step.", UserWarning)
            calib_diagnostics['calibration_skipped_reason'] = 'All data NaN before calibration'
        else:
            if verbose: print("  Calibrating gravity...")
            calib_min_samples = 50
            calib_window = '10s'
            calib_stdtol = 0.013
            calib_chunksize = int(1e6)

            # #########!
            # process_df = process_df.loc[~process_df.isna().any(axis=1)]
            # print(process_df.isna().any(axis=1).sum())

            process_df_calibrated, calib_diagnostics_update = calibrate_gravity(
                data=process_df.copy(), 
                calib_cube=calib_cube, 
                calib_min_samples=calib_min_samples,
                window=calib_window, 
                stdtol=calib_stdtol, 
                chunksize=calib_chunksize,
                max_iters=max_iters_calib,
            )
            calib_diagnostics.update(calib_diagnostics_update)
            
            if calib_diagnostics_update.get('CalibOK', 0) == 1:
                calibration_successful = True
                process_df = process_df_calibrated
                calibration_applied = calib_diagnostics_update.get('CalibNumIters', 0) > 0
                if verbose: print(f'  Calibration successful and {"applied" if calibration_applied else "not applied (low initial error)"}.')
            else:
                warnings.warn(f"Calibration failed (CalibOK != 1). Proceeding with uncalibrated data. Diagnostics: {calib_diagnostics_update}", UserWarning)
                calibration_successful = False
            if verbose: print(f"  Calibration info: {calib_diagnostics}")
                
    if not gravity_calibration:
        calib_diagnostics['calibration_skipped_reason'] = 'Skipped by user flag (gravity_calibration=False)'

    all_info['calibration_successful'] = calibration_successful
    all_info['calibration_applied'] = calibration_applied
    all_info['calibration_diagnostics'] = calib_diagnostics

    # --- ROBUSTNESS RESTORATION: This block restores the original timezone state ---
    # 3. If there WAS an original timezone, add it back to the processed data.
    if original_tz is not None:
        if verbose:
            print(f"  Re-applying original timezone ({original_tz}) to index after actipy processing.")
        process_df.index = process_df.index.tz_localize(original_tz)
    # If the original data was naive, this is skipped, and the output remains naive.
    # --- END OF ROBUSTNESS RESTORATION ---

    output_df = process_df.rename(columns={
        'x': f"{x_col}{output_suffix}",
        'y': f"{y_col}{output_suffix}",
        'z': f"{z_col}{output_suffix}"
    })
    all_info['output_columns'] = list(output_df.columns)
    all_info['output_shape'] = list(output_df.shape)

    return output_df, all_info

def calibrate_gravity(
    data,
    calib_cube = 0.3,
    calib_min_samples = 50,
    window = '10s',
    stdtol = 0.015,
    return_coeffs = True,
    chunksize = 1_000_000,
    max_iters = 1500,
    improv_tol = 0.0001,
    error_tol = 0.01,
    in_mem = True,
):  # noqa: C901
    """
    MODIFIED VERSION:
    Modified version of the same function from the actipy package:
    https://github.com/OxWearables/actipy/blob/master/src/actipy/processing.py
    This version allows max_iters, improv_tol, and error_tol to be set as 
    arguments rather than being constants. Additionally, it does not skip calibration
    if the initial error is below the tolerance. Finally, the orignal version would
    say calibration failed if MAX_ITERS was reached, despite the error being 
    below error_tol, this version fixed that.

    Gravity calibration method of van Hees et al. 2014 (https://pubmed.ncbi.nlm.nih.gov/25103964/)

    :param data: A pandas.DataFrame of acceleration time-series. It must contain
        at least columns `x,y,z` and the index must be a DateTimeIndex.
    :type data: pandas.DataFrame.
    :param calib_cube: Calibration cube criteria. See van Hees et al. 2014 for details. Defaults to 0.3.
    :type calib_cube: float, optional.
    :param calib_min_samples: Minimum number of stationary samples required to run calibration. Defaults to 50.
    :type calib_min_samples: int, optional.
    :param window: Rolling window to use to check for stationary periods. Defaults to 10 seconds ("10s").
    :type window: str, optional
    :param stdtol: Standard deviation under which the window is considered stationary. Defaults to 15 milligravity (0.015).
    :type stdtol: float, optional
    :param chunksize: Chunk size for chunked processing. Defaults to 1_000_000 rows.
    :type chunksize: int, optional
    :return: Processed data and processing info.
    :rtype: (pandas.DataFrame, dict)
    """
    info = {}

    stationary_indicator = (  # this is more memory friendly than of data[['x', 'y', 'z']].std()
        data['x'].resample(window, origin='start').std().lt(stdtol)
        & data['y'].resample(window, origin='start').std().lt(stdtol)
        & data['z'].resample(window, origin='start').std().lt(stdtol)
    )

    xyz = (
        data[['x', 'y', 'z']]
        .resample(window, origin='start').mean()
        [stationary_indicator]
        .dropna()
        .to_numpy()
    )
    # Remove any nonzero vectors as they cause nan issues
    nonzero = np.linalg.norm(xyz, axis=1) > 1e-8
    xyz = xyz[nonzero]

    hasT = 'temperature' in data
    if hasT:
        T = (
            data['temperature']
            .resample(window, origin='start').mean()
            [stationary_indicator]
            .dropna()
            .to_numpy()
        )
        T = T[nonzero]

    del stationary_indicator
    del nonzero

    info['CalibNumSamples'] = len(xyz)

    if len(xyz) < calib_min_samples:
        info['CalibErrorBefore(mg)'] = np.nan
        info['CalibErrorAfter(mg)'] = np.nan
        info['CalibOK'] = 0
        warnings.warn(f"Skipping calibration: Insufficient stationary samples: {len(xyz)} < {calib_min_samples}")
        return data, info

    intercept = np.array([0.0, 0.0, 0.0], dtype=xyz.dtype)
    slope = np.array([1.0, 1.0, 1.0], dtype=xyz.dtype)
    best_intercept = np.copy(intercept)
    best_slope = np.copy(slope)

    if hasT:
        slopeT = np.array([0.0, 0.0, 0.0], dtype=T.dtype)
        best_slopeT = np.copy(slopeT)

    curr = xyz
    target = curr / np.linalg.norm(curr, axis=1, keepdims=True)

    errors = np.linalg.norm(curr - target, axis=1)
    err = np.mean(errors)  # MAE more robust than RMSE. This is different from the paper
    init_err = err
    best_err = 1e16

    info['CalibErrorBefore(mg)'] = init_err * 1000

    # Check that we have sufficiently uniformly distributed points:
    # need at least one point outside each face of the cube
    if (np.max(xyz, axis=0) < calib_cube).any() or (np.min(xyz, axis=0) > -calib_cube).any():
        info['CalibErrorAfter(mg)'] = init_err * 1000
        info['CalibNumIters'] = 0
        info['CalibOK'] = 0

        return data, info

    for it in range(max_iters):

        # Weighting. Outliers are zeroed out
        # This is different from the paper
        maxerr = np.quantile(errors, .995)
        weights = np.maximum(1 - errors / maxerr, 0)

        # Optimize params for each axis
        for k in range(3):

            inp = curr[:, k]
            out = target[:, k]
            if hasT:
                inp = np.column_stack((inp, T))
            inp = sm.add_constant(inp, prepend=True, has_constant='add')
            params = sm.WLS(out, inp, weights=weights).fit().params
            # In the following,
            # intercept == params[0]
            # slope == params[1]
            # slopeT == params[2]  (if exists)
            intercept[k] = params[0] + (intercept[k] * params[1])
            slope[k] = params[1] * slope[k]
            if hasT:
                slopeT[k] = params[2] + (slopeT[k] * params[1])

        # Update current solution and target
        curr = intercept + (xyz * slope)
        if hasT:
            curr = curr + (T[:, None] * slopeT)
        target = curr / np.linalg.norm(curr, axis=1, keepdims=True)

        # Update errors
        errors = np.linalg.norm(curr - target, axis=1)
        err = np.mean(errors)
        err_improv = (best_err - err) / best_err

        if err < best_err:
            best_intercept = np.copy(intercept)
            best_slope = np.copy(slope)
            if hasT:
                best_slopeT = np.copy(slopeT)
            best_err = err
        if err_improv < improv_tol:
            break

    info['CalibErrorAfter(mg)'] = best_err * 1000
    info['CalibNumIters'] = it + 1

    if (best_err > error_tol) and (it + 1 >= max_iters):
        info['CalibOK'] = 0
        return data, info

    elif np.any(np.abs(best_intercept) > 0.3) and np.any(best_slope < 0.7):
        print('Likely degenerate calibration solution detected.')
        print('Intercepts:', best_intercept)
        print('Slopes:', best_slope)
        info['CalibOK'] = 0
        return data, info

    else:
        if in_mem:
            data = data.copy()
            data[['x', 'y', 'z']] = (best_intercept
                                    + best_slope * data[['x', 'y', 'z']].to_numpy())
            if hasT:
                data[['x', 'y', 'z']] = (data[['x', 'y', 'z']]
                                        + best_slopeT * (data['temperature'].to_numpy()[:, None]))
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                # We use TemporaryDirectory() + filename instead of NamedTemporaryFile()
                # because we don't want to open the file just yet:
                # https://stackoverflow.com/questions/26541416/generate-temporary-file-names-without-creating-actual-file-in-python
                # and Windows doesn't allow opening a file twice:
                # https://docs.python.org/3.9/library/tempfile.html#tempfile.NamedTemporaryFile

                n = len(data)
                mmap_fname = os.path.join(tmpdir, 'data.mmap')
                data_mmap = mmap_like(data, mmap_fname, shape=(n,))

                for i in range(0, n, chunksize):

                    # If last chunk, adjust chunksize
                    if i + chunksize > n:
                        chunksize = n - i

                    chunk = data.iloc[i:i + chunksize]
                    chunk_xyz = chunk[['x', 'y', 'z']].to_numpy()
                    chunk_xyz = best_intercept + best_slope * chunk_xyz
                    if hasT:
                        chunk_T = chunk['temperature'].to_numpy()
                        chunk_xyz = chunk_xyz + best_slopeT * chunk_T[:, None]
                    chunk = chunk.copy(deep=True)  # copy to avoid modifying original data
                    chunk[['x', 'y', 'z']] = chunk_xyz

                    copy2mmap(chunk, data_mmap[i:i + chunksize])

                del data

                # We need to copy so that the mmap file can be trully deleted: 
                # https://stackoverflow.com/questions/24178460/in-python-is-it-possible-to-overload-numpys-memmap-to-delete-itself-when-the-m
                data = mmap2df(data_mmap, copy=True)

                del data_mmap

        info['CalibOK'] = 1

        if return_coeffs:
            info['CalibxIntercept'] = best_intercept[0]
            info['CalibyIntercept'] = best_intercept[1]
            info['CalibzIntercept'] = best_intercept[2]
            info['CalibxSlope'] = best_slope[0]
            info['CalibySlope'] = best_slope[1]
            info['CalibzSlope'] = best_slope[2]
            if hasT:
                info['CalibxSlopeT'] = best_slopeT[0]
                info['CalibySlopeT'] = best_slopeT[1]
                info['CalibzSlopeT'] = best_slopeT[2]

    return data, info
