# AcceleRest

## Getting started
Currently, your raw data files (.cwa, .csv, .bin, etc.) must first be preprocessed manually, e.g. resampling to 30Hz, calibrated to gravity, and non-wear periods set to NaN, and then saved as an array of [axes X n_samples] into a .h5 format with the data stored in 'data/accelerometry'.

The following code:

```
with h5py.File(file, 'r', rdcc_nbytes=1024**3) as f:
    data = f['data/accelerometry']
    n_samples = data.shape[1]
    print(n_samples)
```
Should result in the number of samples being printed.

Use this commandline prompt and edit the paths appropriately to your directory structure: 
```
python /path/to/repo/AcceleRest/accelerest_main.py --data_file_dir /path/to/data/dir/ --output_dir /path/to/output/dir/ --lstm_sleepstages --linear_sleepstages --linear_respevents --context_window_shift 1 --max_batch_size 16 --window_wise_predictions 
```
The --data_file_dir should be a path to a folder with appropriately formatted h5 files (see above).
The output_dir will have a subdirectory for each input file with the same name containing the outputs for that file, depending on which flags were used.

These flags determine what prediction heads are used:
```
--lstm_sleepstages # For lightweight LSTM sequence-model sleep stage predictions.
--linear_sleepstages # For Linear patch-wise (30s epochs) sleep stage predictions.
--linear_respevents # For Linear patch-wise respiratory event predictions.
```
You must specify at least one prediction head. If more are selected they are all run on the output of the same AcceleRest encoder backbone to save compute.

The other options control the following behaviors:
```
 --context_window_shift # Number of patches (30s epochs) to shift the model context window between consecutive forward-passes. Prediction are averaged across overlapping context windows.
 --max_batch_size # Number of context windows to include in a single forward-pass. Higher is faster but more memory intensive.
 --window_wise_predictions # Include to return a memmap file with an array of predictions per context window in addition to the cross-window averaged predictions.
 ```
### Outputs
For each specified prediction head the following files are saved:
```
/path/to/output/dir/{prefix}_soft_preds.npy
```
Where the {prefix} is the corresponding prediction head flag. This file contains a [n_patches X n_classes] array of the patch-wise probabilities of each class.

For sleep stages the class indeces correspond to:
deep: 0, light: 1, rem: 2, wake: 3

```
# To get the probabilitis for each stage:
soft_preds = np.load(os.path.join(output_dir/original_filename, "{prefix}_soft_preds.npy"), allow_pickle=True).item()
soft_preds[:, 0] # for patch-wise deep sleep probabilities.
soft_preds[:, 2] # for patch-wise REM sleep probabilities.

# To get the hard predictions for each patch
hard_preds = np.argmax(soft_preds, axis = 1)
hard_preds[0] # The predicted sleep stage for the first patch.

```

For respiratory events the index label map is:
no_event: 0, event: 1

if the window_wise_prediction flag was used:
```
/path/to/output/dir/{prefix}_window_wise_logits.dat
/path/to/output/dir/{prefix}_window_wise_logits_meta.npy
```
The {prefix}_window_wise_logits.dat file contains the [n_windows X n_classes X window_lenght] array of context window-wise soft predictions.
The {prefix}_window_wise_logits_meta.npy contains the shape and data type of the array in the .dat file and is used when loading the array.

```
# How to load memmap output
meta = np.load(os.path.join(output_dir/original_filename, "{prefix}_window_wise_logits_meta.npy"), allow_pickle=True).item()
mm = np.memmap(
  os.path.join(output_dir/original_filename, "{prefix}_window_wise_logits.dat"),
  mode="r", dtype=meta["dtype"], shape=tuple(meta["shape"])
)
```