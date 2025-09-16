import os
import numpy as np
import torch

if __name__ == '__main__':
    # Get path to this file's directory
    _module_dir = os.path.dirname(__file__)

    # Load the .npy file
    sequence_weights = np.load(os.path.join(_module_dir, "weights/norm_auc_weights.npy"))
    sequence_weights = torch.from_numpy(sequence_weights)
    print(sequence_weights.shape)
