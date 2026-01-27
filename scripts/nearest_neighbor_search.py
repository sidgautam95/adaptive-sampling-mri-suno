"""
Nearest-neighbor (NN) based scan-adaptive mask prediction.

Given:
  - training aliased images (adjoint recon from low-frequency k-space, A^H y_lf)
  - corresponding optimized scan-adaptive masks (e.g., ICD masks)

This script finds the nearest training aliased image to a test aliased image
(using a simple Frobenius norm distance) and returns the corresponding mask.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append("../utils")
from utils import crop_img  # crop_img should be defined in utils.py


def load_npy_stack_from_dir(path, crop_hw=(640, 368)):
    """
    Load all .npy files from a directory, optionally crop each, and return a stacked array.

    Parameters
    ----------
    path : str
        Directory containing .npy files.
    crop_hw : tuple or None
        If provided, crops each image using crop_img(img, crop_hw[0], crop_hw[1]).

    Returns
    -------
    np.ndarray
        Stacked array of loaded files (N, H, W, ...) depending on saved shapes.
    """
    filenames = sorted([f for f in os.listdir(path) if f.endswith(".npy")])
    data_list = []

    for fname in filenames:
        file_path = os.path.join(path, fname)
        arr = np.load(file_path)

        if crop_hw is not None:
            arr = crop_img(arr, crop_hw[0], crop_hw[1])

        data_list.append(arr)

    return np.stack(data_list, axis=0)


# -------------------------------------------------------------------------
# USER PATHS (PLEASE UPDATE)
# -------------------------------------------------------------------------

# Directory containing training aliased images (A^H y_lf) saved as .npy
train_img_dir = "../../train-img-aliased/"     # <-- CHANGE THIS

# Directory containing corresponding optimized masks saved as .npy
train_mask_dir = "../../train-masks/"          # <-- CHANGE THIS

# Test aliased image (A^H y_lf) saved as .npy
test_img_path = "../../img_aliased_test.npy"   # <-- CHANGE THIS

# Output filenames
out_mask_npy = "suno_mask.npy"
out_mask_png = "suno_mask.png"

# -------------------------------------------------------------------------
# LOAD TRAINING DATA
# -------------------------------------------------------------------------

img_aliased_train = load_npy_stack_from_dir(train_img_dir, crop_hw=(640, 368))   # shape: (N, ...)
train_masks = load_npy_stack_from_dir(train_mask_dir, crop_hw=None)             # masks usually shouldn't be cropped

nTrain = img_aliased_train.shape[0]
if train_masks.shape[0] != nTrain:
    raise ValueError("Mismatch: number of training images and masks must be the same.")

# Load test aliased image
img_aliased_test = np.load(test_img_path)
img_aliased_test = crop_img(img_aliased_test, 640, 368)

# -------------------------------------------------------------------------
# NORMALIZE (SAFE)
# -------------------------------------------------------------------------

eps = 1e-12
img_aliased_train = img_aliased_train / (np.max(np.abs(img_aliased_train), axis=tuple(range(1, img_aliased_train.ndim)), keepdims=True) + eps)
img_aliased_test = img_aliased_test / (np.max(np.abs(img_aliased_test)) + eps)

# Convert to torch for distance computation
train_t = torch.tensor(img_aliased_train)
test_t = torch.tensor(img_aliased_test)

# -------------------------------------------------------------------------
# NEAREST NEIGHBOR SEARCH (FROBENIUS NORM ON MAGNITUDE)
# -------------------------------------------------------------------------

metric_value = np.zeros(nTrain, dtype=np.float64)

# Compute || |x_test| - |x_train_i| ||_F for each training sample
test_mag = torch.abs(test_t)
for i in range(nTrain):
    metric_value[i] = torch.linalg.norm(test_mag - torch.abs(train_t[i]), ord="fro").item()

nearest_neighbor = int(np.argmin(metric_value))
suno_mask = train_masks[nearest_neighbor]

# -------------------------------------------------------------------------
# SAVE + PLOT
# -------------------------------------------------------------------------

np.save(out_mask_npy, suno_mask)

plt.figure()
plt.imshow(suno_mask, cmap="gray")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(out_mask_png, dpi=300, bbox_inches="tight")
plt.close()

print(f"Nearest neighbor index: {nearest_neighbor}")
print(f"Saved SUNO mask to: {out_mask_npy} and {out_mask_png}")
