# -------------------------------------------------------------------------
# Data preprocessing script for MoDL and U-Net training
# -------------------------------------------------------------------------
# This script preprocesses fastMRI multicoil data to generate:
#   - aliased images
#   - undersampling masks
#   - ground-truth images
#   - coil sensitivity maps
#
# NOTE:
# 1) Please download the fastMRI multicoil dataset from:
#       https://fastmri.med.nyu.edu/
# 2) Store each scan as a .npz file containing:
#       - 'kspace' : [nslices, ncoils, H, W]
#       - 'sensitivity_maps_4x' (or similar)
# 3) Create your own scan-level train/validation split (text files)
# 4) Update the paths below accordingly before running this script.
# -------------------------------------------------------------------------

import numpy as np
import os
import torch
import sys

sys.path.append("../utils")
sys.path.append("../models")

from modl_cg_functions import *
from utils import *

# -------------------------------------------------------------------------
# USER-DEFINED PATHS (PLEASE UPDATE)
# -------------------------------------------------------------------------

# Path to directory containing fastMRI multicoil .npz files
kspace_path = "/path/to/fastmri_multicoil_npz/"     # <-- CHANGE THIS

# Directory to save preprocessed training/validation data
save_dir = "modl-training-data/"                   # <-- CHANGE IF NEEDED

# Path to optimized ICD mask (generated using get_icd_mask.py)
icd_mask_path = "/path/to/optimized_icd_mask.npy"  # <-- CHANGE THIS

# Scan-level train/validation split files
training_scans_file   = "/path/to/train_scans.txt" # <-- CHANGE THIS
validation_scans_file = "/path/to/val_scans.txt"   # <-- CHANGE THIS

# -------------------------------------------------------------------------
# LOAD TRAIN / VALIDATION SPLITS
# -------------------------------------------------------------------------

training_scans = open(training_scans_file, "r").read().splitlines()
validation_scans = open(validation_scans_file, "r").read().splitlines()

training_scans = set(training_scans)
total_scans = training_scans.union(validation_scans)

# -------------------------------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# -------------------------------------------------------------------------

os.makedirs(save_dir, exist_ok=True)

subdirs = [
    "train-img-aliased", "train-masks", "train-img-gt", "train-maps",
    "val-img-aliased", "val-masks", "val-img-gt", "val-maps"
]

for sd in subdirs:
    os.makedirs(os.path.join(save_dir, sd), exist_ok=True)

# -------------------------------------------------------------------------
# MAIN PREPROCESSING LOOP
# -------------------------------------------------------------------------

for scan in sorted(total_scans):

    file = np.load(os.path.join(kspace_path, scan + ".npz"))

    volume_kspace = file["kspace"]

    # Handle possible naming differences for sensitivity maps
    if "sensitivity_maps_4x" in file:
        sensitivity_maps = file["sensitivity_maps_4x"]
    else:
        sensitivity_maps = file["sensitivty_maps_4x"]

    nslices, ncoils, height, width = volume_kspace.shape

    # Ignore first and last few slices (as commonly done for fastMRI)
    for slc_idx in range(10, nslices - 5):

        kspace = volume_kspace[slc_idx]
        maps   = sensitivity_maps[slc_idx]

        icd_mask = np.load(icd_mask_path)

        # Fully sampled ground truth image
        img_gt, _, _ = preprocess_data(
            kspace, maps, np.ones((height, width))
        )

        # Aliased image using scan-adaptive ICD mask
        img_aliased, mask, _ = preprocess_data(
            kspace, maps, icd_mask
        )

        # -----------------------------------------------------------------
        # SAVE PREPROCESSED DATA
        # -----------------------------------------------------------------

        if scan in training_scans:
            np.save(f"{save_dir}/train-img-aliased/train_img_aliased_{scan}_slc{slc_idx}",
                    img_aliased.cpu().numpy())
            np.save(f"{save_dir}/train-masks/train_masks_{scan}_slc{slc_idx}",
                    mask.cpu().numpy())
            np.save(f"{save_dir}/train-maps/train_maps_{scan}_slc{slc_idx}",
                    maps.cpu().numpy())
            np.save(f"{save_dir}/train-img-gt/train_img_gt_{scan}_slc{slc_idx}",
                    img_gt.cpu().numpy())
        else:
            np.save(f"{save_dir}/val-img-aliased/val_img_aliased_{scan}_slc{slc_idx}",
                    img_aliased.cpu().numpy())
            np.save(f"{save_dir}/val-masks/val_masks_{scan}_slc{slc_idx}",
                    mask.cpu().numpy())
            np.save(f"{save_dir}/val-maps/val_maps_{scan}_slc{slc_idx}",
                    maps.cpu().numpy())
            np.save(f"{save_dir}/val-img-gt/val_img_gt_{scan}_slc{slc_idx}",
                    img_gt.cpu().numpy())
