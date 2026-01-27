"""
ICD sampling optimization for a single multicoil MRI scan/slice.

User specifies:
  - reconstruction method (e.g., U-Net or MoDL variant)
  - metric weights (alpha1..alpha4)
  - undersampling factor (us_factor)
  - number of ICD passes (num_icd_iter)

Expected inputs:
  ksp   : complex array, shape [ncoils, H, W]
  mps   : complex array, shape [ncoils, H, W]  (coil sensitivity maps)
  img_gt: complex array, shape [H, W]          (ground truth image)
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append("../utils")
sys.path.append("../models")
sys.path.append("../sampling-optimization")

from utils import *
from unet_fbr import Unet
from didn import DIDN
from icd_sampling_optimization import icd_sampling_optimization

torch.set_num_threads(4)

# -------------------------------------------------------------------------
# USER SETTINGS (PLEASE UPDATE THESE PATHS AS NEEDED)
# -------------------------------------------------------------------------

# ICD parameters
num_icd_iter = 1        # number of ICD passes over the mask
recon = "unet"          # options: "unet" or "modl" (as implemented below)
us_factor = 4           # undersampling factor

# Metric weights:
# alpha1=1 corresponds to NRMSE (see utils.py for details)
alpha1, alpha2, alpha3, alpha4 = 1, 0, 0, 0

nChannels = 2           # number of channels used by the reconstructor

# Paths to pretrained models (update if you store them elsewhere)
unet_model_path = "../../saved-models/unet_fastmri_vdrs_4x.pt"
modl_model_path = "../../saved-models/modl_didn_fastmri_vdrs_4x.pt"

# Paths to example data (update to your own files)
ksp_path = "../../data/ksp.npy"
mps_path = "../../data/mps.npy"
img_gt_path = "../../data/img_gt.npy"

# -------------------------------------------------------------------------
# DEVICE SETUP
# -------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# LOAD RECONSTRUCTOR
# -------------------------------------------------------------------------

if recon == "unet":
    model = Unet(in_chans=nChannels, out_chans=nChannels, chans=64)
    model_path = unet_model_path
elif recon == "modl":
    model = DIDN(nChannels, nChannels, num_chans=64, pad_data=True,
                 global_residual=True, n_res_blocks=2)
    model_path = modl_model_path
else:
    raise ValueError(f"Unknown recon='{recon}'. Supported: 'unet', 'modl'.")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# -------------------------------------------------------------------------
# LOAD DATA (SINGLE SCAN/SLICE)
# -------------------------------------------------------------------------

ksp = np.load(ksp_path)        # [ncoils, H, W], complex
mps = np.load(mps_path)        # [ncoils, H, W], complex

# If sensitivity maps are not available, they can be estimated using ESPIRiT
# (requires sigpy). Example:
#   mps = sigpy.mri.app.EspiritCalib(ksp).run()
# Source: https://sigpy.readthedocs.io/en/latest/generated/sigpy.mri.app.EspiritCalib.html

img_gt = np.load(img_gt_path)  # [H, W], complex

H, W = ksp.shape[1], ksp.shape[2]

# Crop and normalize ground truth for the optimization example
img_gt = crop_img(img_gt)                     # e.g., central ROI crop
img_gt = img_gt / (np.abs(img_gt).max() + 1e-12)

# -------------------------------------------------------------------------
# INITIAL MASK / BUDGET
# -------------------------------------------------------------------------

budget = W // us_factor
num_centre_lines = budget // 3  # low-frequency center lines are kept fixed

# Initialize with a variable-density random sampling (VDRS) mask
# Alternatives: low-frequency mask, equispaced mask, LOUPE mask, etc.
initial_mask = make_vdrs_mask(H, W, budget, num_centre_lines)

# -------------------------------------------------------------------------
# CONVERT TO TORCH
# -------------------------------------------------------------------------

ksp = torch.tensor(ksp).to(device)
mps = torch.tensor(mps).to(device)
img_gt = torch.tensor(img_gt).to(device)
initial_mask = torch.tensor(initial_mask).to(device)

# -------------------------------------------------------------------------
# RUN ICD SAMPLING OPTIMIZATION
# -------------------------------------------------------------------------

icd_mask, loss_icd_list = icd_sampling_optimization(
    ksp, mps, img_gt, initial_mask, budget, num_centre_lines,
    model, device, num_icd_iter, nChannels,
    recon, alpha1, alpha2, alpha3, alpha4,
    print_loss=True,
    save_recon=True,
    num_modl_iter=6
)
