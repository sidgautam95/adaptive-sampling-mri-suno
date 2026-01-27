"""
Inference script: run a pretrained MoDL reconstructor on a SUNO-predicted mask
for a single multicoil MRI scan/slice.

Expected inputs:
  ksp  : complex array, shape [ncoils, H, W]
  mps  : complex array, shape [ncoils, H, W]  (coil sensitivity maps)
  mask : binary array, shape [H, W]           (SUNO-predicted sampling mask)

Output:
  test_img_recon.npy : reconstructed image (complex or 2-channel, depending on implementation)
"""

import os
import sys
import numpy as np
import torch

sys.path.append("../utils")
sys.path.append("../models")

from didn import DIDN
from modl_cg_functions import modl_recon  # assumes modl_recon is defined here

# -------------------------------------------------------------------------
# USER SETTINGS (PLEASE UPDATE THESE PATHS AS NEEDED)
# -------------------------------------------------------------------------

nChannels = 2

# Pretrained MoDL weights
modl_path = "../../saved-models/modl_didn_fastmri_vdrs_4x.pt"  # <-- CHANGE IF NEEDED

# Test data (single scan/slice)
ksp_path = "../../data/ksp.npy"       # <-- CHANGE THIS
mps_path = "../../data/mps.npy"       # <-- CHANGE THIS

# SUNO predicted mask (output of nearest_neighbor_search.py)
mask_path = "suno_mask.npy"           # <-- CHANGE IF NEEDED

# Output filename
out_recon_path = "test_img_recon.npy"

# -------------------------------------------------------------------------
# DEVICE SETUP
# -------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------------------

model = DIDN(nChannels, nChannels, num_chans=64, pad_data=True,
             global_residual=True, n_res_blocks=2)

model.load_state_dict(torch.load(modl_path, map_location=device))
model.to(device)
model.eval()

print(f"Loaded MoDL weights from: {modl_path}")

# -------------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------------

ksp = np.load(ksp_path)    # expected: [ncoils, H, W], complex
mps = np.load(mps_path)    # expected: [ncoils, H, W], complex
mask = np.load(mask_path)  # expected: [H, W], binary

ksp = torch.tensor(ksp).to(device)
mps = torch.tensor(mps).to(device)
mask = torch.tensor(mask).to(device)

# -------------------------------------------------------------------------
# RUN MODL RECONSTRUCTION
# -------------------------------------------------------------------------

with torch.no_grad():
    img_recon_modl = modl_recon(ksp, mps, mask, model, device=device)

# -------------------------------------------------------------------------
# SAVE OUTPUT
# -------------------------------------------------------------------------

np.save(out_recon_path, torch.squeeze(img_recon_modl).cpu().numpy())
print(f"Saved reconstruction to: {out_recon_path}")
