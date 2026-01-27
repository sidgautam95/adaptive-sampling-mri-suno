"""
Train MoDL (DIDN-based) on multicoil MR images undersampled by scan/slice-adaptive masks.

Reference:
Aggarwal, Hemant K., Merry P. Mani, and Mathews Jacob.
"MoDL: Model-based deep learning architecture for inverse problems."
IEEE TMI 38.2 (2018): 394-405.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("../utils")
sys.path.append("../models")

from didn import DIDN
from modl_cg_functions import modl_recon_training  # assumes this exists
from utils import loss_fn  # assumes loss_fn exists (NRMSE etc.)

# -------------------------------------------------------------------------
# USER SETTINGS (PLEASE UPDATE THESE)
# -------------------------------------------------------------------------

# Root directory produced by your preprocessing step.
# Expected subfolders:
#   train-img-aliased, train-masks, train-img-gt, train-maps,
#   val-img-aliased,   val-masks,   val-img-gt,   val-maps
data_root = "modl-training-data/"    # <-- CHANGE THIS

# Training hyperparameters
learning_rate = 1e-4
nepochs = 100

# Model configuration (kept consistent with your current setup)
nChannels = 2
num_modl_iter = 6   # used inside modl_recon_training if applicable

# Output files
out_model_path = "model.pt"
out_loss_plot = "loss.png"

# -------------------------------------------------------------------------
# DEVICE SETUP
# -------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# MODEL / OPTIMIZER
# -------------------------------------------------------------------------

model = DIDN(nChannels, nChannels, num_chans=64, pad_data=True,
             global_residual=True, n_res_blocks=2).float().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

# -------------------------------------------------------------------------
# FILE LISTS
# -------------------------------------------------------------------------

train_aliased_dir = os.path.join(data_root, "train-img-aliased")
val_aliased_dir   = os.path.join(data_root, "val-img-aliased")

train_filenames = sorted([f for f in os.listdir(train_aliased_dir) if f.endswith(".npy")])
val_filenames   = sorted([f for f in os.listdir(val_aliased_dir) if f.endswith(".npy")])

ntrain = len(train_filenames)
nval = len(val_filenames)

if ntrain == 0 or nval == 0:
    raise RuntimeError("No training/validation files found. Please check data_root and subfolder names.")

train_loss_hist = []
val_loss_hist = []

# -------------------------------------------------------------------------
# HELPER: PARSE SCAN ID AND SLICE INDEX FROM FILENAME
# Expected pattern: train_img_aliased_<scan>_slc<idx>.npy (or val_...)
# -------------------------------------------------------------------------

def parse_scan_and_slice(fname):
    base = os.path.splitext(fname)[0]  # remove .npy
    # Example: train_img_aliased_..._slc12
    if "_slc" not in base:
        raise ValueError(f"Filename does not contain '_slc': {fname}")
    prefix, slc_str = base.rsplit("_slc", 1)
    scan = prefix.split("_")[-1]       # last token before _slc
    slc_idx = slc_str
    return scan, slc_idx

# -------------------------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------------------------

for epoch in range(nepochs):

    model.train()
    train_loss_total = 0.0

    for fname in train_filenames:

        scan, slc_idx = parse_scan_and_slice(fname)

        # Load training arrays
        img_gt = np.load(os.path.join(data_root, f"train-img-gt/train_img_gt_{scan}_slc{slc_idx}.npy"))
        img_aliased = np.load(os.path.join(data_root, f"train-img-aliased/train_img_aliased_{scan}_slc{slc_idx}.npy"))
        mask = np.load(os.path.join(data_root, f"train-masks/train_masks_{scan}_slc{slc_idx}.npy"))
        mps = np.load(os.path.join(data_root, f"train-maps/train_maps_{scan}_slc{slc_idx}.npy"))

        # Forward (MoDL reconstruction)
        img_recon_modl = modl_recon_training(img_aliased, mask, mps, model, device=device)

        target = torch.tensor(img_gt).to(device).float().unsqueeze(0)

        optimizer.zero_grad()
        loss = loss_fn(target, img_recon_modl)
        loss.backward()
        optimizer.step()

        train_loss_total += float(loss)

    # ---------------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------------

    model.eval()
    val_loss_total = 0.0

    with torch.no_grad():
        for fname in val_filenames:

            scan, slc_idx = parse_scan_and_slice(fname)

            img_gt = np.load(os.path.join(data_root, f"val-img-gt/val_img_gt_{scan}_slc{slc_idx}.npy"))
            img_aliased = np.load(os.path.join(data_root, f"val-img-aliased/val_img_aliased_{scan}_slc{slc_idx}.npy"))
            mask = np.load(os.path.join(data_root, f"val-masks/val_masks_{scan}_slc{slc_idx}.npy"))
            mps = np.load(os.path.join(data_root, f"val-maps/val_maps_{scan}_slc{slc_idx}.npy"))

            img_recon_modl = modl_recon_training(img_aliased, mask, mps, model, device=device)
            target = torch.tensor(img_gt).to(device).float().unsqueeze(0)

            loss = loss_fn(target, img_recon_modl)
            val_loss_total += float(loss)

    train_loss_epoch = train_loss_total / ntrain
    val_loss_epoch = val_loss_total / nval

    train_loss_hist.append(train_loss_epoch)
    val_loss_hist.append(val_loss_epoch)

    scheduler.step(val_loss_epoch)

    print(f"Epoch {epoch+1:03d}/{nepochs} | Train Loss: {train_loss_epoch:.6f} | Val Loss: {val_loss_epoch:.6f}")

    # ---------------------------------------------------------------------
    # SAVE LOSS PLOT + CHECKPOINT
    # ---------------------------------------------------------------------

    plt.figure()
    plt.plot(np.array(train_loss_hist))
    plt.plot(np.array(val_loss_hist))
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.title("MoDL Training: Loss vs Epoch")
    plt.savefig(out_loss_plot, dpi=200, bbox_inches="tight")
    plt.close()

    torch.save(model.state_dict(), out_model_path)
