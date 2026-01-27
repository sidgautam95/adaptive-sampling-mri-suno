"""
Train a U-Net reconstructor on multi-coil MR images undersampled by scan/slice-adaptive masks.

This script assumes preprocessing has already created:
  - train-img-aliased / val-img-aliased
  - train-img-gt      / val-img-gt

Each .npy file is expected to contain one slice (2-channel representation).
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("../utils")
sys.path.append("../models")

from unet_fbr import Unet
from utils import loss_fn

# -------------------------------------------------------------------------
# USER SETTINGS (PLEASE UPDATE THESE)
# -------------------------------------------------------------------------

# Root directory produced by your preprocessing step.
# Expected subfolders:
#   train-img-aliased, train-img-gt, val-img-aliased, val-img-gt
data_root = "modl-training-data/"      # <-- CHANGE THIS

learning_rate = 1e-4
nepochs = 100

# Output files
out_model_path = "unet_model.pt"
out_loss_plot = "unet_loss.png"

# -------------------------------------------------------------------------
# DEVICE SETUP
# -------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------------------
# MODEL / OPTIMIZER
# -------------------------------------------------------------------------

model = Unet(in_chans=2, out_chans=2, num_pool_layers=4, chans=64).float().to(device)
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
eps = 1e-12

# -------------------------------------------------------------------------
# HELPER: PARSE SCAN ID AND SLICE INDEX FROM FILENAME
# Expected pattern: train_img_aliased_<scan>_slc<idx>.npy (or val_...)
# -------------------------------------------------------------------------

def parse_scan_and_slice(fname):
    base = os.path.splitext(fname)[0]  # remove .npy
    if "_slc" not in base:
        raise ValueError(f"Filename does not contain '_slc': {fname}")
    prefix, slc_str = base.rsplit("_slc", 1)
    scan = prefix.split("_")[-1]
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

        img_gt = np.load(os.path.join(data_root, f"train-img-gt/train_img_gt_{scan}_slc{slc_idx}.npy"))
        img_aliased = np.load(os.path.join(data_root, f"train-img-aliased/train_img_aliased_{scan}_slc{slc_idx}.npy"))

        target = torch.tensor(img_gt).to(device).float().unsqueeze(0)
        inp = torch.tensor(img_aliased).to(device).float().unsqueeze(0)

        # Normalize by max magnitude (safe)
        target = target / (torch.abs(target).max() + eps)
        inp = inp / (torch.abs(inp).max() + eps)

        pred = model(inp)

        optimizer.zero_grad()
        loss = loss_fn(target, pred)
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

            target = torch.tensor(img_gt).to(device).float().unsqueeze(0)
            inp = torch.tensor(img_aliased).to(device).float().unsqueeze(0)

            target = target / (torch.abs(target).max() + eps)
            inp = inp / (torch.abs(inp).max() + eps)

            pred = model(inp)
            loss = loss_fn(target, pred)

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
    plt.title("U-Net Training: Loss vs Epoch")
    plt.savefig(out_loss_plot, dpi=200, bbox_inches="tight")
    plt.close()

    torch.save(model.state_dict(), out_model_path)
