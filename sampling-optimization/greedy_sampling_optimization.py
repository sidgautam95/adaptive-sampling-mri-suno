"""
Greedy scan-adaptive mask optimization for multi-coil MRI.

Reference:
  Gözcü, Baran, et al. "Learning-based compressive MRI."
  IEEE Transactions on Medical Imaging 37.6 (2018): 1394–1406.

Notes:
- Assumes 1D Cartesian undersampling along the ky axis (width dimension).
- Masks are 2D arrays/tensors of shape [H, W], constant across H (rows identical).
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional

import numpy as np
import torch

# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from utils import (
    compute_loss,
    plot_mr_image,
    unet_recon_batched,
    modl_recon_batched,
)

EPS = 1e-12


def greedy_sampling_optimization(
    ksp: torch.Tensor,
    mps: torch.Tensor,
    img_gt: torch.Tensor,
    initial_mask: torch.Tensor,
    budget: int,
    model,
    device: torch.device,
    nChannels: int = 2,
    recon: str = "unet",
    alpha1: float = 1.0,
    alpha2: float = 0.0,
    alpha3: float = 0.0,
    alpha4: float = 0.0,
    print_loss: bool = False,
    save_outputs: bool = True,
    output_dir: str = "outputs-greedy",
    plot_every: int = 0,
    num_modl_iter: int = 6,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Parameters
    ----------
    ksp : torch.Tensor
        Complex k-space, shape [nCoils, H, W].
    mps : torch.Tensor
        Complex sensitivity maps, shape [nCoils, H, W].
    img_gt : torch.Tensor
        Ground-truth complex image, shape [H, W] (or compatible).
    initial_mask : torch.Tensor
        Initial mask, shape [H, W], boolean or 0/1.
        Assumed to be identical along height (i.e., 1D ky mask broadcast).
    budget : int
        Total number of ky lines to sample (including center lines).
    model : torch.nn.Module
        Trained reconstructor (U-Net or MoDL).
    device : torch.device
        cpu/cuda device.
    recon : {"unet", "modl"}
        Reconstruction backend used for evaluation during optimization.
    alpha1..alpha4 : float
        Weights used in compute_loss() (NRMSE, SSIM, NMAE, HFEN).
    save_outputs : bool
        Save intermediate outputs (mask/loss curves/plots).
    output_dir : str
        Where to write outputs if save_outputs=True.
    plot_every : int
        If >0, save plots every `plot_every` greedy iterations (and at the end).
        If 0, only saves at the end (if save_outputs=True).
    num_modl_iter : int
        MoDL unroll iterations during evaluation.

    Returns
    -------
    greedy_mask : torch.Tensor
        Optimized mask, shape [H, W].
    loss_history : list[float]
        Loss after each greedy addition step.
    """
    if recon not in {"unet", "modl"}:
        raise ValueError("recon must be 'unet' or 'modl'.")

    # Ensure tensor types + device
    img_gt = img_gt.to(device)
    initial_mask = initial_mask.to(device)

    # Normalize ground truth (avoid div by 0)
    img_gt = img_gt / (torch.abs(img_gt).max() + EPS)

    nCoils, H, W = ksp.shape
    if budget <= 0 or budget > W:
        raise ValueError(f"budget must be in (0, {W}] but got {budget}.")

    # We assume 1D ky sampling => take any row (0th) as the ky mask
    # ky_mask: [W]
    ky_mask = initial_mask[0].bool()
    current_lines = int(ky_mask.sum().item())

    if current_lines > budget:
        raise ValueError(f"Initial mask already has {current_lines} lines, exceeds budget {budget}.")

    us_factor = W // budget

    greedy_mask = initial_mask.bool().clone()
    loss_history: List[float] = []

    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

    # Initial reconstruction + loss
    if recon == "unet":
        img_recon_initial = unet_recon_batched(ksp, mps, greedy_mask, model, device=device)
    else:
        img_recon_initial = modl_recon_batched(ksp, mps, greedy_mask, model=model, num_iter=num_modl_iter, device=device)

    img_recon_initial = torch.squeeze(img_recon_initial).to(device)
    loss_initial = compute_loss(img_gt, img_recon_initial, alpha1, alpha2, alpha3, alpha4).item()

    if print_loss:
        print("Running greedy sampling optimization")
        print(f"Undersampling factor: {us_factor}x | Budget: {budget} | Initial lines: {current_lines}")
        print(f"Initial loss: {loss_initial:.6f} | Reconstructor: {recon}")

    # Greedy additions until we reach budget
    greedy_iter = 0
    while int(greedy_mask[0].bool().sum().item()) < budget:
        greedy_iter += 1

        # Candidate ky indices are where ky_mask == False
        ky_mask = greedy_mask[0].bool()                # [W]
        candidate_ky = torch.where(~ky_mask)[0]         # indices along width

        num_candidates = int(candidate_ky.numel())
        if num_candidates == 0:
            break

        # Candidate masks: [Ncand, H, W]
        # Start by repeating current mask
        candidate_masks = greedy_mask.unsqueeze(0).repeat(num_candidates, 1, 1)

        # For each candidate, flip one ky line to True (along width axis)
        # Mask is [H, W], so set [:, ky] = True
        for i, ky in enumerate(candidate_ky):
            candidate_masks[i, :, ky] = True

        # Reconstruct all candidates
        if recon == "unet":
            img_recons = unet_recon_batched(ksp, mps, candidate_masks, model, device=device)
        else:
            img_recons = modl_recon_batched(
                ksp, mps, candidate_masks, model=model, num_iter=num_modl_iter, device=device
            )

        # Compute loss for each candidate
        losses = torch.empty((num_candidates,), device=device, dtype=torch.float32)
        for i in range(num_candidates):
            losses[i] = compute_loss(img_gt, img_recons[i].to(device), alpha1, alpha2, alpha3, alpha4)

        best_idx = int(torch.argmin(losses).item())
        greedy_mask = candidate_masks[best_idx].clone()
        best_loss = float(losses[best_idx].item())
        loss_history.append(best_loss)

        if print_loss:
            lines_now = int(greedy_mask[0].bool().sum().item())
            print(f"Iter {greedy_iter:03d} | lines={lines_now}/{budget} | loss={best_loss:.6f}")

        # Optionally save
        if save_outputs:
            np.savez(
                os.path.join(output_dir, f"greedy_mask_{us_factor}x.npz"),
                greedy_mask_1d=greedy_mask[0].detach().cpu().numpy().astype(bool),
                loss_history=np.array(loss_history, dtype=np.float32),
                initial_mask_1d=initial_mask[0].detach().cpu().numpy().astype(bool),
            )

            do_plot = (plot_every > 0 and (greedy_iter % plot_every == 0)) or (int(greedy_mask[0].sum()) == budget)
            if do_plot and plt is not None:
                # Recon of best candidate for plotting
                img_recon_greedy = torch.squeeze(img_recons[best_idx]).to(device)

                # Recon figure
                plt.figure(figsize=(10, 6))
                plt.subplot(2, 3, 1)
                plot_mr_image(img_gt, title="Ground Truth")
                plt.subplot(2, 3, 2)
                plot_mr_image(img_recon_initial, title=f"Initial\nloss={loss_initial:.3f}")
                plt.subplot(2, 3, 3)
                plot_mr_image(img_recon_greedy, title=f"Greedy\nloss={best_loss:.3f}")
                plt.subplot(2, 3, 5)
                plot_mr_image(torch.abs(img_gt - img_recon_initial), vmax=0.2, normalize=False, title="Err init")
                plt.subplot(2, 3, 6)
                plot_mr_image(torch.abs(img_gt - img_recon_greedy), vmax=0.2, normalize=False, title="Err greedy")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"recon_{us_factor}x_iter{greedy_iter:03d}.png"), dpi=200)
                plt.close()

                # Mask + curve figure
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(initial_mask.detach().cpu().numpy(), cmap="gray")
                plt.title("Initial mask")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(greedy_mask.detach().cpu().numpy(), cmap="gray")
                lines_now = int(greedy_mask[0].bool().sum().item())
                plt.title(f"Greedy mask\nlines={lines_now}, loss={best_loss:.3f}")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.plot(np.arange(len(loss_history)) + 1, np.array(loss_history))
                plt.grid(True)
                plt.xlabel("Greedy step")
                plt.ylabel("Loss")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"greedy_mask_{us_factor}x_iter{greedy_iter:03d}.png"), dpi=200)
                plt.close()

    return greedy_mask, loss_history
