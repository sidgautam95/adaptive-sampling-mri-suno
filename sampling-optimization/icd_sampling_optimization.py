"""
ICD-based scan-adaptive mask optimization for multi-coil MRI.

Author: Siddhant Gautam, Michigan State University

Notes
-----
- Assumes 1D Cartesian undersampling along ky (width dimension).
- Masks are 2D arrays/tensors [H, W] and typically constant across H.
- The algorithm keeps the center (ACS) lines fixed and only relocates non-center lines.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch

# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from utils import (
    crop_img,
    compute_loss,
    plot_mr_image,
    unet_recon_batched,
    modl_recon_batched,
)

EPS = 1e-12


def icd_sampling_optimization(
    ksp: torch.Tensor,
    mps: torch.Tensor,
    img_gt: torch.Tensor,
    initial_mask: torch.Tensor,
    budget: int,
    num_centre_lines: int,
    model,
    device: torch.device,
    num_icd_passes: int = 1,
    nChannels: int = 2,
    recon: str = "unet",
    alpha1: float = 1.0,
    alpha2: float = 0.0,
    alpha3: float = 0.0,
    alpha4: float = 0.0,
    print_loss: bool = False,
    save_outputs: bool = True,
    save_recon: bool = False,
    output_dir: str = "outputs-icd",
    plot_every: int = 0,
    num_modl_iter: int = 6,
) -> Tuple[torch.Tensor, np.ndarray]:
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
        Initial mask, shape [H, W], bool or 0/1.
    budget : int
        Total ky lines to sample (including center lines).
    num_centre_lines : int
        Number of fixed center (ACS) ky lines.
    model : torch.nn.Module
        Trained reconstructor.
    device : torch.device
        cpu/cuda.
    num_icd_passes : int
        Number of full passes over movable lines.
    recon : {"unet","modl"}
        Which reconstruction model to use for evaluation during ICD.
    alpha1..alpha4 : float
        Weights used in compute_loss().
    save_outputs : bool
        Save npz logs (and optionally plots).
    save_recon : bool
        Save final reconstruction array.
    output_dir : str
        Output directory (created if needed).
    plot_every : int
        If > 0, save plots every `plot_every` accepted moves (and at the end).
        If 0, only saves at the end (if save_outputs=True and matplotlib is available).
    num_modl_iter : int
        MoDL unroll iterations during evaluation.

    Returns
    -------
    icd_mask : torch.Tensor
        Optimized mask [H, W].
    loss_history : np.ndarray
        Loss values after each attempted move (includes initial loss as first entry).
    """
    if recon not in {"unet", "modl"}:
        raise ValueError("recon must be 'unet' or 'modl'.")

    nCoils, H, W = ksp.shape
    if budget <= 0 or budget > W:
        raise ValueError(f"budget must be in (0, {W}] but got {budget}.")
    if num_centre_lines <= 0 or num_centre_lines > budget:
        raise ValueError("num_centre_lines must be in (0, budget].")

    model = model.to(device)
    model.eval()

    img_gt = img_gt.to(device)
    initial_mask = initial_mask.to(device).bool()

    # Normalize + crop GT for consistent metric computation (mirrors your original intent)
    img_gt = crop_img(img_gt)
    img_gt = img_gt / (torch.abs(img_gt).max() + EPS)
    if nChannels == 1:
        img_gt = torch.abs(img_gt)

    icd_mask = initial_mask.clone()

    # Enforce center (ACS) lines in the mask (fixed, never moved)
    c1 = (W - num_centre_lines) // 2
    c2 = c1 + num_centre_lines
    centre_ky = torch.arange(c1, c2, device=device)

    icd_mask[:, centre_ky] = True

    # Basic info
    us_factor = W // budget
    if print_loss:
        print("Running ICD sampling optimization")
        print(f"Undersampling factor: {us_factor}x | Budget: {budget} | Centre lines: {num_centre_lines}")
        print(f"Reconstructor: {recon} | ICD passes: {num_icd_passes}")

    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

    # Initial reconstruction + loss
    with torch.no_grad():
        if recon == "unet":
            img_recon_initial = unet_recon_batched(ksp, mps, icd_mask, model, device=device)
        else:
            img_recon_initial = modl_recon_batched(ksp, mps, icd_mask, model=model, num_iter=num_modl_iter, device=device)

    img_recon_initial = torch.squeeze(img_recon_initial).to(device)
    loss_current = compute_loss(img_gt, img_recon_initial, alpha1, alpha2, alpha3, alpha4)

    img_recon_best = img_recon_initial.clone()
    loss_history: List[float] = [float(loss_current.item())]

    moves_accepted = 0
    attempts = 0

    # Determine movable ky lines: all sampled ky except centre_ky
    # We recompute this each pass since the mask changes.
    for p in range(num_icd_passes):
        if print_loss:
            print(f"Pass {p+1}/{num_icd_passes}")

        # Current sampled ky indices
        sampled_ky = torch.where(icd_mask[0])[0]
        movable_ky = sampled_ky[~torch.isin(sampled_ky, centre_ky)]

        # Loop over each movable ky line and try relocating it
        for ky_to_move in movable_ky.tolist():
            attempts += 1

            # Candidate ky locations are currently NOT sampled
            candidate_ky = torch.where(~icd_mask[0])[0]  # along width dim

            # Create candidate masks: remove ky_to_move, add one candidate ky
            num_candidates = int(candidate_ky.numel())
            if num_candidates == 0:
                loss_history.append(float(loss_current.item()))
                continue

            candidate_masks = icd_mask.unsqueeze(0).repeat(num_candidates, 1, 1)

            # Remove the line we are moving
            candidate_masks[:, :, ky_to_move] = False

            # Add each candidate ky line
            for i, ky_new in enumerate(candidate_ky):
                candidate_masks[i, :, ky_new] = True

            # Ensure centre lines remain True (safety)
            candidate_masks[:, :, centre_ky] = True

            # Reconstruct each candidate
            with torch.no_grad():
                if recon == "unet":
                    img_recons = unet_recon_batched(ksp, mps, candidate_masks, model, device=device)
                else:
                    img_recons = modl_recon_batched(
                        ksp, mps, candidate_masks, model=model, num_iter=num_modl_iter, device=device
                    )

            # Evaluate loss for each candidate
            losses = torch.empty((num_candidates,), device=device, dtype=torch.float32)
            for i in range(num_candidates):
                losses[i] = compute_loss(img_gt, img_recons[i].to(device), alpha1, alpha2, alpha3, alpha4)

            best_idx = int(torch.argmin(losses).item())
            best_loss = losses[best_idx]

            # Accept move if improvement
            if best_loss < loss_current:
                icd_mask = candidate_masks[best_idx].clone()
                loss_current = best_loss.clone()
                img_recon_best = torch.squeeze(img_recons[best_idx]).to(device)
                moves_accepted += 1

                if print_loss:
                    print(f"  attempt {attempts:04d} | moved ky={ky_to_move:04d} | ACCEPT | loss={loss_current.item():.6f}")

            else:
                if print_loss:
                    print(f"  attempt {attempts:04d} | moved ky={ky_to_move:04d} | reject | loss={loss_current.item():.6f}")

            loss_history.append(float(loss_current.item()))

            # Save logs
            if save_outputs:
                np.savez(
                    os.path.join(output_dir, f"icd_mask_{us_factor}x.npz"),
                    icd_mask_1d=icd_mask[0].detach().cpu().numpy().astype(bool),
                    loss_history=np.array(loss_history, dtype=np.float32),
                    initial_mask_1d=initial_mask[0].detach().cpu().numpy().astype(bool),
                    num_centre_lines=int(num_centre_lines),
                    budget=int(budget),
                )

                if save_recon:
                    np.save(
                        os.path.join(output_dir, f"img_recon_icd_{us_factor}x.npy"),
                        img_recon_best.detach().cpu().numpy(),
                    )

                # Optional plotting cadence
                do_plot = (
                    (plot_every > 0 and moves_accepted > 0 and (moves_accepted % plot_every == 0))
                    or (p == num_icd_passes - 1 and ky_to_move == movable_ky.tolist()[-1])
                )
                if do_plot and plt is not None:
                    _save_icd_plots(
                        img_gt=img_gt,
                        img_recon_initial=img_recon_initial,
                        img_recon_best=img_recon_best,
                        initial_mask=initial_mask,
                        icd_mask=icd_mask,
                        loss_history=loss_history,
                        us_factor=us_factor,
                        outdir=output_dir,
                        tag=f"pass{p+1}_attempt{attempts:04d}",
                    )

    return icd_mask, np.array(loss_history, dtype=np.float32)


def _save_icd_plots(
    img_gt: torch.Tensor,
    img_recon_initial: torch.Tensor,
    img_recon_best: torch.Tensor,
    initial_mask: torch.Tensor,
    icd_mask: torch.Tensor,
    loss_history: List[float],
    us_factor: int,
    outdir: str,
    tag: str,
):
    """Small helper for saving ICD plots (requires matplotlib)."""
    # Recon figure
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 3, 1)
    plot_mr_image(img_gt, title="Ground Truth")
    plt.subplot(2, 3, 2)
    plot_mr_image(img_recon_initial, title=f"Initial\nloss={loss_history[0]:.3f}")
    plt.subplot(2, 3, 3)
    plot_mr_image(img_recon_best, title=f"ICD best\nloss={loss_history[-1]:.3f}")
    plt.subplot(2, 3, 5)
    plot_mr_image(torch.abs(img_gt - img_recon_initial), vmax=0.2, normalize=False, title="Err init")
    plt.subplot(2, 3, 6)
    plot_mr_image(torch.abs(img_gt - img_recon_best), vmax=0.2, normalize=False, title="Err ICD")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"recon_{us_factor}x_{tag}.png"), dpi=200)
    plt.close()

    # Mask + curve
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(initial_mask.detach().cpu().numpy(), cmap="gray")
    plt.title("Initial mask")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(icd_mask.detach().cpu().numpy(), cmap="gray")
    plt.title("ICD mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.plot(np.arange(len(loss_history)), np.array(loss_history))
    plt.grid(True)
    plt.xlabel("Attempt")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"icd_mask_{us_factor}x_{tag}.png"), dpi=200)
    plt.close()
