"""
Helper utilities for SUNO / MoDL / U-Net experiments.

Includes:
  - sampling mask constructors (VDRS, low-frequency/ACS)
  - cropping utilities
  - metric/loss helpers (NRMSE, NMAE, combined loss)
  - k-space preprocessing (adjoint recon + normalization)
  - batched U-Net / MoDL reconstruction wrappers

Conventions
-----------
- Complex data may be stored as complex dtype (numpy/torch) or as 2-channel real tensors:
    [2, H, W] where channel 0 = real, channel 1 = imag
- k-space and sensitivity maps are complex arrays of shape:
    [ncoils, H, W]
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

# Optional deps (used only if corresponding metrics/plots are requested)
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:  # pragma: no cover
    ssim = None

# Import only what we actually use from modl_cg_functions
from modl_cg_functions import (
    fft2,
    ifft2,
    complex_matmul,
    complex_conj,
    OPAT2,
    CG_fn,
)

EPS = 1e-12


# -------------------------------------------------------------------------
# Masks
# -------------------------------------------------------------------------

def make_vdrs_mask(
    height: int,
    width: int,
    budget: int,
    num_centre_lines: int,
    power: float = 4.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    1D variable-density random sampling along width (ky), broadcast across height.

    Parameters
    ----------
    height, width : int
        Mask size.
    budget : int
        Total number of ky lines sampled (including the ACS block).
    num_centre_lines : int
        Size of the fully sampled center (ACS) block.
    power : float
        Exponent for center-heavy sampling (larger -> more center concentration).
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    mask : np.ndarray (bool), shape [height, width]
    """
    if num_centre_lines <= 0:
        raise ValueError("num_centre_lines must be positive.")
    if budget < num_centre_lines or budget > width:
        raise ValueError("budget must be in [num_centre_lines, width].")

    rng = np.random.default_rng(seed)
    mask = np.zeros((height, width), dtype=bool)

    # Center/ACS block
    c1 = (width - num_centre_lines) // 2
    c2 = c1 + num_centre_lines
    mask[:, c1:c2] = True

    remaining = budget - num_centre_lines
    if remaining == 0:
        return mask

    ky = np.arange(width)
    kc = (width - 1) / 2.0
    d = np.abs(ky - kc)
    dmax = max(float(d.max()), 1.0)

    # p(k) ∝ (1 - d/dmax)^power
    w = (1.0 - d / dmax) ** power
    w[c1:c2] = 0.0  # exclude ACS (already included)

    s = float(w.sum())
    if s <= 0:
        raise RuntimeError("VD weights degenerate; check width/ACS/power.")
    probs = w / s

    extra = rng.choice(ky, size=remaining, replace=False, p=probs)
    mask[:, extra] = True
    return mask


def make_lf_mask(height: int, width: int, budget: int) -> np.ndarray:
    """Centered low-frequency mask with `budget` contiguous ky lines."""
    if budget <= 0 or budget > width:
        raise ValueError("budget must be in (0, width].")
    mask = np.zeros((height, width), dtype=bool)
    start = (width - budget) // 2
    end = start + budget
    mask[:, start:end] = True
    return mask


# -------------------------------------------------------------------------
# Cropping and format helpers
# -------------------------------------------------------------------------

def crop_img(img: Union[np.ndarray, Tensor], height: int = 320, width: int = 320):
    """
    Center-crop images.

    Supported shapes:
      - [H, W]
      - [B, H, W]
      - [B, C, H, W]
    """
    if torch.is_tensor(img):
        h, w = img.shape[-2], img.shape[-1]
    else:
        h, w = img.shape[-2], img.shape[-1]

    h1 = (h - height) // 2
    h2 = h1 + height
    w1 = (w - width) // 2
    w2 = w1 + width

    if img.ndim == 2:
        return img[h1:h2, w1:w2]
    if img.ndim == 3:
        return img[:, h1:h2, w1:w2]
    if img.ndim == 4:
        return img[:, :, h1:h2, w1:w2]

    raise ValueError(f"Unsupported shape for crop_img: {img.shape}")


def two_channel_to_complex(img_2channel: Union[np.ndarray, Tensor]):
    """
    Convert a 2-channel real/imag representation into a complex image.

    Input shape:
      - [2, H, W] or [1, 2, H, W] (squeezed internally)

    Returns
    -------
    complex array/tensor of shape [H, W]
    """
    if torch.is_tensor(img_2channel):
        x = torch.squeeze(img_2channel)
        return x[0] + 1j * x[1]
    x = np.squeeze(img_2channel)
    return x[0] + 1j * x[1]


# -------------------------------------------------------------------------
# Metrics / losses
# -------------------------------------------------------------------------

def compute_nrmse_numpy(img_gt: np.ndarray, img_recon: np.ndarray) -> float:
    return float(np.linalg.norm(img_gt - img_recon) / (np.linalg.norm(img_gt) + EPS))


def compute_nrmse(img_gt: Tensor, img_recon: Tensor) -> Tensor:
    return torch.linalg.norm(img_gt - img_recon) / (torch.linalg.norm(img_gt) + EPS)


def compute_nmae(img_gt: Tensor, img_recon: Tensor) -> Tensor:
    return torch.mean(torch.abs(img_gt - img_recon)) / (torch.mean(torch.abs(img_gt)) + EPS)


def compute_loss(
    img_gt: Tensor,
    img_recon: Tensor,
    alpha1: float = 1.0,
    alpha2: float = 0.0,
    alpha3: float = 0.0,
    alpha4: float = 0.0,
) -> Tensor:
    """
    Weighted combination of metrics:
      alpha1 * NRMSE
    + alpha2 * (1 - SSIM)
    + alpha3 * NMAE
    + alpha4 * HFEN   (HFEN must be defined by you if you use alpha4 != 0)

    NOTE: SSIM requires scikit-image. HFEN requires a compute_hfen() function (not provided here).
    """
    denom = alpha1 + alpha2 + alpha3 + alpha4
    if denom == 0:
        raise ValueError("At least one alpha must be non-zero.")

    total = 0.0

    if alpha1 != 0:
        total = total + alpha1 * compute_nrmse(img_gt, img_recon)

    if alpha2 != 0:
        if ssim is None:
            raise ImportError("scikit-image is required for SSIM (alpha2 != 0).")
        gt_np = img_gt.detach().cpu().numpy()
        re_np = img_recon.detach().cpu().numpy()
        ssim_val = ssim(np.abs(gt_np), np.abs(re_np))
        total = total + alpha2 * (1.0 - torch.tensor(ssim_val, device=img_gt.device, dtype=img_gt.dtype))

    if alpha3 != 0:
        total = total + alpha3 * compute_nmae(img_gt, img_recon)

    if alpha4 != 0:
        # If you use HFEN, define compute_hfen somewhere and import it here.
        raise NotImplementedError("HFEN term requested (alpha4 != 0) but compute_hfen is not implemented/imported.")

    return total / denom


def loss_fn(img_input: Tensor, img_output: Tensor) -> Tensor:
    """
    Batch NRMSE loss.

    img_input/img_output: [B, C, H, W]
    """
    bsz = img_input.shape[0]
    nrmse = torch.zeros((bsz,), device=img_input.device, dtype=img_input.dtype)
    for i in range(bsz):
        nrmse[i] = torch.linalg.norm(img_input[i] - img_output[i]) / (torch.linalg.norm(img_input[i]) + EPS)
    return torch.mean(nrmse)


# -------------------------------------------------------------------------
# Visualization (optional)
# -------------------------------------------------------------------------

def plot_mr_image(img, title: str = "", vmax: float = 0.7, colorbar: bool = False,
                  normalize: bool = True, crop: bool = True):
    """
    Quick visualization helper. Requires matplotlib.

    NOTE: kept compatible with your original behavior (flips vertically for knee).
    """
    if plt is None:
        raise ImportError("matplotlib is required for plot_mr_image().")

    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    if normalize:
        img = img / (np.abs(img).max() + EPS)

    img = np.squeeze(img)

    if img.ndim >= 3 and img.shape[0] == 2:
        img = two_channel_to_complex(img)

    if crop:
        img = crop_img(img)

    img = np.flipud(np.abs(img))
    plt.imshow(np.abs(img), cmap="gray", vmax=vmax)
    plt.axis("off")
    plt.title(title)
    if colorbar:
        plt.colorbar()


# -------------------------------------------------------------------------
# Preprocessing: k-space + maps + mask -> aliased image (adjoint recon) in 2-channel form
# -------------------------------------------------------------------------

def preprocess_data(ksp, mps, mask, device: torch.device = torch.device("cpu")):
    """
    Convert complex k-space/mps/mask to:
      - img (2-channel adjoint recon): [2, H, W]
      - mask (2-channel): [2, H, W]
      - mps_tensor: [ncoils, 2, H, W]

    Inputs
    ------
    ksp : complex array/tensor [ncoils, H, W]
    mps : complex array/tensor [ncoils, H, W]
    mask: bool/float array/tensor [H, W]
    """
    if not torch.is_tensor(ksp):
        ksp = torch.tensor(ksp, device=device)
    else:
        ksp = ksp.to(device)

    if not torch.is_tensor(mps):
        mps = torch.tensor(mps, device=device)
    else:
        mps = mps.to(device)

    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, device=device)
    else:
        mask = mask.to(device)

    ncoils, height, width = ksp.shape

    # Normalize k-space and maps (avoid division by 0)
    kmax = torch.abs(ksp).max() + EPS
    smax = torch.abs(mps).max() + EPS

    k_r = torch.real(ksp) / kmax
    k_i = torch.imag(ksp) / kmax
    s_r = torch.real(mps) / smax
    s_i = torch.imag(mps) / smax

    # Two-channel: [2, ncoils, H, W]
    k_np = torch.stack((k_r, k_i), dim=0)
    s_np = torch.stack((s_r, s_i), dim=0)

    # Mask as 2-channel: [2, H, W]
    mask_2ch = mask.unsqueeze(0).repeat(2, 1, 1).float()

    # k-space for fft2/ifft2 expects last dim=2
    A_k = k_np.permute(1, 0, 2, 3)  # [ncoils, 2, H, W]
    A_I = ifft2(A_k.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [ncoils, 2, H, W]

    mps_tensor = s_np.permute(1, 0, 2, 3)  # [ncoils, 2, H, W]

    adjoint_recon = torch.sum(complex_matmul(A_I, complex_conj(mps_tensor)), dim=0)  # [2, H, W]
    scale = torch.max(torch.abs(adjoint_recon)) + EPS

    A_I = A_I / scale
    A_k = fft2(A_I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [ncoils, 2, H, W]

    AT = OPAT2(mps_tensor)
    img = AT(A_k, mask_2ch)

    return img, mask_2ch, mps_tensor


# -------------------------------------------------------------------------
# Reconstruction wrappers
# -------------------------------------------------------------------------

def unet_recon_batched(ksp, mps, masks, model, batch_size: int = 4, device: torch.device = torch.device("cpu")):
    """
    Run U-Net reconstruction for a set of masks on the same (ksp, mps).

    ksp: [ncoils, H, W] complex
    mps: [ncoils, H, W] complex
    masks: [Nm, H, W] or [H, W]
    """
    if not torch.is_tensor(masks):
        masks = torch.tensor(masks, device=device)
    else:
        masks = masks.to(device)

    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    nImages, height, width = masks.shape
    nchannels = 2

    img_aliased = torch.zeros((nImages, nchannels, height, width), device=device)

    for i in range(nImages):
        with torch.no_grad():
            img_aliased[i], _, _ = preprocess_data(ksp, mps, masks[i], device=device)

    img_h, img_w = 320, 320
    img_recon = torch.zeros((nImages, img_h, img_w), dtype=torch.complex64, device=device)

    # Batch indices
    idxs = np.arange(nImages)
    splits = np.array_split(idxs, max(1, nImages // batch_size + 1))

    model.eval()
    with torch.no_grad():
        for batch in splits:
            if len(batch) == 0:
                continue
            out = model(crop_img(img_aliased[batch])).float()  # [B, 2, 320, 320]
            img_recon[batch] = (out[:, 0] + 1j * out[:, 1]).to(device)

    # Normalize each recon
    for i in range(nImages):
        img_recon[i] = img_recon[i] / (torch.abs(img_recon[i]).max() + EPS)

    return torch.squeeze(img_recon)


def modl_recon_training(
    img_aliased,
    mask,
    mps,
    model,
    tol: float = 1e-5,
    lamda: float = 1e2,
    num_iter: int = 6,
    device: torch.device = torch.device("cpu"),
    print_loss: bool = False,
):
    """
    MoDL reconstruction for training (inputs are already precomputed arrays):
      img_aliased: [2, H, W]
      mask      : [2, H, W] or [H, W]
      mps       : [ncoils, 2, H, W]
    """
    if not torch.is_tensor(img_aliased):
        img_aliased = torch.tensor(img_aliased, device=device)
    else:
        img_aliased = img_aliased.to(device)

    if not torch.is_tensor(mps):
        mps = torch.tensor(mps, device=device)
    else:
        mps = mps.to(device)

    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, device=device)
    else:
        mask = mask.to(device)

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).repeat(2, 1, 1)

    img_aliased = img_aliased.float().unsqueeze(0)  # [1, 2, H, W]
    mps = mps.float().unsqueeze(0)                  # [1, ncoils, 2, H, W]
    mask = mask.float().unsqueeze(0)                # [1, 2, H, W]

    net_input = img_aliased.clone()

    for _ in range(num_iter):
        net_output = model(net_input)
        cg_output = CG_fn(net_output, tol=tol, lamda=lamda, mps=mps, mask=mask,
                          aliased_image=img_aliased, device=device)
        net_input = cg_output.clone()

    return net_input.clone()


def modl_recon(
    ksp,
    mps,
    mask,
    model,
    tol: float = 1e-5,
    lamda: float = 1e2,
    num_iter: int = 6,
    crop: bool = True,
    device: torch.device = torch.device("cpu"),
):
    """
    MoDL reconstruction for inference from raw (ksp, mps, mask).

    Returns a complex image (cropped + normalized by default).
    """
    if torch.is_tensor(ksp):
        ksp_t = ksp.to(device)
    else:
        ksp_t = torch.tensor(ksp, device=device)

    if torch.is_tensor(mps):
        mps_t = mps.to(device)
    else:
        mps_t = torch.tensor(mps, device=device)

    if torch.is_tensor(mask):
        mask_t = mask.to(device)
    else:
        mask_t = torch.tensor(mask, device=device)

    with torch.no_grad():
        img_aliased, mask_2ch, mps_tensor = preprocess_data(ksp_t, mps_t, mask_t, device=device)

        img_aliased = img_aliased.float().unsqueeze(0)
        mps_tensor = mps_tensor.float().unsqueeze(0)
        mask_2ch = mask_2ch.float().unsqueeze(0)

        net_input = img_aliased.clone()

        for _ in range(num_iter):
            net_output = model(net_input)
            cg_output = CG_fn(net_output, tol=tol, lamda=lamda, mps=mps_tensor, mask=mask_2ch,
                              aliased_image=img_aliased, device=device)
            net_input = cg_output.clone()

        img_recon = net_input.clone()  # [1, 2, H, W]
        img_recon_c = two_channel_to_complex(img_recon)  # [H, W] complex

        if crop:
            img_recon_c = crop_img(img_recon_c)

        img_recon_c = img_recon_c / (torch.abs(img_recon_c).max() + EPS)

    return img_recon_c


def modl_recon_batched(
    ksp,
    mps,
    masks,
    model,
    tol: float = 1e-5,
    lamda: float = 1e2,
    num_iter: int = 6,
    device: torch.device = torch.device("cpu"),
):
    """
    Run MoDL reconstruction for a set of masks on the same (ksp, mps).

    masks: [Nm, H, W] or [H, W]
    returns: [Nm, 320, 320] complex (cropped inside modl_recon)
    """
    if not torch.is_tensor(masks):
        masks = torch.tensor(masks)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    masks = masks.to(device)
    nmasks = masks.shape[0]

    out = torch.zeros((nmasks, 320, 320), dtype=torch.complex64, device=device)

    for i in range(nmasks):
        with torch.no_grad():
            out[i] = modl_recon(ksp, mps, masks[i], model, tol=tol, lamda=lamda,
                                num_iter=num_iter, crop=True, device=device)

    return torch.squeeze(out)
