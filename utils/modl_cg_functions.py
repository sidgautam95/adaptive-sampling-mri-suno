"""
MoDL-CG utilities for multicoil Cartesian MRI.

This module provides:
  - Centered FFT/IFFT utilities for 2D complex data stored as (..., 2) real/imag
  - Complex arithmetic helpers (mul, conj)
  - SENSE forward / adjoint operators
  - (A^H A + λI) operator for Cartesian MRI
  - Conjugate Gradient (CG) block + differentiable CG layer used in MoDL

Notes
-----
- Complex tensors are represented with the last dimension size 2: [..., 2] = (real, imag).
- Shapes in operator classes follow the conventions used in this repository.

Inspired by:
https://github.com/JeffFessler/BLIPSrecon/tree/main
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from torch.optim import lr_scheduler


# -----------------------------
# FFT helpers (centered FFT2/IFFT2)
# -----------------------------

def fft_new(image: Tensor, ndim: int, normalized: bool = False) -> Tensor:
    """N-D FFT for complex-as-real tensor with last dim size 2."""
    norm = "ortho" if normalized else None
    dims = tuple(range(-ndim, 0))
    return torch.view_as_real(
        torch.fft.fftn(torch.view_as_complex(image.contiguous()), dim=dims, norm=norm)
    )


def ifft_new(image: Tensor, ndim: int, normalized: bool = False) -> Tensor:
    """N-D IFFT for complex-as-real tensor with last dim size 2."""
    norm = "ortho" if normalized else None
    dims = tuple(range(-ndim, 0))
    return torch.view_as_real(
        torch.fft.ifftn(torch.view_as_complex(image.contiguous()), dim=dims, norm=norm)
    )


def roll(x: Tensor, shift: Union[int, Sequence[int]], dim: Union[int, Sequence[int]]) -> Tensor:
    """Similar to np.roll but applies to PyTorch Tensors."""
    if isinstance(shift, (tuple, list)):
        assert isinstance(dim, (tuple, list)) and len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x

    assert isinstance(dim, int)
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x: Tensor, dim: Optional[Union[int, Sequence[int]]] = None) -> Tensor:
    """Similar to np.fft.fftshift but applies to PyTorch Tensors."""
    if dim is None:
        dim_tuple = tuple(range(x.dim()))
        shift = [d // 2 for d in x.shape]
        return roll(x, shift, dim_tuple)

    if isinstance(dim, int):
        return roll(x, x.shape[dim] // 2, dim)

    shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x: Tensor, dim: Optional[Union[int, Sequence[int]]] = None) -> Tensor:
    """Similar to np.fft.ifftshift but applies to PyTorch Tensors."""
    if dim is None:
        dim_tuple = tuple(range(x.dim()))
        shift = [(d + 1) // 2 for d in x.shape]
        return roll(x, shift, dim_tuple)

    if isinstance(dim, int):
        return roll(x, (x.shape[dim] + 1) // 2, dim)

    shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def fft2(data: Tensor) -> Tensor:
    """
    Apply centered 2D FFT.
    Expects last dimension of size 2 (real/imag).
    """
    assert data.size(-1) == 2, "Expected complex-as-real with last dim == 2"
    data = ifftshift(data, dim=(-3, -2))
    data = fft_new(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data: Tensor) -> Tensor:
    """
    Apply centered 2D IFFT.
    Expects last dimension of size 2 (real/imag).
    """
    assert data.size(-1) == 2, "Expected complex-as-real with last dim == 2"
    data = ifftshift(data, dim=(-3, -2))
    data = ifft_new(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


# -----------------------------
# Complex arithmetic helpers
# -----------------------------

def complex_matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Multiply complex tensors a and b represented as real/imag channels.

    Supported shapes (per original code):
      - [2, H, W]
      - [B, 2, H, W]
      - [B, C, 2, H, W]
    """
    if len(a.size()) == 3:
        return torch.cat(
            ((a[0, ...] * b[0, ...] - a[1, ...] * b[1, ...]).unsqueeze(0),
             (a[0, ...] * b[1, ...] + a[1, ...] * b[0, ...]).unsqueeze(0)),
            dim=0
        )
    if len(a.size()) == 4:
        return torch.cat(
            ((a[:, 0, ...] * b[:, 0, ...] - a[:, 1, ...] * b[:, 1, ...]).unsqueeze(1),
             (a[:, 0, ...] * b[:, 1, ...] + a[:, 1, ...] * b[:, 0, ...]).unsqueeze(1)),
            dim=1
        )
    if len(a.size()) == 5:
        return torch.cat(
            ((a[:, :, 0, ...] * b[:, :, 0, ...] - a[:, :, 1, ...] * b[:, :, 1, ...]).unsqueeze(2),
             (a[:, :, 0, ...] * b[:, :, 1, ...] + a[:, :, 1, ...] * b[:, :, 0, ...]).unsqueeze(2)),
            dim=2
        )
    raise ValueError(f"Unsupported tensor rank for complex_matmul: {a.dim()}")


def complex_conj(a: Tensor) -> Tensor:
    """Complex conjugate for complex-as-real tensors (same supported ranks as above)."""
    if len(a.size()) == 3:
        return torch.cat((a[0, ...].unsqueeze(0), -a[1, ...].unsqueeze(0)), dim=0)
    if len(a.size()) == 4:
        return torch.cat((a[:, 0, ...].unsqueeze(1), -a[:, 1, ...].unsqueeze(1)), dim=1)
    if len(a.size()) == 5:
        return torch.cat((a[:, :, 0, ...].unsqueeze(2), -a[:, :, 1, ...].unsqueeze(2)), dim=2)
    raise ValueError(f"Unsupported tensor rank for complex_conj: {a.dim()}")


# -----------------------------
# (Optional) training utilities
# -----------------------------

def get_norm_layer(norm_type: str = "instance"):
    """Return a normalization layer constructor or None."""
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    if norm_type == "none":
        return None
    raise NotImplementedError(f"Normalization layer '{norm_type}' is not supported.")


def get_scheduler(optimizer, opt):
    """Learning-rate scheduler helper (kept for compatibility)."""
    if opt.lr_policy == "lambda":
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    else:
        raise NotImplementedError(f"Learning rate policy '{opt.lr_policy}' is not implemented.")
    return scheduler


def init_weights(net, init_type: str = "normal", gain: float = 0.02):
    """Initialize network weights (kept for compatibility)."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f"Initialization method '{init_type}' is not implemented.")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print(f"initialize network with {init_type}")
    net.apply(init_func)


def init_net(net, init_type: str = "normal", init_gain: float = 0.02, gpu_ids: Sequence[int] = ()):
    """Initialize a network (optional DataParallel support)."""
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, list(gpu_ids))
    init_weights(net, init_type, gain=init_gain)
    return net


# -----------------------------
# MRI operators (SENSE forward/adjoint)
# -----------------------------

class OPAT(nn.Module):
    """
    Adjoint SENSE operator (batched).

    Smap: [B, Nc, 2, H, W]
    k   : [B, Nc, 2, H, W]
    mask: [B, 2, H, W]
    out : [B, 2, H, W]
    """
    def __init__(self, Smap: Tensor):
        super().__init__()
        self.Smap = Smap

    def forward(self, k: Tensor, mask: Tensor) -> Tensor:
        batch_size, num_coil, _, height, width = self.Smap.size()
        mask = mask.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        k_under = k * mask
        im_u = ifft2(k_under.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        im = complex_matmul(im_u, complex_conj(self.Smap)).sum(1)
        return im


class OPAT2(nn.Module):
    """
    Adjoint SENSE operator (single example, non-batched).

    Smap : [Nc, 2, H, W]
    k    : [Nc, 2, H, W]
    mask : [2, H, W]
    out  : [2, H, W]
    """
    def __init__(self, Smap: Tensor):
        super().__init__()
        self.Smap = Smap

    def forward(self, k: Tensor, mask: Tensor) -> Tensor:
        num_coil, ch, height, width = self.Smap.size()
        mask = mask.repeat(num_coil, 1, 1, 1)
        k_under = k * mask
        im_u = ifft2(k_under.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # F^H y
        im = complex_matmul(im_u, complex_conj(self.Smap)).sum(0)       # sum_c S_c^H F^H y_c
        return im


class OPA(nn.Module):
    """
    Forward SENSE operator (batched).

    Smap: [B, Nc, 2, H, W]
    im  : [B, 2, H, W]
    mask: [B, 2, H, W]
    out : [B, Nc, 2, H, W]
    """
    def __init__(self, Smap: Tensor):
        super().__init__()
        self.Smap = Smap

    def forward(self, im: Tensor, mask: Tensor) -> Tensor:
        batch_size, num_coil, _, height, width = self.Smap.size()
        im = im.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        image_s = complex_matmul(im, self.Smap)
        k_full = fft2(image_s.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        mask = mask.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        return k_full * mask


class OPATA_plus_lambda_I(nn.Module):
    """
    Gram operator for multi-coil Cartesian MRI: (A^H A + λI)x.

    Smap: [B, Nc, 2, H, W]
    x   : [B, 2, H, W]
    mask: [B, 2, H, W]
    out : [B, 2, H, W]
    """
    def __init__(self, Smap: Tensor, lamda: Tensor):
        super().__init__()
        self.Smap = Smap
        self.lamda = lamda

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch_size, num_coil, _, height, width = self.Smap.size()

        mask_rep = mask.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)

        # Coil-wise multiply by sensitivity maps
        x_rep = x.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        sx = complex_matmul(x_rep, self.Smap)

        fsx = fft2(sx.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        mfsx = mask_rep * fsx
        fh_mfsx = ifft2(mfsx.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)

        ah_ax = complex_matmul(fh_mfsx, complex_conj(self.Smap)).sum(1)
        return ah_ax + self.lamda * x


# -----------------------------
# Conjugate Gradient (CG)
# -----------------------------

def cg_block(
    smap: Tensor,
    mask: Tensor,
    b0: Tensor,
    AHb: Tensor,
    lamda: Tensor,
    tol: Tensor,
    M: Optional[Tensor] = None,
    zn: Optional[Tensor] = None,
    max_iter: int = 50,
) -> Tensor:
    """
    Conjugate Gradient solver for (A^H A + λI)x = b0.

    Inputs
    ------
    zn   : optional initialization for CG (e.g., denoiser output)
    tol  : stopping threshold on ||r||
    max_iter : safety cap to avoid infinite loops

    Returns
    -------
    xk : approximate solution
    """
    ata = OPATA_plus_lambda_I(smap, lamda)

    xk = torch.zeros_like(b0)
    if zn is not None:
        xk = zn

    rk = b0 - ata(xk, mask)
    pk = rk.clone()

    num_loop = 0
    # Use .item() for scalar comparisons; keep behavior similar, but safer/cleaner.
    while torch.norm(rk).item() > float(tol.item()) and num_loop < max_iter:
        rktrk = torch.pow(torch.norm(rk), 2)

        # pktapk = <p, A p> (complex inner product form kept from original code)
        pktapk = torch.sum(complex_matmul(complex_conj(pk), ata(pk, mask)))

        alpha = rktrk / pktapk
        xk = xk + alpha * pk
        rk1 = rk - alpha * ata(pk, mask)

        rk1trk1 = torch.pow(torch.norm(rk1), 2)
        beta = rk1trk1 / rktrk
        pk = rk1 + beta * pk
        rk = rk1
        num_loop += 1

    return xk


class CG(torch.autograd.Function):
    """
    Differentiable CG layer for MoDL.

    Solves: (A^H A + λI)^{-1} (A^H b + λ z_n) via cg_block.
    """
    @staticmethod
    def forward(ctx, zn, tol, lamda, smap, mask, AHb, device):
        tol_t = torch.tensor(tol, device=zn.device, dtype=zn.dtype)
        lamda_t = torch.tensor(lamda, device=zn.device, dtype=zn.dtype)

        ctx.save_for_backward(tol_t, lamda_t, smap, mask, AHb)

        b0 = AHb + lamda_t * zn
        return cg_block(smap, mask, b0, AHb, lamda_t, tol_t, zn=zn)

    @staticmethod
    def backward(ctx, dx):
        tol, lamda, smap, mask, AHb = ctx.saved_tensors
        # Gradient w.r.t. zn is λ * (A^H A + λI)^{-1} dx
        return lamda * cg_block(smap, mask, dx, AHb, lamda, tol), None, None, None, None, None, None


def CG_fn(output, tol, lamda, mps, mask, aliased_image, device):
    """Convenience wrapper matching original API."""
    return CG.apply(output
