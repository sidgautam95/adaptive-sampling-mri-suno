# MoDL-CG Implementation from https://github.com/JeffFessler/BLIPSrecon/tree/main

import torch
import torch.nn as nn
from torch.nn import init
import matplotlib.pyplot as plt
import functools
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
from utils import *
import scipy.linalg

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch_size':
        norm_layer = functools.partial(nn.batch_sizeNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('batch_sizeNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net

class OPAT(nn.Module):
    # Adjoint sense operator
    # Initialize: Sensitivity maps: [batch_size, num_coils, num_channels, height, width]
    # Input: K: [batch_size, num_coils, num_channels, height, width]
    #        Mask: [batch_size, num_channels, height, width]
    # Output: Image: [batch_size, num_channels, height, width]
    def __init__(self, Smap):
        super(OPAT, self).__init__()
        self.Smap = Smap

    def forward(self, k, mask):
        batch_size, num_coil, _, height, width = self.Smap.size()
        mask = mask.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        k_under = k * mask
        im_u = ifft2((k_under).permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        im = complex_matmul(im_u, complex_conj(self.Smap)).sum(1)
        return im
        
class OPAT2(nn.Module):
    # Adjoint sense operator
    # Initialize: Sensitivity maps: [num_coils, num_channels, height, width]
    # Input: kspace: [num_coils, num_channels, height, width]
    #        Mask: [num_channels, height, width]
    # Output: Image: [num_channels, height, width]
    def __init__(self, Smap):
        super(OPAT2, self).__init__()
        self.Smap = Smap

    def forward(self, k, mask):
        num_coil,ch,height, width = self.Smap.size()
        mask = mask.repeat(num_coil, 1, 1, 1)#.permute(1, 0, 2, 3)
        k_under = k * mask
        im_u = ifft2((k_under).permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # F^H y
        im = complex_matmul(im_u, complex_conj(self.Smap)).sum(0) # SENSE Reconstruction sum_{c=1}^nc S_c^H F_H y_c
        return im

class OPA(nn.Module):
    # Sense operator
    # Initialize: Sensitivity maps: [batch_size, num_coils, num_channels, height, width]
    # Input: Image: [batch_size, num_channels, height, width]
    #        Mask: [batch_size, num_channels, height, width]
    # Return: K: [batch_size, num_coils, num_channels, height, width]
    def __init__(self, Smap):
        super(OPA, self).__init__()
        self.Smap = Smap

    def forward(self, im, mask):
        batch_size, num_coil, _, height, width = self.Smap.size()
        im = im.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        Image_s = complex_matmul(im, self.Smap)
        k_full = fft2(Image_s.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        mask = mask.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)

        k_under = k_full * mask
        return k_under


class OPATA_plus_lambda_I(nn.Module):
    # Gram operator for multi-coil Cartesian MRI
    # Implements (A^HA + \lambda I)x = (A^H A +lambda I)x = A^H Ax +lambda x
    # Initialize with sensitivity maps: [batch_size, num_coils, num_channels, height, width]
    # Input: Denoiser output x, shape: [batch_size, num_channels, height, width]
    #        Undersampling mask mask, shape: [batch_size, num_channels, height, width]
    # Returns: (A^HA +lambda I)x: [batch_size, num_channels, height, width]
    def __init__(self, Smap, lamda):
        super(OPATA_plus_lambda_I, self).__init__()
        self.Smap = Smap
        self.lamda = lamda

    def forward(self, x, mask):
        batch_size, num_coil, _, height, width = self.Smap.size()
        mask = mask.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4) # reshaping mask into same shape as image x
        Sx = complex_matmul(x.repeat(num_coil, 1, 1, 1, 1).permute(1, 0, 2, 3, 4), self.Smap) # Sx: coil wise ifft image weighted by adjoint of sensitivity maps
        FSx = fft2(Sx.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        MFSx = mask*FSx # Ax = MFSx
        FHMFSx = ifft2((MFSx).permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3) #F^H MFSx = F^H Ax
        AHAx = complex_matmul(FHMFSx, complex_conj(self.Smap)).sum(1) # S^H (F^H M FSx) = A^H A x: applying adjoint of forward MRI operator on Ax
           
        return AHAx + self.lamda * x


class CG(torch.autograd.Function):
    # Modified solver for (A^H A+\lambda I)^{-1} (\lambda A^H b + zn)
    # For reference, see: https://ieeexplore.ieee.org/document/8434321
    # Input: zn (denoised image from network (previous step) - UNet or DIDN) shape: [batch_size, num_channels, height, width]
    #        AHb (adjoint of forward MRI operator applied on undersampled kspace): [batch_size, num_channels, height, width]
    #        mask (undersampling mask): [batch_size, num_channels, height, width]
    #        smap (sensitivity maps): [batch_size, ncoil, num_channels, height, width]
    #        tol: exiting threshold
    # Returns: Image: [batch_size, num_channels, height, width]
    @staticmethod
    def forward(ctx, zn, tol, lamda, smap, mask, AHb, device):
        tol = torch.tensor(tol).to(zn.device, dtype=zn.dtype)
        lamda = torch.tensor(lamda).to(zn.device, dtype=zn.dtype)
        ctx.save_for_backward(tol, lamda, smap, mask, AHb)
        b0 = AHb + lamda*zn # right most term in eq. 11 of MoDL paper
        return cg_block(smap, mask, b0, AHb, lamda, tol, zn=zn)

    @staticmethod
    def backward(ctx, dx):
        tol, lamda, smap, mask, AHb = ctx.saved_tensors
        return lamda * cg_block(smap, mask, dx, AHb, lamda, tol), None, None, None, None, None, None


def cg_block(smap, mask, b0, AHb, lamda, tol, M=None, zn=None):
    # A specified conjugated gradient block for MR system matrix A
    # Input: zn (denoised image from network (previous step) - UNet or DIDN) shape: [batch_size, num_channels, height, width]
    #        AHb (adjoint of forward MRI operator applied on undersampled kspace): [batch_size, num_channels, height, width]
    #        mask (undersampling mask): [batch_size, num_channels, height, width]
    #        smap (sensitivity maps): [batch_size, ncoil, num_channels, height, width]
    #        tol: exiting threshold
    #        b0:  AHb + lamda*zn # right most term in eq. 11 of MoDL paper
    # Returns: Image: [batch_size, num_channels, height, width]

    ATA = OPATA_plus_lambda_I(smap, lamda) # initializing the function A^H A (Gram operator)

    x0 = torch.zeros_like(b0) # initializing

    if zn is not None:
        x0 = zn # if denoised output provided, use that as initialization for CG

    mask
    num_loop = 0
    r0 = b0 - ATA(x0, mask)
    p0 = r0 
    rk = r0
    pk = p0
    xk = x0 

    while torch.norm(rk).data.cpu().numpy().tolist() > tol:
        # print('Residual =',torch.norm(rk).data.cpu().numpy().tolist(),' Tolerance =',tol)
        rktrk = torch.pow(torch.norm(rk), 2) 
        pktapk = torch.sum(complex_matmul(complex_conj(pk), ATA(pk, mask))) 
        alpha = rktrk / pktapk 
        xk1 = xk + alpha * pk
        rk1 = rk - alpha * ATA(pk, mask)
        rk1trk1 = torch.pow(torch.norm(rk1), 2)
        beta = rk1trk1 / rktrk
        pk1 = rk1 + beta * pk
        xk = xk1
        rk = rk1
        pk = pk1
        num_loop = num_loop + 1
    # print('Finished cg iteration after',num_loop+1,'iterations') 
    return xk