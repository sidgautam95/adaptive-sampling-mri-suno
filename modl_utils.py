import torch
import numpy as np
from torch import Tensor
from modl_cg_functions import *
from utils import *
import modl_cg_functions

def fft_new(image: Tensor, ndim: int, normalized: bool = False) -> Tensor:
    norm = "ortho" if normalized else None
    dims = tuple(range(-ndim, 0))

    image = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(image.contiguous()), dim=dims, norm=norm
        )
    )

    return image


def ifft_new(image: Tensor, ndim: int, normalized: bool = False) -> Tensor:
    norm = "ortho" if normalized else None
    dims = tuple(range(-ndim, 0))
    image = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(image.contiguous()), dim=dims, norm=norm
        )
    )

    return image

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)

def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)

def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)



def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = fft_new(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = ifft_new(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def complex_matmul(a, b):
    # function to multiply two complex variable in pytorch, the real/imag channel are in the third last two channels ((batch), (coil), 2, height, width).
    if len(a.size()) == 3:
        return torch.cat(((a[0, ...] * b[0, ...] - a[1, ...] * b[1, ...]).unsqueeze(0),
                          (a[0, ...] * b[1, ...] + a[1, ...] * b[0, ...]).unsqueeze(0)), dim=0)
    if len(a.size()) == 4:
        return torch.cat(((a[:, 0, ...] * b[:, 0, ...] - a[:, 1, ...] * b[:, 1, ...]).unsqueeze(1),
                          (a[:, 0, ...] * b[:, 1, ...] + a[:, 1, ...] * b[:, 0, ...]).unsqueeze(1)), dim=1)
    if len(a.size()) == 5:
        return torch.cat(((a[:, :, 0, ...] * b[:, :, 0, ...] - a[:, :, 1, ...] * b[:, :, 1, ...]).unsqueeze(2),
                          (a[:, :, 0, ...] * b[:, :, 1, ...] + a[:, :, 1, ...] * b[:, :, 0, ...]).unsqueeze(2)), dim=2)


def complex_conj(a):
    # function to multiply two complex variable in pytorch, the real/imag channel are in the last two channels.
    if len(a.size()) == 3:
        return torch.cat((a[0, ...].unsqueeze(0), -a[1, ...].unsqueeze(0)), dim=0)
    if len(a.size()) == 4:
        return torch.cat((a[:, 0, ...].unsqueeze(1), -a[:, 1, ...].unsqueeze(1)), dim=1)
    if len(a.size()) == 5:
        return torch.cat((a[:, :, 0, ...].unsqueeze(2), -a[:, :, 1, ...].unsqueeze(2)), dim=2)


def CG_fn(output, tol ,lamda, smap, mask, aliased_image, device):
    return CG.apply(output, tol ,lamda, smap, mask, aliased_image,device)

def preprocess_data_for_modl(kspace,maps,mask):
    
    # kspace: shape: no. of coils x height x width
    # maps: shape: no. of coils x height x width
    # mask: shape: height x width

    ncoils, height, width = kspace.shape
    nchannels = 2

    if torch.is_tensor(kspace):
        kspace = kspace.cpu().detach().numpy()
    if torch.is_tensor(maps):
        maps = maps.cpu().detach().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().detach().numpy()

    # Reshaping mask to make it same dimension as kspace
    if mask.shape[0]!=height or mask.shape[1]!=width:
        mask=crop_pad_mask(mask,height,width)

    # plt.figure()
    # plt.imshow(mask,cmap='gray')
    # plt.savefig('mask.png')

    # Normalizing the kspace and senstivity maps by maximum value
    # Dividing into real and imaginary part
    k_r = np.real(kspace)/abs(kspace).max()
    k_i = np.imag(kspace)/abs(kspace).max()

    s_r = np.real(maps)/abs(maps).max()
    s_i = np.imag(maps)/abs(maps).max()

    ncoil, nx, ny = s_r.shape
   
    # Making two channel image from real and imaginary part of kspace, maps and mask
    k_np = np.stack((k_r, k_i), axis=0) # shape: nchannels x ncoils x nx x ny
    s_np = np.stack((s_r,s_i), axis=0) # shape: nchannels x ncoils x nx x ny

    mask = torch.tensor(np.repeat(mask[np.newaxis], 2, axis=0),dtype=torch.float32) # shape: nchannels x nx x ny

    A_k = torch.tensor(k_np, dtype=torch.float32).permute(1, 0, 2, 3) # kspace tensor

    A_I = ifft2(A_k.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # F^H y, shape: ncoils x nchannels x nx x ny

    # Conversting to torch tensor
    maps_tensor = torch.tensor(s_np, dtype=torch.float32).permute(1, 0, 2, 3) # Maps, shape: ncoils x nchannels x height x width

    sense_recon = torch.sum(complex_matmul(A_I, complex_conj(maps_tensor)),dim=0) # S^H F^H y, shape: nchannels x height x width

    A_I = A_I/torch.max(torch.abs(sense_recon)[:]) # F^H y/|S^HF^Hy|_max

    A_k = fft2(A_I.permute(0,2,3,1)).permute(0,3,1,2) # F F^H y, kspace after taking FFT of normalized image: ncoils x nchannels x height x width

    AT = modl_cg_functions.OPAT2(maps_tensor) # Initialize function for SENSE Reconstruction S^H F^H y

    img = AT(A_k, mask) # Adjoint Reconstructed image: S^H F^H M y, shape: nchannels x height x width

    return img, mask, maps_tensor




def modl_recon_batched(ksp,mps,masks, model, tol = 0.00001, lamda = 1e2, num_iter = 6, dataset='fastmri', device=torch.device('cpu')):

    # Function runs MoDL reconstruction on set of masks. Specifically used for ICD sampling algorithm
    # Inputs
    # ksp: Shape - nCoils x Height x Width
    # mps: Sensitivity Maps - nCoils x Height x Width
    # mask: Mask - Batchsize x Height x Width
    
    # Works for both batchsize 1 and multiple batches input
    # if input is only one slice/no. of dimension = 3, increase dimension by 1.

    #if only one mask is passed as an argument as a 2D array increase dimension by one and make it 1xheightxwidth
    
    if masks.ndim==2:
         masks=(masks).unsqueeze(0)

    ncoils, height, width = ksp.shape
    nmasks=masks.shape[0]

    if dataset=='fastmri':
        height=320
        width=320
    else:
        height = ksp.shape[1]
        width = ksp.shape[2]
    
    img_recon = torch.zeros((nmasks,height,width),dtype=torch.complex64).to(device)

    for count,mask in enumerate(masks):
        with torch.no_grad(): # Gradient computation not required
            img_recon[count] = modl_recon(ksp, mps, mask, model, tol, lamda, device=device) # Doing MoDL reconstruction with two channel

    return torch.squeeze(img_recon)



def modl_recon(ksp,mps,mask, model, tol = 0.00001, lamda = 1e2, num_iter = 6, dataset='fastmri', crop=True, device=torch.device('cpu')):

    ncoils, height, width = ksp.shape
    
    nchannels = 2

    if dataset=='fastmri':
         tol=1e-5
         lamda=1e2
    elif dataset=='ocmr':
        tol = 1e-7
        lamda=1e-2

    img_recon = torch.zeros((height,width),dtype=torch.complex64).to(device)

    with torch.no_grad(): # Gradient computation not required

        img_aliased, mask, maps = preprocess_data_for_modl(ksp,mps,mask) # converting to two channel and computing aliased image

        img_recon_modl = modl_recon_training(img_aliased, mask, maps, model, tol, lamda,device=device) # Doing MoDL reconstruction with two channel

        img_recon_modl = two_channel_to_complex(img_recon_modl)

        if dataset=='fastmri':
            img_recon_modl=crop_img(img_recon_modl)

        img_recon_modl = img_recon_modl/abs(img_recon_modl).max()

    return img_recon_modl

def unet_recon_batched_torch(ksp,mps,masks, model, dataset='fastmri', batch_size=16, device=torch.device('cpu')):

    # Function runs UNET reconstruction on batches of images given kspace, sensitivity maps and mask.
    # Inputs
    # ksp: kspace - Batchsize x nCoils x Height x Width
    # mps: Sensitivity Maps - Batchsize x nCoils x Height x Width
    # mask: Mask - Batchsize x Height x Width
    
    # Works for both batchsize 1 and multiple batches input
    # if input is only one slice/no. of dimension = 3, increase dimension by 1.

    # if batchsize=1, ksp_dim = 3 
    # if batchsize>1, ksp_dim = 4
    #if only one mask is passed as an argument as a 2D array increase dimension by one and make it 1xheightxwidth
    
    if torch.is_tensor(masks):
        masks=masks.cpu().detach().numpy()
        
    if masks.ndim==2:
         masks=np.expand_dims(masks,axis=0)

    nImages, height, width = masks.shape
    nCoils = ksp.shape[0]

    nchannels = 2

    img_aliased = np.zeros((len(masks),nchannels,height,width))

    for count,mask in enumerate(masks):
        with torch.no_grad(): # Gradient computation not required
            img_aliased[count], _,_ = preprocess_data_for_modl(ksp,mps,mask)

    if dataset=='fastmri':
        img_aliased = crop_img(img_aliased)
        height=320
        width=320
    
    img_recon = torch.zeros((len(masks),height,width),dtype=torch.complex64).to(device)

    # Iterating over batches
    for batch in np.array_split(np.arange(nImages), nImages//batch_size+1):
        with torch.no_grad(): # Gradient computation not required
            output = model(torch.tensor(img_aliased[batch]).float().to(device))

            img_recon[batch] = (output[:,0]+1j*output[:,1]).to(device)

    for i in range(len(masks)):
        img_recon[i] = img_recon[i]/abs(img_recon[i]).max()

    return torch.squeeze(img_recon)


    # Iterating over batches
    for batch in np.array_split(np.arange(nImages), nImages//batch_size+1):
        with torch.no_grad(): # Gradient computation not required
            # Network Input: batches of real valued sense reconstructions
            # Input shape: batch_size x number of channels x height x width
            inputs=torch.zeros((img_recon_sense[batch].shape[0],nchannels,height,width)).to(device)
            
            if nchannels==1:
                inputs=torch.tensor(abs(img_recon_sense[batch])).to(device)
            else:
                inputs[:,0] = torch.real(torch.tensor(img_recon_sense[batch])).to(device)
                inputs[:,1] = torch.imag(torch.tensor(img_recon_sense[batch])).to(device)

            # print(device)
            inputs = inputs/abs(inputs).max()
            # Applying UNET
            outputs=net(inputs.float()) # UNET output

        if nchannels==2:
            img_recon_net[batch] = (outputs[:,0]+1j*outputs[:,1]).cpu().detach().numpy()



def modl_recon_training(img_aliased, mask, smap, model, tol=1e-5, lamda=1e2, num_iter=6, device=torch.device('cpu'),print_loss=False):

    if torch.is_tensor(img_aliased) == False:
        img_aliased = torch.tensor(img_aliased)
    if torch.is_tensor(smap) == False:
        smap = torch.tensor(smap)
    if torch.is_tensor(mask) == False:
        mask = torch.tensor(mask)

    # img_gt = img_gt.to(device).float().unsqueeze(0)
    img_aliased = img_aliased.to(device).float().unsqueeze(0)
    smap = smap.to(device).float().unsqueeze(0)
    mask = mask.to(device).float().unsqueeze(0)

    net_input = torch.clone(img_aliased)

    for ii in range(num_iter):

        # loss_previous_iter = torch_nrmse(img_gt,net_input)
        # net_input = net_input/abs(net_input).max()
        net_output = model(net_input)
        # net_output = net_output/abs(net_output).max()
                    
        cg_output = CG_fn(net_output, tol = tol ,lamda = lamda, smap = smap, mask = mask, aliased_image = img_aliased, device = device)
        net_input = torch.clone(cg_output)

        # loss_current_iter = torch_nrmse(img_gt,cg_output)
        # if print_loss:
        #     print('Iteration:',ii+1,'Loss',loss_current_iter)

        # if loss_current_iter>loss_previous_iter:
        #     if print_loss:
        #         print('loss increasing. exiting after',ii+1,'iterations')
        #     break

    final_output = torch.clone(net_input)

    return final_output

