# Helper functions
import numpy as np
import torch
from torch import Tensor
from modl_cg_functions import *

def make_vdrs_mask(height,width,nlines,init_lines):
    
    mask_vdrs=np.zeros((height,width),dtype='bool')
    
    # settint the low frequencies to true
    low1=(width-init_lines)//2
    low2=(width+init_lines)//2
    mask_vdrs[:,low1:low2]=True
    
    nlinesout=(nlines-init_lines)//2
    rng = np.random.default_rng()
    t1 = rng.choice(low1-1, size=nlinesout+1, replace=False)
    t2 = rng.choice(np.arange(low2+1, width), size=nlinesout, replace=False)
    mask_vdrs[:,t1]=True 
    mask_vdrs[:,t2]=True
    
    return mask_vdrs

def make_lf_mask(height,width,nlines):
    mask_lf=np.zeros((height,width),dtype='bool');
    mask_lf[:,(width-nlines)//2:(width+nlines)//2]=True
    return mask_lf


def crop_img(img,height=320,width=320):
    # possible shapes of img: 
    # 1. height x width
    # 2. batch size x height x width
    # 3. batch size x no. of channels x height x width

    dim=img.ndim

    if dim==2:
        img_cropped = img[(img.shape[0]-height)//2:(img.shape[0]+height)//2, (img.shape[1]-width)//2:(img.shape[1]+width)//2]
    elif dim==3:
        img_cropped = img[:,(img.shape[1]-height)//2:(img.shape[1]+height)//2, (img.shape[2]-width)//2:(img.shape[2]+width)//2]
    elif dim==4:
        img_cropped =  img[:,:,(img.shape[2]-height)//2:(img.shape[2]+height)//2, (img.shape[3]-width)//2:(img.shape[3]+width)//2]
    else:
        print('Wrong dimension:',img.shape)
    
    return img_cropped


def compute_nrmse_numpy(img_gt,img_aliased): # Normalized Root Mean Squared Error
    return np.linalg.norm(img_gt-img_aliased)/np.linalg.norm(img_gt)


def two_channel_to_complex(img_2channel):
    # Input: 
    # Img of shape: 2 x height x width
    # Returns two channel converted to complex
    if torch.is_tensor(img_2channel):
        img_2channel = torch.squeeze(img_2channel)
    else:
        img_2channel = np.squeeze(img_2channel)
        
    return img_2channel[0] +1j * img_2channel[1]


def compute_nrmse(img_gt,img_aliased):
    return torch.linalg.norm(img_gt-img_aliased)/torch.linalg.norm(img_gt)

def compute_nmae(img_gt,img_aliased):
    return torch.mean(torch.abs(img_gt-img_aliased))/torch.mean(torch.abs(img_gt))


def compute_loss(img_gt,img_recon,alpha1=1,alpha2=0,alpha3=0,alpha4=0): 
    
    # ## Compute reconstruction quality in terms of linear combination of various metrics

    if alpha1!=0:# Normalized Root Mean Squared Error
        nrmse=compute_nrmse(img_gt,img_recon)
    else:
        nrmse=0

    if alpha2!=0:# SSIM: Structural Similarity Index
        img_gt = img_gt.cpu().detach().numpy()
        img_recon = img_recon.cpu().detach().numpy()
        ssim_recon = torch.tensor(ssim(abs(img_gt), abs(img_recon)))
    else:
        ssim_recon=0

    if alpha3!=0:#NMAE: normalized mean absolute error
        nmae = compute_nmae(img_gt,img_recon)
    else:
        nmae=0

    if alpha4!=0:#HFEN: high frequency error norm
        hfen = compute_hfen(img_gt.cpu().detach().numpy(),img_recon.cpu().detach().numpy())
        hfen = torch.tensor(hfen)
    else:
        hfen=0

    return (alpha1*nrmse+alpha2*(1-ssim_recon)+alpha3*nmae+alpha4*hfen)/(alpha1+alpha2+alpha3+alpha4)



def loss_fn(img_input,img_output):
    # img_input: batch_size x nchannels x height x width
    nrmse=torch.zeros((img_input.shape[0]))
    # loop over batch
    for i in range(img_input.shape[0]):
        nrmse[i]=torch.linalg.norm(img_input[i]-img_output[i])/\
        torch.linalg.norm(img_input[i])
    return torch.mean(nrmse)


def plot_mr_image(img,title='',Vmax=0.7,colorbar=False,normalize=True,crop=True):

    if torch.is_tensor(img):
        img=img.cpu().detach().numpy()

    if normalize: # normalize image
        img = img/abs(img).max()

    img=np.squeeze(img) # remove batch dimension

    if img.shape[0]==2: # if it is a two channel image, convert to complex
        img=two_channel_to_complex(img)

    if crop: # crop to central 320 x 320
        img=crop_img(img)

    img = np.flipud(abs(img)) # flip up-down for knee MR image

    plt.imshow(abs(img),cmap='gray',vmax=Vmax) # flip left right for cardiac MR image

    plt.axis('off')
    plt.title(title)

    if colorbar:
        plt.colorbar()

def preprocess_data(ksp,mps,mask,device=torch.device('cpu')):
    
    # kspace: shape: no. of coils x height x width
    # maps: shape: no. of coils x height x width
    # mask: shape: height x width

    ncoils, height, width = ksp.shape

    if torch.is_tensor(ksp)==False:
        ksp=torch.tensor(ksp).to(device)
    if torch.is_tensor(mps)==False:
        mps=torch.tensor(mps).to(device)
    if torch.is_tensor(mask)==False:
        mask=torch.tensor(mask).to(device)


    # Normalizing the kspace and sensitivity maps by the maximum value
    # Separating into real and imaginary part
    k_r = torch.real(ksp)/abs(ksp).max()
    k_i = torch.imag(ksp)/abs(ksp).max()

    s_r = torch.real(mps)/abs(mps).max()
    s_i = torch.imag(mps)/abs(mps).max()

    ncoil, nx, ny = s_r.shape
   
    # Making two-channel images from the real and imaginary parts of kspace, map,s and mask
    k_np = torch.stack((k_r, k_i), axis=0) # shape: nchannels x ncoils x nx x ny
    s_np = torch.stack((s_r,s_i), axis=0) # shape: nchannels x ncoils x nx x ny

    mask = mask.unsqueeze(0).repeat(2,1,1) # shape: nchannels x nx x ny

    A_k = k_np.permute(1, 0, 2, 3) # kspace tensor

    A_I = ifft2(A_k.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # F^H y, shape: ncoils x nchannels x nx x ny

    mps_tensor = s_np.permute(1, 0, 2, 3) # Maps, shape: ncoils x nchannels x height x width

    adjoint_recon = torch.sum(complex_matmul(A_I, complex_conj(mps_tensor)),dim=0) # S^H F^H y, shape: nchannels x height x width

    A_I = A_I/torch.max(torch.abs(adjoint_recon)[:]) # F^H y/|S^HF^Hy|_max

    A_k = fft2(A_I.permute(0,2,3,1)).permute(0,3,1,2) # F F^H y, kspace after taking FFT of normalized image: ncoils x nchannels x height x width

    AT = OPAT2(mps_tensor) # Initialize function for SENSE Reconstruction S^H F^H y

    img = AT(A_k, mask) # Adjoint Reconstructed image: S^H F^H M y, shape: nchannels x height x width

    return img, mask, mps_tensor

def unet_recon_batched(ksp,mps,masks, model, batch_size=4, device=torch.device('cpu')):

    # Function runs UNET reconstruction on batches of images given kspace, sensitivity maps, and mask.
    # Inputs
    # ksp: kspace - Batchsize x nCoils x height x width
    # mps: Sensitivity Maps - Batchsize x nCoils x height x width
    # mask: Mask - Batchsize x height x width
    
    # Works for both batch-size 1 and multiple batch size
    # if input is only one slice/no. of dimension = 3, increase dimension by 1.

    # if batchsize=1, ksp_dim = 3 
    # if batchsize>1, ksp_dim = 4
    #if only one mask is passed as an argument as a 2D array increase the dimension by one and make it 1xheightxwidth

    if torch.is_tensor(masks)==False:
        masks=torch.tensor(masks).to(device)

    if masks.ndim==2:
         masks=masks.unsqueeze(0)

    nImages, height, width = masks.shape
    nCoils = ksp.shape[0]

    nchannels = 2

    img_aliased = torch.zeros((len(masks),nchannels,height,width))

    for count,mask in enumerate(masks):
        with torch.no_grad(): # Gradient computation not required
            img_aliased[count], _,_ = preprocess_data(ksp,mps,mask,device)

    img_height=320
    img_width=320
    
    img_recon = torch.zeros((len(masks),img_height,img_width),dtype=torch.complex64).to(device)

    # Iterating over batches
    for batch in np.array_split(np.arange(nImages), nImages//batch_size+1):
        with torch.no_grad(): # Gradient computation not required

            output = model(crop_img(img_aliased[batch].to(device))).float()
            img_recon[batch] = (output[:,0] + 1j*output[:,1]).to(device)

    for i in range(len(masks)):
        img_recon[i] = img_recon[i]/abs(img_recon[i]).max()

    return torch.squeeze(img_recon)



def modl_recon_training(img_aliased, mask, mps, model, tol=1e-5, lamda=1e2, num_iter=6, device=torch.device('cpu'), print_loss=False):

    # Function runs MoDL reconstruction
    # Input
    # Aliased Image: Shape - nchannels x height x width
    # mps: Sensitivity Maps - nchannels x nCoils x height x width
    # mask: Mask - nchannels x height x width
    
    # unsqueezing the tensors to add batch dimensions
    img_aliased = torch.tensor(img_aliased).to(device).float().unsqueeze(0)
    mps = torch.tensor(mps).to(device).float().unsqueeze(0)
    mask = torch.tensor(mask).to(device).float().unsqueeze(0)

    net_input = torch.clone(img_aliased)

    for i in range(num_iter): # MoDL unrolling

            net_output = model(net_input)
            cg_output = CG_fn(net_output, tol = tol ,lamda = lamda, mps = mps, mask = mask, aliased_image = img_aliased, device = device)
            net_input = torch.clone(cg_output)

    img_recon_modl = torch.clone(net_input)

    return img_recon_modl



def modl_recon(ksp,mps,mask, model, tol = 0.00001, lamda = 1e2, num_iter = 6, crop=True, device=torch.device('cpu')):

    ncoils, height, width = ksp.shape
    
    nchannels = 2

    img_recon = torch.zeros((height,width),dtype=torch.complex64).to(device)

    with torch.no_grad(): # Gradient computation not required

        img_aliased, mask, mps = preprocess_data(ksp,mps,mask) # converting to two channel and computing aliased image

        # img_gt = img_gt.to(device).float().unsqueeze(0)
        img_aliased = img_aliased.to(device).float().unsqueeze(0)
        mps = mps.to(device).float().unsqueeze(0)
        mask = mask.to(device).float().unsqueeze(0)

        net_input = torch.clone(img_aliased)

        for i in range(num_iter): # MoDL unrolling

            net_output = model(net_input)
            cg_output = CG_fn(net_output, tol = tol ,lamda = lamda, mps = mps, mask = mask, aliased_image = img_aliased, device = device)
            net_input = torch.clone(cg_output)

        img_recon_modl = torch.clone(net_input)

        img_recon_modl = two_channel_to_complex(img_recon_modl) # converting two channel to complex

        img_recon_modl=crop_img(img_recon_modl) # cropping image to region of interest: central 320 x 320 portion

        img_recon_modl = img_recon_modl/abs(img_recon_modl).max() # normalizing image

    return img_recon_modl


def modl_recon_batched(ksp, mps, masks, model, tol = 0.00001, lamda = 1e2, num_iter = 6, device = torch.device('cpu')):

    # Function runs MoDL reconstruction on the set of masks.
    # Inputs
    # ksp: Shape - nCoils x height x width
    # mps: Sensitivity Maps - nCoils x height x width
    # mask: Mask - Batchsize x height x width
    
    # Works for both batch size 1 and multiple batch input
    # if input is only one slice/no. of dimension = 3, increase dimension by 1.

    #if only one mask is passed as an argument as a 2D array increase the dimension by one and make it 1xheightxwidth
    
    if masks.ndim==2:
         masks=(masks).unsqueeze(0)

    ncoils, height, width = ksp.shape
    nmasks=masks.shape[0]

    height=320
    width=320

    img_recon = torch.zeros((nmasks,height,width),dtype=torch.complex64).to(device)

    for count, mask in enumerate(masks):
        with torch.no_grad(): # Gradient computation not required
            img_recon[count] = modl_recon(ksp, mps, mask, model, tol, lamda, device=device) # Doing MoDL reconstruction

    return torch.squeeze(img_recon)
