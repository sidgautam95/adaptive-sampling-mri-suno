import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import sigpy
import scipy
import sklearn
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_laplace
import scipy
from hfen import *

def make_fig(img,Vmax=None,gamma=0.5):
    plt.imshow(np.fliplr(abs(img)**gamma),cmap='gray',vmax=Vmax); plt.axis('off');
    
def make_video(img):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    fig.canvas.draw()

    for i in range(0,img.shape[2]):
        ax.clear()
        ax.imshow(np.abs(img[:,:,i]),cmap='gray')
        ax.set_title('Frame:'+str(i+1));ax.axis('off')
        time.sleep(0.5)
        fig.canvas.draw()
        
def MRI_FFT(im):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(im)))

def MRI_IFFT(y):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(y)))

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


# ## Error Metrics: Computing image reconstruction quality
def compute_nrmse(img_gt,img_aliased): # Normalized Root Mean Squared Error
    return np.linalg.norm(img_gt-img_aliased)/np.linalg.norm(img_gt)

def compute_nmse(img_gt,img_aliased): # Normalized Mean Squared Error
    return np.mean(np.abs(img_gt-img_aliased)**2)/np.mean(np.abs(img_gt)**2)

def compute_mse(img_gt,img_aliased): # Mean Squared Error
    return np.mean(np.abs(img_gt-img_aliased)**2)

def compute_psnr(img_gt,img_aliased):
    mse = np.mean(np.abs(img_gt-img_aliased)**2) # mean squared error
    psnr = 10*np.log10((abs(img_gt).max())**2/mse)
    return psnr

def compute_ssim(img_gt,img_aliased):
    return ssim(abs(img_gt), abs(img_aliased))

def compute_hfen(img_gt,img_aliased,sigma=1.5):
    
    # LoG - Laplacian of Gaussian
    LoG_GT = gaussian_laplace(img_gt, sigma)
    LoG_recon = gaussian_laplace(img_aliased, sigma)

    return np.linalg.norm(LoG_recon-LoG_GT)/np.linalg.norm(LoG_GT)

# def compute_hfen(gt,recon):

#     if gt.dtype =='complex':
#         nchannels = 2
#     else:
#         nchannels = 1

#     gt = torch.tensor(gt)
#     recon = torch.tensor(recon)

#     batch_size = 1
#     height,width = gt.shape

#     img1 = torch.zeros((batch_size,nchannels,height,width))
#     img2 = torch.zeros((batch_size,nchannels,height,width))

#     if nchannels==2:
#         img1[0,0] = torch.real(gt)
#         img1[0,1] = torch.imag(gt)
#         img2[0,0] = torch.real(recon)
#         img2[0,1] = torch.imag(recon)
#     else:
#         img1 = gt.unsqueeze(0).unsqueeze(0)
#         img2 = recon.unsqueeze(0).unsqueeze(0)

#     return hfen(img1,img2)


# ## Error Metrics: Computing image reconstruction quality
def compute_metric_torch(img_gt,img_recon,alpha1=1,alpha2=0,alpha3=0,alpha4=0): 
    
    if alpha1!=0:# Normalized Root Mean Squared Error
        nrmse=torch_nrmse(img_gt,img_recon)
    else:
        nrmse=0

    if alpha2!=0:# SSIM: Structural Similarity Index
        img_gt = img_gt.cpu().detach().numpy()
        img_recon = img_recon.cpu().detach().numpy()
        ssim_recon = torch.tensor(ssim(abs(img_gt), abs(img_recon)))
    else:
        ssim_recon=0

    if alpha3!=0:#NMAE: normalized mean absolute error
        nmae = torch_nmae(img_gt,img_recon)
    else:
        nmae=0

    if alpha4!=0:#HFEN: high frequency error norm
        hfen = compute_hfen(img_gt.cpu().detach().numpy(),img_recon.cpu().detach().numpy())
        hfen = torch.tensor(hfen)
    else:
        hfen=0

    return (alpha1*nrmse+alpha2*(1-ssim_recon)+alpha3*nmae+alpha4*hfen)/(alpha1+alpha2+alpha3+alpha4)

    



# # ## Error Metrics: Computing image reconstruction quality
# def compute_metric(img_gt,img_aliased,metric): 
    
#     if metric=='nrmse':# Normalized Root Mean Squared Error
#         return np.linalg.norm(img_gt-img_aliased)/np.linalg.norm(img_gt)

#     if metric=='nmse': # Normalized Mean Squared Error
#         return np.mean(np.abs(img_gt-img_aliased)**2)/np.mean(np.abs(img_gt)**2)

#     if metric=='psnr':
#         mse = np.mean(np.abs(img_gt-img_aliased)**2)
#         return 10*np.log10((abs(img_gt).max())**2/mse)

#     if metric=='ssim':
#         return ssim(abs(img_gt), abs(img_aliased))

#     if metric=='hfen':
#         # LoG - Laplacian of Gaussian
#         LoG_GT=gaussian_laplace(np.real(img_gt), sigma=1)+\
#         1j*gaussian_laplace(np.imag(img_gt), sigma=1)
        
#         LoG_recon=gaussian_laplace(np.real(img_aliased), sigma=1)+\
#         1j*gaussian_laplace(np.imag(img_aliased), sigma=1)
        
#         return np.linalg.norm(LoG_recon-LoG_GT)/np.linalg.norm(LoG_GT)



def compute_nmae(img_gt,img_aliased):
    return np.mean(np.abs(img_gt-img_aliased))/np.mean(np.abs(img_gt))

def torch_nmae(img_gt,img_aliased):
    return torch.mean(torch.abs(img_gt-img_aliased))/torch.mean(torch.abs(img_gt))

def torch_nrmse(img_gt,img_aliased):
    return torch.linalg.norm(img_gt-img_aliased)/torch.linalg.norm(img_gt)




def make_2d_mask_old(mask_1d,budget,height,width,init_lines):
#     if mask_1d.max()>1:
#         mask_1d=torch.sigmoid(mask_1d_mask)
        
    mask_raw = raw_normalize(mask_1d,budget) # Normalizing the 1D mask to make it to have budget no. of lines
    fullmask = mask_complete(mask_raw,width) # Adding low frequency lines to make a fullmask
    
    # Binarzing fullmask
    fullmask[fullmask>=0.5]=1
    fullmask[fullmask<0.5]=0
            
    mask_2d=fullmask[0].unsqueeze(0).repeat(height,1).detach() # Making 2D mask from 1D mask
    
    return mask_2d





def mask_from_idx(height,width,init_lines,line_idx,us_factor):
    mask_adaptive=np.zeros((height,width))    
    mask_adaptive[:,(width-init_lines)//2:(width+init_lines)//2]=True
    mask_adaptive[:,line_idx[:width//us_factor-init_lines]]=True
    return mask_adaptive


# In[1]:


def save_video(image_folder):
    video_name = str(image_folder)+'.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


# In[1]:


def calc_mask_overlap(mask1,mask2,init_lines):
    # mask1 and mask2 are 2d mask
    #init_lines is the number of low frequnecy lines
    overlap = (sum(mask1[0]==mask2[0])-init_lines)/(mask1.shape[0]-init_lines)
    return overlap



def unet_from_aliased(ksp,mps,mask,net):
    
    img_aliased=sense_recon_numpy(ksp,mps,mask)
    img_aliased=crop_img(img_aliased,320,320)
    img_aliased=abs(img_aliased)/abs(img_aliased).max()
    img_aliased = unet_recon(img_aliased,net)
    img_aliased=abs(img_aliased)/abs(img_aliased).max()
    
    return img_aliased



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





def get_ground_truth_sigpy(ksp,mps):
    img_shape = ksp.shape[1:]
    S = sp.linop.Multiply(img_shape, mps)
    F = sp.linop.FFT(ksp.shape, axes=(-1, -2))
    
    img_gt = S.H * F.H * ksp
    
    return img_gt

# def sense_recon_sigpy(ksp, mask, mps):
    
#     img_shape = mps.shape[1:]

#     S = sp.linop.Multiply(img_shape, mps)
#     F = sp.linop.FFT(ksp.shape, axes=(-1, -2))
#     P = sp.linop.Multiply(ksp.shape, mask)
#     A = P * F * S 

#     return A.H * ksp

def gen_aliased_from_gt(img,mask):
    
    F = sp.linop.FFT(img.shape)
    ksp = F*img
    P = sp.linop.Multiply(ksp.shape, mask)
    
    img_aliased = F.H * P.H *ksp
    
    return img_aliased


# In[2]:


def unet_recon(img,net):
    
    height=img.shape[0]; width=img.shape[1];
    
    img=img/abs(img).max()
    img=torch.tensor(abs(img))
    
    net_output=net(img.reshape(1,1,height,width).float())
    
    img_aliased=torch.squeeze(net_output)
    img_aliased=img_aliased.detach().numpy()
    
    return img_aliased



def loss_fn(img_input,img_output):
    # img_input: batch_size x nchannels x height x width
    nrmse=torch.zeros((img_input.shape[0]))
    # loop over batch
    for i in range(img_input.shape[0]):
        nrmse[i]=torch.linalg.norm(img_input[i]-img_output[i])/\
        torch.linalg.norm(img_input[i])
    return torch.mean(nrmse)



def sense_recon_numpy(kspace,mps,mask):
    
    # Adjoint of forward operator for multicoil data
    # S^H F^H My
    # inputs:
    # Multicoil kspace ksp: ncoils x height x width
    # Coil Sensitivity Maps mps: ncoils x height x width
    # Mask mask: height x width (default: fully-sampled)


    img_sense_recon = np.zeros((kspace.shape[1],kspace.shape[2]),dtype=complex)

    mask = crop_pad_mask(mask,kspace.shape[1],kspace.shape[2])
    
    for coil in range(kspace.shape[0]):
        masked_kspace = np.multiply(mask,kspace[coil])
        img_ifft = MRI_IFFT(masked_kspace)
        img_sense_recon += np.multiply(np.conjugate(mps[coil]),img_ifft)

    return img_sense_recon



def coil_wise_ifft(kspace,mps,mask):
    
    # COil wise reconstruction for multicoil data
    # im_recon_coil = S_coil^H F^H y_coil
    img_sense_recon = np.zeros_like(kspace)
    
    for coil in range(kspace.shape[0]):
        masked_kspace = np.multiply(mask,kspace[coil])
        img_ifft = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(masked_kspace)))
        img_sense_recon[coil] = np.multiply(np.conjugate(mps[coil]),img_ifft)

    return img_sense_recon


# In[5]:


def sense_recon_torch(kspace,mps,mask,device=torch.device('cpu')):

    kspace=kspace.to(device)
    mps=mps.to(device)
    mask=mask.to(device)
    img_sense_recon = torch.zeros((kspace.shape[1],kspace.shape[2]),dtype=torch.complex64).to(device)

    for coil in range(kspace.shape[0]):
        masked_kspace = torch.mul(mask,kspace[coil])
        img_ifft = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(masked_kspace)))
        img_sense_recon += torch.mul(torch.conj(mps[coil].to(device)),img_ifft.to(device))
        
    return img_sense_recon


# In[7]:


def mask_sim_idx(true_mask,pred_mask,base):
    
    true_mask=torch.flatten(true_mask)
    pred_mask=torch.flatten(pred_mask)
    sim_idx=0
    
    for i in range(len(true_mask)):
        if true_mask[i]==pred_mask[i]:
            sim_idx+=1
            
    return round((sim_idx-base)/len(true_mask)*100)



def unreg_recon_sigpy(ksp,mps,mask):
    import sigpy as sp
    img_shape = ksp.shape[1:]
    S = sp.linop.Multiply(img_shape, mps)
    F = sp.linop.FFT(ksp.shape, axes=(-1, -2))
    P = sp.linop.Multiply(ksp.shape, mask)
    
    img_aliased = S.H * F.H * P.H * ksp
    
    return img_aliased



def unet_from_aliased(ksp,mps,mask,net):
    
    img_aliased=sense_recon_numpy(ksp,mps,mask)
    img_aliased=crop_img(img_aliased,320,320)
    img_aliased=abs(img_aliased)/abs(img_aliased).max()
    img_aliased = unet_recon(img_aliased,net)
    img_aliased=abs(img_aliased)/abs(img_aliased).max()
    
    return img_aliased



def unet_recon_sense(ksp,mps,mask,model,crop=True,device=torch.device('cpu')):

    # Computes UNET reconstruction: f_\theta(S^H F^H M y)
    # Inputs:
    # ksp: Multicoil kspace (y). Shape: ncoils x height x width
    # mps: Coil Sensitivity Maps (S). Shape: ncoils x height x width
    # mask: Mask (M). Shape: height x width
    # model: Trained UNET Model Parameters f_\theta

    model.to(device)
    
    img_aliased=sense_recon_numpy(ksp,mps,mask)

    if crop:
        img_aliased = crop_img(img_aliased, 320,320)
        
    img_aliased=img_aliased/abs(img_aliased).max()
    
    height,width=img_aliased.shape
    nchannels = 2
    
    # if nchannels==1:
    #     inputs=torch.tensor(abs(img_aliased)).view(1,1,height,width).to(device)
    # else:
    #     # print(1,nchannels,height,width)
    inputs=torch.zeros((1,nchannels,height,width)).to(device)
    inputs[0,0] = torch.real(torch.tensor(img_aliased)).to(device)
    inputs[0,1] = torch.imag(torch.tensor(img_aliased)).to(device)
    
    # print(inputs.shape)
    outputs=model(inputs.float())
    
    # if nchannels==2:
    img_recon_model = (outputs[0,0]+1j*outputs[0,1]).cpu().detach().numpy()
    # else:
    #     img_recon_model=torch.squeeze(outputs).cpu().detach().numpy()
        
    img_recon_model=img_recon_model/abs(img_recon_model).max()
    
    return img_recon_model, img_aliased

def unet_recon_torch(ksp,mps,mask,model,nchannels=2,dataset='ocmr',device=torch.device('cpu')):

    # Computes UNET reconstruction: f_\theta(S^H F^H M y)
    # Inputs:
    # ksp: Multicoil kspace (y). Shape: ncoils x height x width
    # mps: Coil Sensitivity Maps (S). Shape: ncoils x height x width
    # mask: Mask (M). Shape: height x width
    # model: Trained UNET Model Parameters f_\theta

    model.to(device)
    
    img_aliased=sense_recon_torch(ksp,mps,mask)

    if dataset=='fastmri':
        img_aliased = crop_img(img_aliased, 320,320)
        
    img_aliased=img_aliased/abs(img_aliased).max()
    
    height,width=img_aliased.shape
    
    if nchannels==1:
        inputs=abs(img_aliased).view(1,1,height,width).to(device)
    else:
        inputs=torch.zeros((1,nchannels,height,width)).to(device)
        inputs[0,0] = torch.real(img_aliased).to(device)
        inputs[0,1] = torch.imag(img_aliased).to(device)
    
    outputs=model(inputs.float())
    
    if nchannels==2:
        img_recon_model = (outputs[0,0]+1j*outputs[0,1])
    else:
        img_recon_model=torch.squeeze(outputs)
        
    img_recon_model=img_recon_model/abs(img_recon_model).max()
    
    return img_recon_model


def unet_recon_sense_batched(ksp, mps, masks, net, nchannels=2, batch_size=16, device=torch.device('cpu'), dataset='fastmri'):
    
    # Inputs:
    # ksp: multicoil kspace of size - no. of coils x height x width
    # mps: sensitivity maps of size - no. of coils x height x width
    # masks: binary masks (2D cartesian sampling) of size - no. of masks x height x width
    
    if masks.ndim==2:
         masks=np.expand_dims(masks,axis=0)
    
    ncoils, height, width = ksp.shape
    nimages = masks.shape[0]
    
    img_recon_sense=np.zeros_like(masks,dtype='complex')
    
    # Doing SENSE Reconstruction individually for each masks
    for i in range(nImages):
        img_recon_sense[i] = sense_recon_numpy(ksp,mps,masks[i])
        
    if dataset=='fastmri':
        # Cropping reconstructions to img_size
        # For fastMRI
        img_crop_size=320
        img_recon_sense = img_recon_sense[:, (height-img_crop_size)//2:(height+img_crop_size)//2,\
                                         (width-img_crop_size)//2:(width+img_crop_size)//2] 
        height=img_crop_size
        width=img_crop_size

    # Making a numpy array of the unet output
    if nchannels==2:
        img_recon_net=np.zeros_like(img_recon_sense,dtype='complex') # array for storing UNET output/reconstructions
    else:
        img_recon_net=np.zeros_like(img_recon_sense,dtype='float64') # array for storing UNET output/reconstructions
    
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
        else:
            img_recon_net[batch] = torch.squeeze(outputs).cpu().detach().numpy()
    
        # Clearing cuda memory
        torch.cuda.empty_cache()
        
    # Normalizing UNET output
    for i in range(nImages):
        img_recon_net[i]=img_recon_net[i]/abs(img_recon_net[i]).max()

    img_recon_net = np.squeeze(img_recon_net)
    
    return img_recon_net


# def unet_recon_batched_torch(ksp, mps, masks, net, nchannels=2, batch_size=16, device=torch.device('cpu'), dataset='ocmr'):
    
#     # Inputs:
#     # ksp: multicoil kspace of size - no. of coils x height x width
#     # mps: sensitivity maps of size - no. of coils x height x width
#     # masks: binary masks (2D cartesian sampling) of size - no. of masks x height x width
    
#     if masks.ndim==2:
#          masks=masks.unsqueeze(0)
    
#     nImages, height, width = masks.shape
    
#     img_aliased = torch.zeros_like(masks)
#     img_aliased.type(ksp.dtype)
#     ksp.to(device)
#     mps.to(device)
#     masks.to(device)

#     img_recon_sense=torch.zeros_like(masks,dtype=ksp.dtype)
    
#     # Doing SENSE Reconstruction individually for each masks
#     for i in range(nImages):
#         img_recon_sense[i] = sense_recon_torch(ksp.to(device),mps.to(device),masks[i].to(device))
        
#     if dataset=='fastmri':
#         # Cropping reconstructions to img_size
#         # For fastMRI
#         img_crop_size=320
#         img_recon_sense = img_recon_sense[:, (height-img_crop_size)//2:(height+img_crop_size)//2,\
#                                          (width-img_crop_size)//2:(width+img_crop_size)//2] 
#         height=img_crop_size
#         width=img_crop_size
    
#     img_recon_net=torch.zeros_like(img_recon_sense,dtype=torch.complex64).to(device) # tensor for storing UNET output/reconstructions

#     # Iterating over batches
#     for batch in torch.split(torch.arange(nImages), nImages//batch_size+1):

#         with torch.no_grad(): # Gradient computation not required
#             # Network Input: batches of real valued sense reconstructions
#             # Input shape: batch_size x number of channels x height x width

#             inputs=torch.zeros((img_recon_sense[batch].shape[0],nchannels,height,width)).to(device)
            
#             if nchannels==1:
#                 inputs=torch.tensor(abs(img_recon_sense[batch])).to(device)
#             else:
#                 inputs[:,0] = torch.real((img_recon_sense[batch])).to(device)
#                 inputs[:,1] = torch.imag((img_recon_sense[batch])).to(device)
#             # print(device)
#             inputs = inputs/abs(inputs).max()
#             # Applying UNET
#             outputs=net(inputs.float()) # UNET output

#         img_recon_net[batch] = (outputs[0,0]+1j*outputs[0,1]).to(device)

#         # Clearing cuda memory
#         torch.cuda.empty_cache()
        
#     # Normalizing UNET output
#     for i in range(nImages):
#         img_recon_net[i]=img_recon_net[i]/abs(img_recon_net[i]).max()

#     img_recon_net = torch.squeeze(img_recon_net)
    
#     return img_recon_net


def get_images_from_npz(filename,\
                        icd_path='/egr/research-slim/shared/fastmri-multicoil/icd-masks4x/',\
                        npz_path='/egr/research-slim/shared/fastmri-multicoil/fastmri-val-npz/'):
    
    
    npz_file=np.load(npz_path+filename)
    ground_truth=npz_file['ground_truth']
    volume_kspace=npz_file['kspace']
    sensitivity_maps=npz_file['sensitivty_maps_4x']

    slc_idx = volume_kspace.shape[0]//2

    width = volume_kspace.shape[3]
    
    ksp = volume_kspace[slc_idx,:,:,(width-368)//2:(width+368)//2]
    mps=sensitivity_maps[slc_idx,:,:,(width-368)//2:(width+368)//2]
    img_gt = ground_truth[slc_idx]

    img_gt=crop_img(img_gt,320,320)
    img_gt=abs(img_gt)/abs(img_gt).max()    

#     greedy_mask=np.load(greedy_path+'greedy_mask_'+filename,allow_pickle=True)['greedy_mask']
#     greedy_mask=np.expand_dims(greedy_mask[slc_idx,(width-368)//2:(width+368)//2],0).repeat(640,0)

    icd_mask =np.load(icd_path+'icd_mask_4x_'+filename,allow_pickle=True)['icd_mask']
    icd_mask=np.expand_dims(icd_mask[(width-368)//2:(width+368)//2],0).repeat(640,0)  
        
    return ksp, mps, img_gt, icd_mask



from pygrappa import grappa


def grappa_recon(ksp,mask):

    # GRAPPA + Sum of Squares reconstruction
    # Inputs:
    # kspace: ncoils x height x width
    # mask: height x width

    nCoils, height, width = ksp.shape

    undersampled_ksp = ksp*mask
    budget = sum(mask[0])
    init_lines = budget//3 # no of lines used in acs - 1/3rd of budget

    calib = ksp[:,:,(width-init_lines)//2:(width+init_lines)//2] #auto calibration signal
    grappa_recon_ksp = grappa(np.moveaxis(undersampled_ksp,0,2), np.moveaxis(calib,0,2))

    grappa_recon_ksp = np.moveaxis(grappa_recon_ksp,2,0)

    return grappa_recon_ksp


def sum_of_squares_recon(ksp):

    nCoils, height, width = ksp.shape

    img_recon_sos = np.zeros((height,width))
    # Computing sum of squares reconstruction
    for coil in range(nCoils):
        img_recon_sos += abs(MRI_IFFT(ksp[coil]))**2

    return img_recon_sos

def sense_plus_grappa(ksp,mps,mask,lamda=1e-1):

    nCoils, height, width = ksp.shape

    grappa_recon_ksp = grappa_recon(ksp,mask)

    img_aliased = sigpy.mri.app.SenseRecon(grappa_recon_ksp, mps, lamda=lamda).run()
        
    # img_aliased = np.zeros((height,width),dtype=complex)
    
    # for coil in range(nCoils):
    #     img_ifft = MRI_IFFT(grappa_recon_ksp[coil])
    #     img_aliased += np.multiply(np.conjugate(mps[coil]),img_ifft)
    
    return img_aliased


def compute_cross_correlation(img1,img2):
    # Normalized both images by their norm
    img1 = img1/np.linalg.norm(img1)
    img2 = img2/np.linalg.norm(img2)

    # Cross correlation (x,y) = |x^Hy| 
    cross_correlation = np.abs(np.sum(np.multiply(np.conjugate(img1),img2)))
    # cross_correlation = np.dot(img1.flatten(),img2.flatten())

    return cross_correlation


def complex_to_two_channel_torch(complex_img):
    return torch.stack((torch.real(complex_img), torch.imag(complex_img)))

def complex_to_two_channel(complex_img):
    return np.stack((np.real(complex_img), np.imag(complex_img)))


def png2gif(folder_path):
    from PIL import Image
    import glob

    # Create the frames
    frames = []
    imgs = glob.glob(folder_path+"*.png")

    for img in imgs:
        frames.append(Image.open(img))

    # Save into a GIF file that loops forever
    frames[0].save('cardiac_slice.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=200, loop=0)

def apply_forward_mri_operator(img,mps):
    
    # Forward MRI Operator
    # y = FSx
    # y_i = F * S_i * x
    # inputs:
    # img - ground truth of shape height x width
    # mps -sensitivity maps of shape: ncoils x height x width


    ksp = np.zeros_like(mps)
    
    for coil in range(mps.shape[0]):
        ksp[coil] = mps[coil]*MRI_FFT(img)

    return ksp



def fastmri_crop(ksp,mps,img_gt,crop_height = 640,crop_width = 368):

    ncoils,height,width = ksp.shape

    img_ifft_coil_wise = np.zeros_like(ksp)

    ksp_cropped = np.zeros((ncoils,crop_height,crop_width),dtype='complex')
    mps_cropped = np.zeros((ncoils,crop_height,crop_width),dtype='complex')
    img_gt_cropped = np.zeros((crop_height,crop_width),dtype='complex')

    for coil in range(ncoils):
        img_ifft_coil_wise = MRI_IFFT(ksp[coil])
        img_ifft_coil_wise = crop_img(img_ifft_coil_wise,crop_height,crop_width)
        ksp_cropped[coil] = MRI_FFT(img_ifft_coil_wise)
        mps_cropped[coil] = crop_img(mps[coil],crop_height,crop_width)
        img_gt_cropped += np.multiply(np.conjugate(mps_cropped[coil]),img_ifft_coil_wise)

    return ksp_cropped,mps_cropped,img_gt_cropped


def make_equispaced_mask(height,width,budget,num_centre_freq=30):

    equispaced_mask = np.zeros((height,width),dtype='bool')

    # assiging one to central frequencies
    equispaced_mask[:,(width-num_centre_freq)//2:(width+num_centre_freq)//2]=True

    negative_frequencies = np.linspace(0,(width-num_centre_freq)//2,(budget-num_centre_freq)//2,dtype='int')
    positive_frequencies = np.linspace((width+num_centre_freq)//2,width-1,(budget-num_centre_freq)//2,dtype='int')

    for line in negative_frequencies:
        equispaced_mask[:,line]=True

    for line in positive_frequencies:
        equispaced_mask[:,line]=True

    return equispaced_mask


def convert_1d_mask_to_2d(mask_1d,height):
    mask_2d = np.repeat(np.expand_dims(mask_1d,axis=0),repeats=height,axis=0)

    return mask_2d


def plot_mr_image(img,title='',Vmax=0.7,dataset='fastmri',colorbar=False,normalize=True,crop=True):

    if torch.is_tensor(img):
        img=img.cpu().detach().numpy()

    if normalize: # normalize image
        img = img/abs(img).max()

    img=np.squeeze(img) # remove batch dimension
    if img.shape[0]==2: # if it is a two channel image, convert to complex
        img=two_channel_to_complex(img)

    if crop:
        img=crop_img(img)

    if dataset=='ocmr':
        img = np.fliplr(abs(img)) # flip left right for brain MR image
    elif dataset=='fastmri':
        img = np.flipud(abs(img)) # flip up-down for knee MR image

    plt.imshow(abs(img),cmap='gray',vmax=Vmax) # flip left right for cardiac MR image

    plt.axis('off')
    plt.title(title)

    if colorbar:
        plt.colorbar()


def load_files_from_path(path,dataset='fastmri'):
    filenames = os.listdir(path)

    img_list = []

    for file in filenames:
        img=np.load(path+file)
        if dataset=='fastmri':
            img=crop_img(img)
        img_list.append(img)

    return np.array(img_list)



def crop_pad_mask(mask,ksp_height,ksp_width):
    # required if the kspace and the mask have different shapes/ksp_width
    # if the kspace and mask shapes are different, crop mask in frequency encode or pad in phase encode direction
    # ksp_height x ksp_width: dimension of kspace on which mask needs to be applied

    mask_is_torch_tensor = False
    if torch.is_tensor(mask):
        mask_is_torch_tensor = True
        mask=mask.cpu().detach().numpy()

    if mask.shape[0]==2: # if the mask is 2-channel
        nchannels = mask.shape[0]
        mask = mask[0]
    else:
        nchannels = 1
    
    mask_freq_enc = np.expand_dims(mask[0],axis=0).repeat(ksp_height,0) # repeating mask along height dimension

    mask_reshaped = np.zeros((ksp_height,ksp_width),dtype='bool')

    if mask.shape[1]<ksp_width:
        mask_reshaped[:,(ksp_width-mask.shape[1])//2:(ksp_width+mask.shape[1])//2] = np.copy(mask_freq_enc) # padding in phase encoding direction
    elif mask.shape[1]>ksp_width:
        mask_reshaped = np.copy(mask_freq_enc[:,(mask.shape[1]-ksp_width)//2:(mask.shape[1]+ksp_width)//2]) # cropping in phase encoding direction
    else:
        mask_reshaped = np.copy(mask_freq_enc)

    if nchannels==2:
        mask_reshaped = np.expand_dims(mask_reshaped, axis=0).repeat(nchannels,0)

    if mask_is_torch_tensor:
        mask_reshaped=torch.tensor(mask_reshaped)

    return mask_reshaped



def two_channel_to_complex(img_2channel):
    # Input: 
    # Img of shape: 2 x height x width
    # Returns two channel converted to complex
    if torch.is_tensor(img_2channel):
        img_2channel = torch.squeeze(img_2channel)
    else:
        img_2channel = np.squeeze(img_2channel)
        
    return img_2channel[0] +1j * img_2channel[1]


def get_recon(mask_type,scan,slc_idx,us_factor=4,modl_data_path='/egr/research-slim/shared/fastmri-multicoil/modl-training-data-uncropped/'):

    if mask_type=='gt':
        img = np.load(modl_data_path + 'test-img-gt/test_img_gt_'+scan+'_slc'+str(slc_idx)+'.npy')
    else:
        img = np.load(modl_data_path + 'modl-training-data-'+str(us_factor)+'x-'+mask_type+'/test-img-recon/test_img_recon_'+scan+'_slc'+str(slc_idx)+'.npy')

    img = two_channel_to_complex(img)
    img = crop_img(img)
    img = img/abs(img).max()

    return img

def preprocess_img(img,height=320,width=320,crop=True):

    img = two_channel_to_complex(img)
    if crop:
        img = crop_img(img,height,width)
    img = img/abs(img).max()

    return img

def get_mask(mask_type,us_factor=4,scan='file1000628',slc_idx=23,modl_data_path='/egr/research-slim/shared/fastmri-multicoil/modl-training-data-uncropped/'):
    if mask_type in ['icd','vdrs','nn-global']: # scan and slice adaptive masks
        mask = np.load(modl_data_path + 'modl-training-data-'+str(us_factor)+'x-'+mask_type+'/test-masks/test_masks_'+scan+'_slc'+str(slc_idx)+'.npy')
    else:
        mask = np.load(modl_data_path + 'modl-training-data-'+str(us_factor)+'x-'+mask_type+'/mask.npy')

    mask = mask[0]#returning only first channel

    return mask