# Code for running ICD sampling optimization for a sample multicoil MRI scan/slice
# User specifies the reconstruction method, metric used and the undersampling factor at which the mask is acquired.
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import sys
sys.path.append("../utils") 
sys.path.append("../models")
sys.path.append("../sampling-optimization")
from unet_fbr import Unet
from utils import *
from didn import DIDN
import os
from icd_sampling_optimization import icd_sampling_optimization

torch.set_num_threads(4)

device_id = 5
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters of ICD algorithm
num_icd_iter=1 # No of ICD passes
recon = 'unet' # choose reconstruction method - unet or modl
alpha1=1;alpha2=0;alpha3=0;alpha4=0 # choose metric (alpha1=1 corresponds to NRMSE metric. For more details, check utils.py file)
nChannels=2 # No. of channels
us_factor=4 # Undersampling factor

# Load models
if recon=='unet':
    model = Unet(in_chans=nChannels, out_chans=nChannels, chans=64)
    model_path = '../../saved-models/unet_fastmri_vdrs_4x.pt'
elif recon=='modl':
    model = DIDN(nChannels, nChannels, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=2)
    model_path = '../../saved-models/modl_didn_fastmri_vdrs_4x.pt'


model.load_state_dict(torch.load(model_path,map_location=device)); # Load
model.eval()
model.to(device)

# Load multicoil kspace (ksp), sensitivity maps (coil sensitivity maps) and ground truth (img_gt) for the particular scan and slice
# Required format: 
# ksp - complex array of shape: ncoils x height x width
# mps - complex array of shape: ncoils x height x width
# img_gt - complex array of shape: height x width
ksp = np.load('../../data/ksp.npy')
mps = np.load('../../data/mps.npy') 
# If sensitivity maps are not generated already, use ESPirit calibration app 
# mps = sigpy.mri.app.EspiritCalib(ksp)
# Source: https://sigpy.readthedocs.io/en/latest/generated/sigpy.mri.app.EspiritCalib.html

Height = ksp.shape[1]
Width = ksp.shape[2]

# Once the coil sensitivty maps are generated, use the following function to get the ground truth, if not already generated
# img_gt = preprocess_data(ksp,mps,np.ones((height,width)))
# Or use pre-generated ones 
img_gt = np.load('../../data/img_gt.npy')

img_gt = crop_img(img_gt) # crop to central 320 x 320 region of interest
img_gt = img_gt/abs(img_gt).max() # normalize ground truth

budget=Width//us_factor # Total no. of phase encoding lines in the mask
num_centre_lines = budget//3 # No. of low frequency lines at centre (the algorithm doesn't move those lines)

# Choose initial mask
# Initialize with a variable density random sampling mask
# Alternatives: 1. low frequency mask, 2. equispaced mask, 3. LOUPE mask, etc.
initial_mask =  make_vdrs_mask(Height,Width,budget,num_centre_lines) 

# Converting the kspace, maps, ground truth and initial mask to a pytorch tensor
ksp = torch.tensor(ksp).to(device)
mps = torch.tensor(mps).to(device)
img_gt = torch.tensor(img_gt).to(device)
initial_mask = torch.tensor(initial_mask).to(device)

# Run ICD sampling optimization
icd_mask, loss_icd_list = icd_sampling_optimization(ksp,mps,img_gt,initial_mask,budget,num_centre_lines, model,device, num_icd_iter, nChannels, \
                             recon,alpha1,alpha2,alpha3,alpha4,print_loss=True,save_recon=True,num_modl_iter=6)
