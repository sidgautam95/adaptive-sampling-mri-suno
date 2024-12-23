# Code for running ICD sampling optimization for a sample multicoil MRI scan/slice
# User specifies the reconstruction method, metric used and the undersampling factor at which the mask is acquired.
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import sys
sys.path.append("../utils") 
sys.path.append("../models")
from unet_fbr import Unet
from utils import *
from didn import DIDN
import os
from icd_sampling_optimization import icd_sampling_optimization

torch.set_num_threads(4)

device_id = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters of ICD algorithm
num_icd_iter=1 # No of ICD passes
recon = 'unet'#'unet' # choose reconstruction method
metric = 'nrmse' # choose metric
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

ksp = np.load('../../data/ksp.npy')
mps = np.load('../../data/mps.npy')
img_gt = np.load('../../data/img_gt.npy')

img_gt = crop_img(img_gt) # crop ground truth
img_gt = img_gt/abs(img_gt).max() # normalize ground truth

Width = ksp.shape[2]

budget=Width//us_factor
num_centre_lines = budget//3

initial_mask = np.load('../data/vdrs_mask_4x.npy')

ksp = torch.tensor(ksp).to(device)
mps = torch.tensor(mps).to(device)
img_gt = torch.tensor(img_gt).to(device)
initial_mask = torch.tensor(initial_mask).to(device)


icd_mask, loss_icd_list = icd_sampling_optimization(ksp,mps,img_gt,initial_mask,budget,num_centre_lines, model,device, num_icd_iter, nChannels, \
                             recon,print_loss=False,alpha1=1,alpha2=0,alpha3=0,alpha4=0,save_recon=False,num_modl_iter=6)
