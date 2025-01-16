# Code for preprocessing data for MoDL and U-Net training
import numpy as np
import os
import torch
import sys
sys.path.append("../utils") 
sys.path.append("../models")
from modl_cg_functions import *
from utils import *

## Data preprocessing            
kspace_path = '/egr/research-slim/shared/fastmri-multicoil/fastmri-val-npz/'
save_dir = 'modl-training-data/'

# Divide scans into training and validation and load them into seprate lists
training_scans = open('/egr/research-slim/shared/fastmri-multicoil/fastmri_training_scans.txt', 'r').readlines() 
validation_scans = open('/egr/research-slim/shared/fastmri-multicoil/fastmri_validation_scans.txt', 'r').readlines()

total_scans = training_scans + validation_scans

# Create subfolders for saving training and validation data for each scan and slice
os.makedirs(save_dir)
os.makedirs(save_dir+'train-img-aliased')
os.makedirs(save_dir+'train-masks')
os.makedirs(save_dir+'train-img-gt')
os.makedirs(save_dir+'train-maps')
os.makedirs(save_dir+'val-img-aliased')
os.makedirs(save_dir+'val-masks')
os.makedirs(save_dir+'val-img-gt')
os.makedirs(save_dir+'val-maps')

for count, scan in enumerate(total_scans): # iteratve over total scans

    scan = scan[:-1] # removing new line character from scan string
    
    file = np.load(kspace_path+scan+'.npz')

    volume_kspace = file['kspace']
    sensitivity_maps = file['sensitivty_maps_4x']

    nslices, ncoils, height, width = volume_kspace.shape

    for slc_idx in range(10, nslices-5): # ignoring first 10 and last 5 slices for fastmri dataset

        kspace = volume_kspace[slc_idx] # Get the multicoil kspace for particular scan
        maps = sensitivity_maps[slc_idx] # Get the coil sensitivity maps (can be generated usin ESPiRiT)
        icd_mask = np.load(icd_mask_path) # Get the optimized ICD mask for the particular scan and slice (User enters the path of stored optimized scan-adaptive ICD mask)
    
        img_gt, _, _ = preprocess_data(kspace,maps,np.ones((height,width))) # get the two channel ground truth images

        img_aliased, mask, _ = preprocess_data(kspace,maps,icd_mask) # get the two channel aliased image and the mask

        # Save the ground truth, aliased image, undersampling mask and the coil sensitivity maps (all two-channel) for each scan and slice
        if scan in training_scans:
            np.save(save_dir+'train-img-aliased/train_img_aliased_'+scan+'_slc'+str(slc_idx),img_aliased.detach().numpy())
            np.save(save_dir+'train-masks/train_masks_'+scan+'_slc'+str(slc_idx),mask.detach().numpy())
            np.save(save_dir+'train-maps/train_maps_'+scan+'_slc'+str(slc_idx),mask.detach().numpy())
            np.save(save_dir+'train-img-gt/train_img_gt_'+scan+'_slc'+str(slc_idx),mask.detach().numpy())
        else:
            np.save(save_dir+'val-img-aliased/val_img_aliased_'+scan+'_slc'+str(slc_idx),img_aliased.detach().numpy())
            np.save(save_dir+'val-masks/val_masks_'+scan+'_slc'+str(slc_idx),mask.detach().numpy())
            np.save(save_dir+'val-maps/val_maps_'+scan+'_slc'+str(slc_idx),mask.detach().numpy())
            np.save(save_dir+'val-img-gt/val_img_gt_'+scan+'_slc'+str(slc_idx),mask.detach().numpy())