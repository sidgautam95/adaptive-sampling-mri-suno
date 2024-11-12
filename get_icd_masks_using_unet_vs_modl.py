import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import sys
sys.path.insert(1, '/egr/research-slim/gautamsi/mri-sampling')
from unet_fbr import Unet
from utils import *
from modl_utils import *
from didn import DIDN

npz_path='/egr/research-slim/shared/fastmri-multicoil/fastmri-val-npz/'

all_scans = os.listdir(npz_path)

filenames=[]
gpu_no = 6

torch.set_num_threads(4)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_no)

use_gpu=True

if use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


nChannels=2

unet_4x = Unet(in_chans=nChannels, out_chans=nChannels, chans=64)
unet_4x.load_state_dict(torch.load('/egr/research-slim/shared/fastmri-multicoil/saved-models-Angqi/latest-train-model-4x/unet_fastmri_vdrs_4x.pt',map_location=device)); # Load
unet_4x.eval()
unet_4x.to(device)

modl_4x = DIDN(nChannels, nChannels, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=2)
modl_4x.load_state_dict(torch.load('/egr/research-slim/shared/fastmri-multicoil/saved-models-siddhant/saved-models-uncropped/modl_didn_fastmri_vdrs_4x.pt',map_location=device)) # Load
modl_4x.eval()
modl_4x.to(device)

unet_8x = Unet(in_chans=nChannels, out_chans=nChannels, chans=64)
unet_8x.load_state_dict(torch.load('/egr/research-slim/shared/fastmri-multicoil/saved-models-siddhant/saved-models-unet-cropped/unet_fastmri_vdrs_8x.pt',map_location=device)); # Load
unet_8x.eval()
unet_8x.to(device)

modl_8x = DIDN(nChannels, nChannels, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=2)
modl_8x.load_state_dict(torch.load('/egr/research-slim/shared/fastmri-multicoil/saved-models-siddhant/saved-models-uncropped/modl_didn_fastmri_vdrs_8x.pt',map_location=device)) # Load
modl_8x.to(device)


from randomized_icd_model import randomized_icd_algorithm

filenames = os.listdir(npz_path)
folder_path=''
# os.makedirs(folder_path)
num_icd_iter=1

k=1 # subset size

tol=-1

testing_scans = open('/egr/research-slim/shared/fastmri-multicoil/fastmri_testing_scans.txt', 'r').readlines()

# scan = 'file1000628'
# slc_idx = 21

# scan = 'file1000328'
# slc_idx=22

scan = 'file1000628'
slc_idx=19

npz_file=np.load(npz_path+scan+'.npz')

ground_truth=npz_file['ground_truth']
volume_kspace=npz_file['kspace']
sensitivity_maps=npz_file['sensitivty_maps_4x']

# k-space dimension : (number of slices, number of coils, height, width)
nSlices,nCoils,Height,Width=volume_kspace.shape

ksp=volume_kspace[slc_idx]
mps=sensitivity_maps[slc_idx]

# gt_path = '/egr/research-slim/shared/fastmri-multicoil/modl-training-data-uncropped/test-img-gt/test_img_gt_'+scan+'_slc'+str(slc_idx)+'.npy'

# img_gt=np.load(gt_path)
# img_gt = two_channel_to_complex(img_gt)
img_gt = ground_truth[slc_idx]
img_gt = crop_img(img_gt)
img_gt = img_gt/abs(img_gt).max()

plt.figure()
plot_mr_image(img_gt)
plt.savefig('gt.png')


us_factor=4
nlines=Width//us_factor
budget = nlines
init_lines = nlines//3

num_centre_lines = init_lines

initial_mask = make_vdrs_mask(Height,Width,nlines,init_lines)

randomized_icd_algorithm(ksp,mps,img_gt,initial_mask,budget,num_centre_lines,unet_4x,device,k,folder_path,'vdrs_unet',\
num_icd_iter=num_icd_iter, nChannels=2,recon='unet',dataset='fastmri',alpha1=1,alpha2=0,alpha3=0,print_loss=False,loss_over_roi=False,tol=-1,num_modl_iter=6)

randomized_icd_algorithm(ksp,mps,img_gt,initial_mask,budget,num_centre_lines,modl_4x,device,k,folder_path,'vdrs_modl',\
num_icd_iter=num_icd_iter, nChannels=2,recon='modl',dataset='fastmri',alpha1=1,alpha2=0,alpha3=0,print_loss=False,loss_over_roi=False,tol=-1,num_modl_iter=6)


us_factor=8
nlines=Width//us_factor
budget = nlines
init_lines = nlines//3

num_centre_lines = init_lines

initial_mask = make_vdrs_mask(Height,Width,nlines,init_lines)

randomized_icd_algorithm(ksp,mps,img_gt,initial_mask,budget,num_centre_lines,unet_8x,device,k,folder_path,'vdrs_unet',\
num_icd_iter=num_icd_iter, nChannels=2,recon='unet',dataset='fastmri',alpha1=1,alpha2=0,alpha3=0,print_loss=False,loss_over_roi=False,tol=-1,num_modl_iter=6)

randomized_icd_algorithm(ksp,mps,img_gt,initial_mask,budget,num_centre_lines,modl_8x,device,k,folder_path,'vdrs_modl',\
num_icd_iter=num_icd_iter, nChannels=2,recon='modl',dataset='fastmri',alpha1=1,alpha2=0,alpha3=0,print_loss=False,loss_over_roi=False,tol=-1,num_modl_iter=6)
