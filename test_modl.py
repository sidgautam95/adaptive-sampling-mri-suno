import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(1, '/egr/research-slim/gautamsi/mri-sampling')
from torch.nn import init
from didn import DIDN
import matplotlib.pyplot as plt
from unet_fbr import Unet
from utils import *
from modl_utils import *
import shutil

torch.cuda.empty_cache()
    
gpu_no = 3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_no)

use_gpu=True

if use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


torch.cuda.empty_cache()
nChannels=2
modl_path = '/egr/research-slim/shared/fastmri-multicoil/saved-models-siddhant/saved-models-uncropped/modl_didn_fastmri_icd_4x.pt'
model = DIDN(nChannels, nChannels, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=2)
model.load_state_dict(torch.load(modl_path,map_location=device)) # Load
model.to(device)
model.eval();
print('MoDL loaded from',modl_path)

tol = 0.00001
lamda = 1e2
num_iter = 6

print('lambda=',lamda)

modl_data_path = '/egr/research-slim/shared/fastmri-multicoil/modl-training-data-uncropped/'

scan = ...
slc_idx = ..

img_gt = np.load(modl_data_path + 'test-img-gt/test_img_gt_'+scan+'_slc'+str(slc_idx)+'.npy')
smap = np.load(modl_data_path + 'test-maps/test_maps_'+scan+'_slc'+str(slc_idx)+'.npy')
img_aliased = np.load(modl_data_path + 'modl-training-data-4x-nn-global/test-img-aliased/test_img_aliased_'+scan+'_slc'+str(slc_idx)+'.npy')
mask = np.load(modl_data_path + 'modl-training-data-4x-nn-global/test-masks/test_masks_'+scan+'_slc'+str(slc_idx)+'.npy')

mask = crop_pad_mask(mask,img_gt.shape[1],img_gt.shape[2])

start=time.time()

final_output = modl_recon_training(img_aliased, mask, smap, model, device=device)

np.save('test_img_recon',torch.squeeze(final_output).cpu().detach().numpy())