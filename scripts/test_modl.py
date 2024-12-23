# Code for testing/inference using pre-trained MoDL weights on the SUNO predicted mask for multicoil MRI
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("../utils") 
sys.path.append("../models")
from modl_cg_functions import *
import modl_cg_functions
from didn import DIDN
import matplotlib.pyplot as plt
from unet_fbr import Unet
from utils import *
import os

torch.cuda.empty_cache()
    
device_id = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()
nChannels=2
modl_path = '../../saved-models/modl_didn_fastmri_vdrs_4x.pt'
model = DIDN(nChannels, nChannels, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=2)
model.load_state_dict(torch.load(modl_path,map_location=device)) # Load
model.to(device)
model.eval();
print('MoDL loaded from',modl_path)

# Choose fully sampled kspace and the corresponding coil sensitivity maps of test k-space
ksp = np.load('../../data/ksp.npy')
mps = np.load('../../data/mps.npy')
mask = np.load('suno_mask.npy') # choosing SUNO mask

ksp = torch.tensor(ksp).to(device)
mps = torch.tensor(mps).to(device)
mask = torch.tensor(mask).to(device)

# Performing MoDL reconstruction
img_recon_modl = modl_recon(ksp,mps,mask, model, device=device)

# Saving reconstructed image
np.save('test_img_recon',torch.squeeze(img_recon_modl).cpu().detach().numpy())
