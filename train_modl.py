import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from modl_cg_functions import *
import modl_cg_functions
from modl_utils import *
from torch.nn import init
from didn import DIDN
import matplotlib.pyplot as plt
from unet_fbr import Unet
from utils import *
import os

torch.cuda.empty_cache()
    
device_id = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DIDN(2, 2, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=2)

init_weights(model, init_type='normal', gain=0.02)
model = model.float().to(device)
# Loss and optimizer
learning_rate = 1e-4 # learning rate

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_loss = []
val_loss = []

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

nepochs = 100 # no. of epochs
tol = 1e-5 # tolerance for cg algorithm
lamda = 1e2 # lamda weighting in the MoDL equation
num_iter = 6 # No. of MoDL unrolling

modl_data_path = '/egr/research-slim/gautamsi/shared/fastmri-multicoil/modl-training-data-uncropped/'

# path of the directory containing the training aliased images
train_img_aliased_path = modl_data_path + 'modl-training-data-4x-icd/train-img-aliased'
train_img_aliased_filenames = os.listdir(train_img_aliased_path)

# path of the directory containing the validation aliased images
val_img_aliased_path = modl_data_path + 'modl-training-data-4x-icd/val-img-aliased'
val_img_aliased_filenames = os.listdir(val_img_aliased_path)

ntrain = len(train_img_aliased_filenames) # no. of training images/slices
nval = len(val_img_aliased_filenames) # no. of validation imagaes

for epoch in range(nepochs): # iterate over epochs

    # initialize total training and validation loss for a particular epoch
    train_loss_total = 0
    val_loss_total = 0

    for idx in range(ntrain): # iterate over training images

        scan = train_img_aliased_filenames[idx][18:29]
        slc_idx = train_img_aliased_filenames[idx][33:-4]

        img_gt = np.load(modl_data_path + 'train-img-gt/train_img_gt_'+scan+'_slc'+str(slc_idx)+'.npy')
        smap = np.load(modl_data_path + 'train-maps/train_maps_'+scan+'_slc'+str(slc_idx)+'.npy')
        img_aliased = np.load(modl_data_path + 'modl-training-data-4x-icd/train-img-aliased/train_img_aliased_'+scan+'_slc'+str(slc_idx)+'.npy')
        mask = np.load(modl_data_path + 'modl-training-data-4x-icd/train-masks/train_masks_'+scan+'_slc'+str(slc_idx)+'.npy')

        final_output = modl_recon_training(img_aliased, mask, smap, model, device=device)

        target = torch.tensor(img_gt).to(device).float().unsqueeze(0)

        optimizer.zero_grad()
        loss = loss_fn(crop_img(target), crop_img(final_output)) # computing loss only over ROI i.e. central 320 x 320 region
        loss.backward()
        optimizer.step()

        train_loss_total += float(loss)

        
    with torch.no_grad():

        for idx in range(nval): # iterate over validation images

            scan = val_img_aliased_filenames[idx][16:27]
            slc_idx = val_img_aliased_filenames[idx][31:-4]

            img_gt = np.load(modl_data_path + 'val-img-gt/val_img_gt_'+scan+'_slc'+str(slc_idx)+'.npy')
            smap = np.load(modl_data_path + 'val-maps/val_maps_'+scan+'_slc'+str(slc_idx)+'.npy')
            img_aliased = np.load(modl_data_path + 'modl-training-data-4x-icd/val-img-aliased/val_img_aliased_'+scan+'_slc'+str(slc_idx)+'.npy')
            mask = np.load(modl_data_path + 'modl-training-data-4x-icd/val-masks/val_masks_'+scan+'_slc'+str(slc_idx)+'.npy')

            final_output = modl_recon_training(img_aliased, mask, smap, model, device=device)

            target = torch.tensor(img_gt).to(device).float().unsqueeze(0)

            loss = loss_fn(crop_img(target), crop_img(final_output))

            val_loss_total += float(loss)

    torch.cuda.empty_cache()
    
    scheduler.step(val_loss_total/nval)

    train_loss.append(train_loss_total/ntrain)
    val_loss.append(val_loss_total/nval)

    print('Epoch: {:d} | Training Loss: {:.3f} | validation Loss: {:.3f}'\
        .format(epoch+1 , train_loss_total/ntrain, val_loss_total/nval))

    plt.figure()
    plt.plot(np.array(train_loss));
    plt.plot(np.array(val_loss))
    plt.grid('on');plt.xlabel('Epoch'); plt.ylabel('Loss');
    plt.legend(['Training','valdation']);
    plt.title('Network Training: Loss vs Epoch')
    plt.savefig('loss.png')

    # Saving the network and model parameters
    torch.save(model.state_dict(),"model.pt")
