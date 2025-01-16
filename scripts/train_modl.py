# Code for training MoDL on set of multi-coil MR images undersampled by scan/slice adaptive masks
# Paper: Aggarwal, Hemant K., Merry P. Mani, and Mathews Jacob. "MoDL: Model-based deep learning architecture for inverse problems." IEEE TMI 38.2 (2018): 394-405.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("../utils") 
sys.path.append("../models")
from modl_cg_functions import *
# import modl_cg_functions
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
model = model.float().to(device)


learning_rate = 1e-4 # learning rate
nepochs = 100 # no. of epochs
tol = 1e-5 # tolerance for CG algorithm
lamda = 1e2 # weighting factor
num_iter = 6 # No. of unrolling of denoiser and CG block

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min') # lr scheduler

# path of directory containing the training data:
# Needed data: 1. aliased images, 2. ground truth, 3. sensitivity maps, 4. masks
training_data_path = '/egr/research-slim/shared/fastmri-multicoil/modl-training-data-uncropped/'

# Each training and validation (.npy) file contains only one slice of a particular scan
train_filenames = os.listdir(training_data_path + 'modl-training-data-4x-icd/train-img-aliased') # Getting the training filenames
val_filenames = os.listdir(training_data_path + 'modl-training-data-4x-icd/val-img-aliased') # Getting the validation filenames

ntrain = len(train_filenames) # no. of training images/slices
nval = len(val_filenames) # no. of validation images

train_loss = []
val_loss = []

for epoch in range(nepochs): # iterate over epochs

    # initialize total training and validation loss for a particular epoch
    train_loss_total = 0
    val_loss_total = 0

    for idx in range(ntrain): # iterate over training images

        # Getting the scan name and slice index
        scan = train_filenames[idx][18:29]
        slc_idx = train_filenames[idx][33:-4]

        # Load the training (two-channel) ground truth, aliased image (A^H My), and the undersampling mask, sensitivity maps
        img_gt = np.load(training_data_path + 'train-img-gt/train_img_gt_'+scan+'_slc'+str(slc_idx)+'.npy')
        img_aliased = np.load(training_data_path + 'modl-training-data-4x-icd/train-img-aliased/train_img_aliased_'+scan+'_slc'+str(slc_idx)+'.npy')
        mask = np.load(training_data_path + 'modl-training-data-4x-icd/train-masks/train_masks_'+scan+'_slc'+str(slc_idx)+'.npy')
        mps = np.load(training_data_path + 'train-maps/train_maps_'+scan+'_slc'+str(slc_idx)+'.npy')

        img_recon_modl = modl_recon_training(img_aliased, mask, mps, model, device=device) # Performing MoDL reconstruction

        target = torch.tensor(img_gt).to(device).float().unsqueeze(0)

        optimizer.zero_grad() # Zero out the gradient
        loss = loss_fn(target, img_recon_modl) # computing loss (NRMSE)
        loss.backward() # Computing gradient
        optimizer.step() # Perform the optimization step to update parameters

        train_loss_total += float(loss) # computing total loss over all training samples

    with torch.no_grad(): # gradient computation not required

        for idx in range(nval): # iterate over validation images

            # Getting the scan name and slice index
            scan = val_filenames[idx][16:27]
            slc_idx = val_filenames[idx][31:-4]

            # Load the validation (two-channel) ground truth, aliased image (A^H My), and the undersampling mask, sensitivity maps
            img_gt = np.load(training_data_path + 'val-img-gt/val_img_gt_'+scan+'_slc'+str(slc_idx)+'.npy')
            mps = np.load(training_data_path + 'val-maps/val_maps_'+scan+'_slc'+str(slc_idx)+'.npy')
            img_aliased = np.load(training_data_path + 'modl-training-data-4x-icd/val-img-aliased/val_img_aliased_'+scan+'_slc'+str(slc_idx)+'.npy')
            mask = np.load(training_data_path + 'modl-training-data-4x-icd/val-masks/val_masks_'+scan+'_slc'+str(slc_idx)+'.npy')

            img_recon_modl = modl_recon_training(img_aliased, mask, mps, model, device=device) # Getting validation output

            target = torch.tensor(img_gt).to(device).float().unsqueeze(0)

            loss = loss_fn(target, img_recon_modl)

            val_loss_total += float(loss)

    torch.cuda.empty_cache()
    
    scheduler.step(val_loss_total/nval) # Using LR scheduler to prevent overfitting

    train_loss.append(train_loss_total/ntrain)
    val_loss.append(val_loss_total/nval)

    # printing training and validation loss for each epoch
    print('Epoch: {:d} | Training Loss: {:.3f} | validation Loss: {:.3f}'\
        .format(epoch+1 , train_loss_total/ntrain, val_loss_total/nval))

    # Plotting training and validation loss in a single figure
    plt.figure()
    plt.plot(np.array(train_loss));
    plt.plot(np.array(val_loss))
    plt.grid('on');plt.xlabel('Epoch'); plt.ylabel('Loss');
    plt.legend(['Training','Valdation']);
    plt.title('Network Training: Loss vs Epoch')
    plt.savefig('loss.png')

    # Saving the model parameters
    torch.save(model.state_dict(),"model.pt")
