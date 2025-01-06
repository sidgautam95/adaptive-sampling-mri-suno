# Code for implementing the greedy sampling optimization for multicoil MRI
# Source: Gözcü, Baran, et al. "Learning-based compressive MRI." IEEE TMI 37.6 (2018): 1394-1406.
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append("../utils") 
sys.path.append("../models")
from utils import *


def greedy_sampling_optimization(ksp, mps, img_gt, initial_mask, budget, model, device, nChannels=2, \
                             recon='unet', print_loss=False, alpha1=1, alpha2=0, alpha3=0, alpha4=0, save_recon=False, num_modl_iter=6):
    
    '''
    Inputs:
        ksp - multicoil kspace, shape: no. of coils x height x width
        mps - sensitivity maps, shape: no. of coils x height x width
        img_gt - Ground truth image, shape: height x width
        initial_mask - Initial Mask (e.g. VDRS, LF, Equispaced, Greedy), shape: height x width
        model - Trained model on images undersampled by initial mask - UNet or MoDL
        recon (str) - Reconstructor to be used: UNet/MoDL, default: unet
        device - CPU/GPU, default: CPU
        nChannels - No. of Channels (1 for real, 2 for complex), default: 2

    Outputs:
        greedy_mask - Optimized Mask, shape: height x width
        loss_greedy_list - Loss for every iteration, shape: 1 x (width-budget)
    '''

    img_gt=img_gt/abs(img_gt).max() # Normalizing ground truth
    img_gt=torch.tensor(img_gt).to(device)
    
    nCoils, Height, Width = ksp.shape

    lines=np.full(Width, False) # Lines to be included in greedy_mask

    greedy_mask = np.copy(initial_mask)
      
    num_lines_to_be_added=budget-sum(lines)

    us_factor = Width//budget
    
    print('greedy_mask initiliased with lines:',sum(greedy_mask[0]))
    print('Lines to be added:',num_lines_to_be_added)

    if recon=='unet':
        img_recon_initial = unet_recon_batched(ksp,mps,initial_mask,model, device=device)
    elif recon=='modl':
        img_recon_initial = modl_recon_batched(ksp,mps,initial_mask, model = model, device = device)
    else:
        print('Incorrect choice specified')
        
    img_recon_initial = torch.squeeze(img_recon_initial).to(device)

    loss_initial = compute_loss(img_gt, img_recon_initial,alpha1,alpha2,alpha3,alpha4) # loss of initial mask
    
    line_indices=[]; loss_greedy_list=[]; iter_time=[]

    greedy_iter=0

    while sum(greedy_mask[0])<budget:  # Iterate until the desired number of lines have been added

        greedy_iter+=1
        candidate_lines = np.nonzero(np.logical_not(greedy_mask[0]))[0] # Get indices where the line could move

        num_candidate_masks = len(candidate_lines)

        # Shape of candidate_masks: (Width-Budget) x Height x Width
        # Creating array of candidate masks = num_lines_to_be_added number of greedy masks
        candidate_masks = np.expand_dims(greedy_mask, 0).repeat(num_candidate_masks,0)

        # Assigning lines to candidate masks
        for count, candidate_lines in enumerate(candidate_lines):
            candidate_masks[count,:,candidate_lines] = True

        if recon=='unet':
            img_recons = unet_recon_batched(ksp,mps,candidate_masks,model, device=device) # UNET Reconstruction
        elif recon=='modl':
            img_recons = modl_recon_batched(ksp,mps,candidate_masks, model, num_iter= num_modl_iter, device=device) # MODL Reconstruction

        # Initializing the array for storing nrmse values for candidate masks
        # Length of this numpy array is equal to number of candidate masks generated
        loss_candidate_masks = np.zeros((num_candidate_masks)) 

        # Computing loss for all candidate masks
        for count in range(num_candidate_masks): # Iterating over candidate masks
            loss_candidate_masks[count] = compute_loss(img_gt,img_recons[count].to(device),alpha1,alpha2,alpha3,alpha4) # Computing loss
             
        # Finding the index of candidate mask with lowest nrmse
        min_loss_mask = np.argmin(loss_candidate_masks)

        # Making the candidate mask with minimum nrmse as the new greedy mask
        greedy_mask = np.copy(candidate_masks[min_loss_mask])

        img_recon_greedy = img_recons[min_loss_mask]
                
        loss_greedy_list.append(np.min(loss_candidate_masks))
      
        if print_loss:
            print('Iteration:',greedy_iter,'| Lines added:',sum(greedy_mask[0]),'| Lines to be added: ',budget-sum(greedy_mask[0]),\
                  '| Loss:',round(np.min(loss_candidate_masks),4))

        np.savez('greedy_mask_'+str(us_factor)+'x.npz',greedy_mask_1d=greedy_mask[0],loss_greedy_list=np.array(loss_greedy_list),initial_mask=initial_mask[0])
        
        # Save ground truth and reconstructed image
        plt.figure()
        plt.subplot(2,3,1)
        plot_mr_image(img_gt,title='Ground Truth')
        plt.subplot(2,3,2)
        plot_mr_image(img_recon_initial,title='Initial Recon\nLoss='+str(round(np.array(loss_greedy_list)[0],3)))
        plt.subplot(2,3,3)
        plot_mr_image(img_recon_greedy,title='Greedy Recon\nLoss='+str(round(np.array(loss_greedy_list)[-1],3)))
        plt.subplot(2,3,5)
        plot_mr_image(abs(img_gt.cpu()-img_recon_initial.cpu()),Vmax=0.2,normalize=False)                
        plt.subplot(2,3,6)
        plot_mr_image(abs(img_gt.cpu()-img_recon_greedy.cpu()),Vmax=0.2,normalize=False)
        plt.tight_layout()
        plt.savefig('recon_'+str(us_factor)+'x.png')

        # Saving initial and the optimized Greedy Mask
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(initial_mask,cmap='gray')
        plt.title('Initial Mask\nLoss='+str(round(np.array(loss_greedy_list)[0],3)))
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(greedy_mask,cmap='gray')
        plt.title('Greedy Mask\nNo. of lines='+str(sum(greedy_mask[0]))+'. Loss='+str(round(np.array(loss_greedy_list)[-1],3)))
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.plot(np.arange(len(loss_greedy_list))+1,np.array(loss_greedy_list))
        plt.grid('on')
        plt.xlabel('No. of iterations')
        plt.ylabel('Loss')
        plt.savefig('greedy_mask_'+str(us_factor)+'x.png')

    return greedy_mask, line_indices, np.array(loss_greedy_list)