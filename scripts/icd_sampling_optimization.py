# Code for running the ICD sampling optimization for multicoil MRI
import numpy as np
import matplotlib.pyplot as plt
import torch
# import sys
# sys.path.append("../utils") 
# sys.path.append("../models")
from utils import *


def icd_sampling_optimization(ksp,mps,img_gt,initial_mask,budget,num_centre_lines, model,device, num_icd_passes=1, nChannels=2, \
                             recon='unet',print_loss=False,alpha1=1,alpha2=0,alpha3=0,alpha4=0,save_recon=False,num_modl_iter=6):
    
    '''
    Inputs:
        ksp: multicoil kspace, shape: no. of coils x height x width
        mps: sensitivity maps, shape: no. of coils x height x width
        img_gt - Ground truth image, shape: height x width
        initial_mask - Initial Mask (e.g. VDRS, LF, Equispaced, Greedy), shape: height x width
        model - Trained model on images undersampled by initial mask - UNet or MoDL
        num_icd_passes - No. of ICD passes
        scan (str) - name of scan
        slc_idx (int) - slice index
        recon (str) - Reconstructor to be used: UNet/MoDL, default: unet
        device - CPU/GPU, default: CPU
        nChannels - No. of Channels (1 for real, 2 for complex), default: 2

    Outputs
    '''

    model.to(device)
   
    nCoils, height, width = ksp.shape

    img_gt=crop_img(img_gt) # crop to central 320 x 320 region 
    img_gt=img_gt/abs(img_gt).max() # Normalizing ground truth by maximum magniude

    if nChannels==1:
        img_gt = abs(img_gt)

    icd_mask=torch.clone(initial_mask) # Initializing the ICD mask to be initial mask
    us_factor=width//budget


    # Getting ICD loss and reconstruction using the original ICD mask
    if recon=='unet':
        img_recon_initial = unet_recon_batched(ksp,mps,initial_mask,model, device=device)
    elif recon=='modl':
        img_recon_initial = modl_recon_batched(ksp,mps,initial_mask, model = model, device = device)
    else:
        print('Incorrect choice specified')
        
    img_recon_initial = torch.squeeze(img_recon_initial).to(device)

    loss_initial = compute_loss(img_gt, img_recon_initial,alpha1,alpha2,alpha3,alpha4) # loss of initial mask

    iter_count=0

    print('Running ICD algorithm')
    print('Undersampling factor:',us_factor)
    print('Budget:',budget)
    print('Lines initialized at centre:',num_centre_lines)
    print('No. of possible sampling locations available:',(budget-num_centre_lines))
    print('Reconstructor Used:',recon)

    loss_icd = torch.clone(loss_initial)
    img_recon_icd = torch.clone(img_recon_initial)

    loss_icd_list = []
    loss_icd_list.append(loss_icd.cpu().detach().numpy())

    for iteration in range(num_icd_passes): # No. of ICD Iterations

        print('ICD Iteration', iteration+1,'out of',num_icd_passes)

        # Find index of lines already added/present
        added_lines_with_low_frequency = np.nonzero(icd_mask[0].cpu().detach().numpy())[0]

        low_frequency_indices = np.arange((width-num_centre_lines)//2,(width+num_centre_lines)//2)

        # Enforcing the low frequency part to be True
        icd_mask[:,low_frequency_indices] = True

        # removing the low frequnecy lines from the indices of added lines
        lines_to_be_moved = np.array(list(set(added_lines_with_low_frequency)-set(low_frequency_indices)))

        for current_idx in lines_to_be_moved: # Loop over added lines

            # Get indices where the line could move
            candidate_lines = np.nonzero(np.logical_not(icd_mask[0].cpu().detach().numpy()))[0] 

            num_candidate_masks = len(candidate_lines)

            # Shape of candidate_masks: (width-Budget) x height x width
            # Creating array of candidate masks by expanding dimensions of ICD masks and repeating it over one dimension
            candidate_masks = icd_mask.unsqueeze(0).repeat(num_candidate_masks,1,1)
            
            # Removing the line from icd mask which is to be moved
            candidate_masks[:, :, current_idx] = False 

            # Assigning lines to candidate masks
            for count, candidate_lines in enumerate(candidate_lines):
                candidate_masks[count,:,candidate_lines] = True

            if recon=='unet':
                img_recons = unet_recon_batched(ksp,mps,candidate_masks,model, device=device) # UNET Reconstruction
            elif recon=='modl':
                img_recons = modl_recon_batched(ksp,mps,candidate_masks, model, num_iter= num_modl_iter, device=device)

            # Initializing the array for storing loss values for candidate masks
            loss_candidate_masks = torch.zeros((num_candidate_masks)) 

            # Computing loss for all candidate masks
            for count in range(num_candidate_masks): # Iterating over candidate masks
                loss_candidate_masks[count] = compute_loss(img_gt,img_recons[count].to(device),alpha1,alpha2,alpha3,alpha4) # Computing loss
             
            # End of loop for finding candidate masks
            ####################################################

            # Checking if any candidate mask has lesser loss than that of original ICD mask
            if torch.min(loss_candidate_masks) < loss_icd:

                # Finding the index of candidate mask with lowest loss
                min_loss_mask = torch.argmin(loss_candidate_masks)

                # Making the candidate mask with minimum loss as the new ICD mask
                icd_mask = torch.clone(candidate_masks[min_loss_mask])

                loss_icd = torch.clone(torch.min(loss_candidate_masks))

                img_recon_icd = img_recons[min_loss_mask]
            else:
                if print_loss:
                    print('Line not moved.')

            loss_icd_list.append(loss_icd.cpu().detach().numpy())

            np.savez('icd_mask_'+str(us_factor)+'x.npz',\
            icd_mask_1d=icd_mask[0].cpu().detach().numpy(),loss_icd_list=np.array(loss_icd_list),\
            initial_mask=initial_mask[0].cpu().detach().numpy())

            if save_recon:
                np.save('img_recon_icd_'+str(us_factor)+'x',img_recon_icd.cpu().detach().numpy())

            # Save ground truth and reconstructed image
            plt.figure()
            plt.subplot(2,3,1)
            plot_mr_image(img_gt,title='Ground Truth')
            plt.subplot(2,3,2)
            plot_mr_image(img_recon_initial,title='Initial Recon\nLoss='+str(round(np.array(loss_icd_list)[0],3)))
            plt.subplot(2,3,3)
            plot_mr_image(img_recon_icd,title='ICD Recon\nLoss='+str(round(np.array(loss_icd_list)[-1],3)))
            plt.subplot(2,3,5)
            plot_mr_image(abs(img_gt.cpu()-img_recon_initial.cpu()),Vmax=0.2,normalize=False)                
            plt.subplot(2,3,6)
            plot_mr_image(abs(img_gt.cpu()-img_recon_icd.cpu()),Vmax=0.2,normalize=False)
            plt.tight_layout()
            plt.savefig('recon_'+str(us_factor)+'x.png')

            # Saving initial and the optimized ICD Mask
            plt.figure(figsize=(15,5))
            plt.subplot(1,3,1)
            plt.imshow(initial_mask.cpu().detach().numpy(),cmap='gray')
            plt.title('Initial Mask\nloss='+str(round(np.array(loss_icd_list)[0],3)))
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.imshow(icd_mask.cpu().detach().numpy(),cmap='gray')
            plt.title('ICD Mask\nloss='+str(round(np.array(loss_icd_list)[-1],3)))
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.plot(np.arange(len(loss_icd_list))+1,np.array(loss_icd_list))
            plt.grid('on')
            plt.xlabel('No. of iterations')
            plt.ylabel('Loss')
            plt.savefig('icd_mask_'+str(us_factor)+'x.png')

    return icd_mask, np.array(loss_icd_list) # returns ICD mask and loss over iterations
