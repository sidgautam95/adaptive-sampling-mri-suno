# Code for implementing the scan-adaptive version of greedy mask optimization for multicoil MRI
# Paper: Gözcü, Baran, et al. "Learning-based compressive MRI." IEEE TMI 37.6 (2018): 1394-1406.

import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import *

def greedy_sampling_optimization(ksp, mps, img_gt, initial_mask, budget, model, device, nChannels=2, 
                                 recon='unet', alpha1=1, alpha2=0, alpha3=0, alpha4=0, print_loss=False,
                                 save_recon=False, num_modl_iter=6):
    """
    Inputs:
        ksp - multicoil kspace, shape: no. of coils x height x width
        mps - sensitivity maps, shape: no. of coils x height x width
        img_gt - Ground truth image, shape: height x width
        initial_mask - Initial Mask (e.g., VDRS, LF, Equispaced, Greedy), shape: height x width
        budget - Total number of lines to include in the mask
        model - Trained model weights (e.g., UNet or MoDL)
        device - CPU/GPU, default: CPU
        recon - Reconstructor to be used: 'unet' or 'modl', default: 'unet'
        nChannels - Number of channels (1 for real, 2 for complex), default: 2
        alpha1, alpha2, alpha3, alpha4 - weighting for NRMSE, SSIM, NMAE and HFEN term in computing loss

    Outputs:
        greedy_mask - Optimized mask, shape: height x width
        loss_greedy_list - Loss at each iteration, shape: 1 x (budget - num_centre_lines)
    """

    # Normalize ground truth
    img_gt = img_gt / torch.abs(img_gt).max()
    img_gt = img_gt.to(device)

    nCoils, Height, Width = ksp.shape
    greedy_mask = initial_mask.clone()  # PyTorch tensor for the mask
    num_lines_to_add = budget - torch.sum(greedy_mask[0]).item()
    us_factor = Width // budget

    print('Running greedy sampling optimization algorithm')
    print('Undersampling factor:',us_factor)
    print('Budget:',budget)
    print(f"Greedy mask initialized with lines: {torch.sum(greedy_mask[0])}")
    print(f"Lines to be added: {num_lines_to_add}")
    print('Reconstructor Used:',recon)

    if recon == 'unet':
        img_recon_initial = unet_recon_batched(ksp, mps, initial_mask, model, device=device)
    elif recon == 'modl':
        img_recon_initial = modl_recon_batched(ksp, mps, initial_mask, model=model, device=device)
    else:
        raise ValueError("Incorrect choice specified for recon")

    img_recon_initial = img_recon_initial.squeeze().to(device)
    loss_initial = compute_loss(img_gt, img_recon_initial, alpha1, alpha2, alpha3, alpha4)

    loss_greedy_list = []
    greedy_iter = 0

    while torch.sum(greedy_mask[0]).item() < budget:
        greedy_iter += 1

        # Get indices of candidate lines to add
        candidate_lines = torch.where(~greedy_mask[0])[0]
        num_candidate_masks = len(candidate_lines)

        # Create candidate masks
        candidate_masks = greedy_mask.unsqueeze(0).repeat(num_candidate_masks, 1, 1)
        for i, line in enumerate(candidate_lines):
            candidate_masks[i, :, line] = True

        # Perform reconstruction with candidate masks
        if recon == 'unet':
            img_recons = unet_recon_batched(ksp, mps, candidate_masks, model, device=device)
        elif recon == 'modl':
            img_recons = modl_recon_batched(ksp, mps, candidate_masks, model=model, num_iter=num_modl_iter, device=device)

        # Compute loss for each candidate mask
        loss_candidate_masks = torch.zeros(num_candidate_masks, device=device)
        for i in range(num_candidate_masks):
            loss_candidate_masks[i] = compute_loss(img_gt, img_recons[i].to(device), alpha1, alpha2, alpha3, alpha4)

        # Find and apply the best candidate mask
        min_loss_idx = torch.argmin(loss_candidate_masks)
        greedy_mask = candidate_masks[min_loss_idx]

        img_recon_greedy = img_recons[min_loss_idx]
        loss_greedy_list.append(loss_candidate_masks[min_loss_idx].item())

        if print_loss:
            print(f"Iteration: {greedy_iter} | Lines added: {torch.sum(greedy_mask[0]).item()} | "
                  f"Lines to be added: {budget - torch.sum(greedy_mask[0]).item()} | Loss: {loss_candidate_masks[min_loss_idx].item():.4f}")

        # Save mask and reconstruction
        np.savez('greedy_mask_'+str(us_factor)+'x.npz',greedy_mask_1d=greedy_mask[0].cpu().detach().numpy(),\
        loss_greedy_list=np.array(loss_greedy_list),initial_mask=initial_mask[0].cpu().detach().numpy())
    
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
        plt.imshow(initial_mask.cpu(),cmap='gray')
        plt.title('Initial Mask\nLoss='+str(round(np.array(loss_greedy_list)[0],3)))
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(greedy_mask.cpu(),cmap='gray')
        plt.title('Greedy Mask\nNo. of lines='+str(sum(greedy_mask[0]).item())+'. Loss='+str(round(np.array(loss_greedy_list)[-1],3)))
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.plot(np.arange(len(loss_greedy_list))+1,np.array(loss_greedy_list))
        plt.grid('on')
        plt.xlabel('No. of iterations')
        plt.ylabel('Loss')
        plt.savefig('greedy_mask_'+str(us_factor)+'x.png')

    return greedy_mask, loss_greedy_list
