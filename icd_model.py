import numpy as np
import matplotlib.pyplot as plt
from utils import *
from modl_utils import *

def icd_model(ksp,mps,img_gt,initial_mask,model,num_icd_iter,filename,folder_path='',recon='unet',device=torch.device('cpu'),dataset='fastmri',nChannels=2,print_loss=False):
    
    '''
    Inputs:
        ksp: multicoil kspace of size - no. of coils x height x width
        mps: sensitivity maps of size - no. of coils x height x width
        img_gt - Ground truth image with shape (Height, Width)
        initial_mask - 2D Initial Mask
        model - Trained model model on initial masks
        device - CPU or GPU on which to put Unet model
        scan - name of scan containing kspace
        
    '''
    
    model.to(device)
    nrmse_icd_list = []
    print('Running ICD algorithm')

    # Normalizing ground truth
    img_gt=img_gt/abs(img_gt).max()

    nCoils, Height, Width = ksp.shape

    icd_mask=np.copy(initial_mask) # Initializing the ICD mask to be initial mask

    us_factor = initial_mask.shape[-1]//sum(initial_mask[0]) # undersampling factor

    for iteration in range(num_icd_iter): # No. of ICD Iterations

        print('ICD Iteration', iteration+1,'out of',num_icd_iter)

        added_lines = np.nonzero(icd_mask[0])[0]

        for current_idx in added_lines: # Loop over every added line

#             print('Moving line:', current_idx+1)

            candidate_lines = np.nonzero(np.logical_not(icd_mask[0]))[0] # Get indices where the line could move

            # Shape of candidate_masks: (Width-Budget) x Height x Width
            # Creating array of candidate masks = (Width - budget) number of ICD masks
            candidate_masks = np.expand_dims(icd_mask, 0).repeat(len(candidate_lines),0)

            # Removing the line from icd mask which is to be moved
            candidate_masks[:, :, current_idx] = False 

            # Assigning lines to candidate masks
            for count, candidate_idx in enumerate(candidate_lines): 
                candidate_masks[count, :, candidate_idx] = True # Setting the index to which the line is moved to be true

            if recon=='unet':
                # Performing batched reconstruction on all candidate masks
                img_recons = unet_recon_sense_batched(ksp,mps,candidate_masks,model, device=device,dataset=dataset) # UNET Reconstruction
            else:
                img_recons = modl_recon_batched(ksp,mps,candidate_masks,model, device, device=device,dataset=dataset) # MoDL Reconstruction
            
            if torch.is_tensor(img_recons):
                img_recons = img_recons.cpu().detach().numpy()
            
            # plt.figure()
            # plot_mr_image(img_recons[0])
            # plt.savefig('icd_recon.png')
            # print(compute_nrmse(img_gt,img_recons[0]))

            # breakpoint()

            # Initializing the array for storing nrmse values for candidate masks
            # Length of this numpy array is equal to number of candidate masks generated
            nrmse_candidate_masks = np.zeros((len(candidate_lines))) 

            # Computing NRMSE for all candidate masks
            for count in range(len(candidate_lines)): # Iterating over candidate masks
                nrmse_candidate_masks[count] = compute_nrmse(img_gt,img_recons[count]) # Computing NRMSE

            # End of loop for finding candidate masks
            ####################################################

            # Getting ICD NRMSE and reconstruction using the original ICD mask

            if recon=='unet':
            # Performing batched reconstruction on all candidate masks
                img_recon_icd = unet_recon_sense_batched(ksp,mps,np.expand_dims(icd_mask, 0),model, device=device,dataset=dataset) # UNET Reconstruction
            else:
                img_recons = modl_recon_batched(ksp,mps,np.expand_dims(icd_mask, 0),model, device, device=device,dataset=dataset) # MoDL Reconstruction
            
            if torch.is_tensor(img_recons):
                img_recons = img_recons.cpu().detach().numpy()

            
            nrmse_icd = compute_nrmse(img_gt, img_recon_icd) 

            # Checking if any candidate mask has lesser nrmse than that of original ICD mask
            if np.min(nrmse_candidate_masks) < nrmse_icd: 

                # Finding the index of candidate mask with lowest nrmse
                min_nrmse_mask = np.argmin(nrmse_candidate_masks)

                # Making the candidate mask with minimum nrmse as the new ICD mask
                icd_mask = np.copy(candidate_masks[min_nrmse_mask])

                # Getting ICD NRMSE and reconstruction using the original ICD mask
                img_recon_icd = unet_recon_sense_batched(ksp,mps,np.expand_dims(icd_mask, 0),model, device=device,dataset=dataset)
                nrmse_icd = compute_nrmse(img_gt, img_recon_icd)

                if print_loss:
                    print('Line', current_idx+1, 'moved to', str(min_nrmse_mask+1))

            else:
                if print_loss:
                    print('Line not moved.')
                
            if print_loss:        
                print('NRMSE ICD Mask:',nrmse_icd)
            
            nrmse_icd_list.append(nrmse_icd)

            np.savez(folder_path + 'icd_mask_'+str(us_factor)+'x_'+filename+'.npz',icd_mask=icd_mask[0],nrmse_icd_list=np.array(nrmse_icd_list))

            # Saving Intermediate ICD icd_mask plot
            plt.figure()
            plt.imshow(icd_mask,cmap='gray')
            plt.title('ICD Mask\nNRMSE='+str(round(nrmse_icd,2))+'\n'+filename)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(folder_path + 'icd_mask_'+str(us_factor)+'x_'+filename+'.png')

            # Saving Intermediate nrmse_candidate_masksor vs Iteration plot            
            plt.figure()
            plt.plot(np.array(nrmse_icd_list))
            plt.grid('on')
            plt.xlabel('No. of iterations')
            plt.ylabel('NRMSE')
            plt.title(filename)
            plt.savefig(folder_path + 'nrmse_icd_'+str(us_factor)+'x_'+filename+'.png')

            plt.figure()
            plot_mr_image(img_recon_icd,title='Reconstruction using ICD Mask\nNRMSE='+str(round(nrmse_icd,2))+'\n'+filename)
            plt.savefig(folder_path + 'img_recon_'+str(us_factor)+'x_'+filename+'.png')

            if len(nrmse_icd_list)>3:
                #check whether the last element is greater than the second last element
                if nrmse_icd_list[-1]>nrmse_icd_list[-2]:
                    print('Loss going up for next iteration. Breaking out of loop')
                    break

        #check whether the last element is greater than the second last element
        if nrmse_icd_list[-1]>nrmse_icd_list[-2]:
            print('Loss going up for next iteration. Breaking out of loop')
            break
            

    return icd_mask, np.array(nrmse_icd_list)