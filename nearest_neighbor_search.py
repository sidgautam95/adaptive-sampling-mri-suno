# # Nearest Neighbor Search based icd mask prediction for fastMRI multicoil dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from skimage.metrics import structural_similarity as ssim
import sys
sys.path.insert(1, '/egr/research-slim/gautamsi/mri-sampling')
from utils import *
from unet_fbr import Unet
from modl_utils import *
import os


img_train_path = '/egr/research-slim/shared/fastmri-multicoil/modl-training-data-uncropped/train-img-gt/'
img_test_path = '/egr/research-slim/shared/fastmri-multicoil/modl-training-data-uncropped/test-img-gt/'

img_train_list = load_files_from_path(img_train_path)
img_test_list = load_files_from_path(img_test_path)


# train_masks_list = load_files_from_path(train_masks_path)

train_filenames = os.listdir(img_train_path)
test_filenames = os.listdir(img_test_path)


nTrain = len(img_train_list)
nTest = len(img_test_list)

print('No. of training images:',nTrain)
print('No. of testing images:',nTest)

# normalizing both train and test list
for i in range(nTrain):
    img_train_list[i]=img_train_list[i]/abs(img_train_list[i]).max()
    
for i in range(nTest):
    img_test_list[i]=img_test_list[i]/abs(img_test_list[i]).max()


###################################################################
# Computing nearest neighbors only on magnitude images
metrics = ['ksp-dist']

metric_value=np.zeros((nTrain))

img_test = ...

# # Nearest Neighbor Search based icd mask prediction for fastMRI multicoil dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from skimage.metrics import structural_similarity as ssim
import sys
sys.path.insert(1, '/egr/research-slim/gautamsi/mri-sampling')
from utils import *
from unet_fbr import Unet
from modl_utils import *
import os

# path of directory containing the aliased images (adjoint reconstruction from low frequency kspace) of training and testing kspace
img_recon_train_path = '/egr/research-slim/shared/fastmri-multicoil/nearest-neighbor-prediction-data-fastmri-4x/train-img-aliased/'
img_recon_test_path = '/egr/research-slim/shared/fastmri-multicoil/nearest-neighbor-prediction-data-fastmri-4x/test-img-aliased/'

# directory containing the generated icd masks
train_masks_path = '/egr/research-slim/shared/fastmri-multicoil/nearest-neighbor-prediction-data-fastmri-4x/train-masks/'

img_recon_train_list = load_files_from_path(img_recon_train_path)
img_recon_test_list = load_files_from_path(img_recon_test_path)

img_recon_filenames = os.listdir(img_recon_train_path)

print(img_recon_filenames[:10])


train_filenames = os.listdir(img_recon_train_path)
test_filenames = os.listdir(img_recon_test_path)

nTrain = len(img_recon_train_list)
nTest = len(img_recon_test_list)

print('No. of training images:',nTrain)
print('No. of testing images:',nTest)

# normalizing both train and test list
for i in range(nTrain):
    img_recon_train_list[i]=img_recon_train_list[i]/abs(img_recon_train_list[i]).max()
    
for i in range(nTest):
    img_recon_test_list[i]=img_recon_test_list[i]/abs(img_recon_test_list[i]).max()


###################################################################
# Computing nearest neighbors only on magnitude images
metrics = ['euc-dist']#['ksp-dist','ncc','man-dist']#'euc-dist','ssim',


img_recon_test = img_recon_test_list[test_idx] 

# Compare the aliased reconstruction of test kspace with the reconstruction from training kspace
for i in range(nTrain): # iterative over all training images

    if metric=='euc-dist':
        # Euclidean Distance
        metric_value[i]= np.linalg.norm(abs(img_recon_test)-abs(img_recon_train_list[i]))

    elif metric=='ssim':
        # Structural Similarity Index
        metric_value[i] = ssim(abs(img_recon_train_list[i]),abs(img_recon_test))

    elif metric=='ksp-dist':
        # distance in kspace domain
        ksp_test = MRI_FFT(img_recon_test)
        ksp_train = MRI_FFT(img_recon_train_list[i])

        # normalizing training and test kspace
        ksp_test = ksp_test/np.linalg.norm(ksp_test)
        ksp_train = ksp_train/np.linalg.norm(ksp_train)

        metric_value[i]=np.linalg.norm(abs(ksp_test)-abs(ksp_train))
        
    elif metric=='ncc':
        # Normalized Cross-Correlation
        metric_value[i] = compute_cross_correlation(abs(img_recon_test),abs(img_recon_train_list[i]))

    elif metric=='man-dist': # Manhattan or L1 distance
        metric_value[i]= np.sum(abs(abs(img_recon_test)-abs(img_recon_train_list[i])))

# sorting and getting the indices in increasing order
nn_mask_indices = np.argsort(metric_value)

# choosing the first nearest neighbor
pred_scan = img_recon_filenames[nn_mask_indices[0]][18:29]
pred_slc_idx = img_recon_filenames[nn_mask_indices[0]][33:-4]

pred_nearest_neighbor = np.load(img_recon_train_path+'train_img_aliased_'+pred_scan+'_slc'+str(pred_slc_idx)+'.npy')
pred_icd_mask = np.load(train_masks_path + 'train_mask_'+pred_scan+'_slc'+str(pred_slc_idx) + '.npy')

np.save('nn-mask-indices-fastmri-4x/'+metric+'/nn_mask_indices_'+filename[17:],nn_mask_indices)
np.save('pred_icd_mask',pred_icd_mask) 
np.save(pred_nn_folder+metric+'/pred_nearest_neighbor_'+filename[17:],pred_nearest_neighbor) 
