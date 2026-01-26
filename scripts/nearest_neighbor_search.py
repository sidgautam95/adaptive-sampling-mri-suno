# # Nearest Neighbor Search based icd mask prediction for fastMRI multicoil dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from skimage.metrics import structural_similarity as ssim
import sys
sys.path.append("../utils") 
sys.path.append("../models")
from utils import *
from unet_fbr import Unet
from modl_cg_functions import *
import os

# Function to load numpy files form a directory into a list
def load_files_from_path(path):
    """
    Load all NumPy (.npy) files from a directory, apply center cropping,
    and return them as a single NumPy array.

    Parameters
    ----------
    path : str
        Path to the directory containing .npy files.

    Returns
    -------
    np.ndarray
        Array containing all loaded and cropped images.
    """
    filenames = sorted(os.listdir(path))
    img_list = []

    for fname in filenames:
        file_path = os.path.join(path, fname)

        if not fname.endswith(".npy"):
            continue

        img = np.load(file_path)
        img = crop_img(img, 640, 368)

        img_list.append(img)

    return np.array(img_list)


# path of directory containing the pre-generated aliased images (adjoint reconstruction from low frequency kspace (A^H y_lf)) of training kspace
img_aliased_train_path = '../../train-img-aliased/'

# directory containing the corresponding optimized scan adaptive masks
train_masks_path = '../../train-masks/'

# making list of training aliased images and corresponding optimized scan adaptive masks
img_aliased_train_list = load_files_from_path(img_aliased_train_path)
train_masks_list = load_files_from_path(train_masks_path)

img_aliased_test = np.load('../../img_aliased_test.npy') # loading the test aliased image (adjoint reconstruction from low frequency kspace (A^H y_lf)) of test kspace

nTrain = len(img_aliased_train_list) # no. of training images

# normalizing both training and testing images
for i in range(nTrain):
    img_aliased_train_list[i]=img_aliased_train_list[i]/abs(img_aliased_train_list[i]).max()
    
img_aliased_test=img_aliased_test/abs(img_aliased_test).max()

###################################################################
# choosing metric used to compute nearest neighbors
metric = ['euc-dist']#['ksp-dist','ncc','man-dist']

metric_value = np.zeros((nTrain))

# expanding the testing image into 3D tensor for efficient neighbor finding
img_aliased_test_expanded = np.expand_dims(img_aliased_test,axis=0).repeat(nTrain,0) 
img_aliased_test_expanded = torch.tensor(img_aliased_test_expanded)

# Compare the aliased reconstruction of test kspace with the reconstruction from training kspace
metric_value = torch.linalg.matrix_norm(abs(img_aliased_test_expanded)-abs(img_aliased_train_list), axis=(1,2), ord='fro').detach().numpy()

# sorting and getting the indices in increasing order
neighbor_indices = np.argsort(metric_value) # neighbor indices
nearest_neighbor = neighbor_indices[0] # index of first/best nearest neighbor
suno_mask = train_masks_list[nearest_neighbor] # choosing the SUNO mask

np.save('suno_mask',suno_mask) # Saving the SUNO predicted mask

# Plotting the SUNO mask
plt.figure()
plt.imshow(suno_mask,cmap='gray')
plt.axis('off')
plt.savefig('suno_mask.png')
