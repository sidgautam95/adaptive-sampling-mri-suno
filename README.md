# Scan-adaptive Sampling for MRI (SUNO)
## Scan-Adaptive MRI Undersampling Using Neighbor-based Optimization (SUNO)

Python implementation/Code for reproducing results for the paper:

Siddhant Gautam, Angqi Li, Nicole Seiberlich, Jeffrey A. Fessler, Saiprasad Ravishankar (2024), "Scan-Adaptive MRI Undersampling Using Neighbor-based Optimization (SUNO)."

arxiv link: 


#### Alternating training framework:
In this paper, we propose a novel approach for jointly learning a set of scan-adaptive Cartesian undersampling patterns along with a reconstructor trained on such undersampling patterns. 
Using a training set consisting of fully sampled $k$-space and corresponding ground truth images, we learn a collection of scan-adaptive sampling masks and a reconstructor from the training data. The joint optimization problem can be formulated as:


$$
\underset{\theta, M_i}{\min} \sum_{i=1}^N || f_{\theta} ({A}_i^H {M}_i \mathbf{y}^{full}_i ) - \mathbf{x}^{gt}_i ||_2^2
$$

where M is the ith training mask that inserts zeros at non-sampled locations, y_full and x_gt are the ith fully-sampled multi-coil training $k$-space and the corresponding ground truth image, respectively and $N$ is the number of training images. $\mathcal{C}$ is the set of all 1D Cartesian undersampling patterns with a specified sampling budget. $A_i^H$ is the adjoint of the fully-sampled multicoil MRI measurement operator for the ith training scan, and $f_{\theta}$ is the reconstruction network trained on the set of sampling patterns $M_i$'s. 

![alt text](https://github.com/sidgautam95/adaptive-sampling-mri-suno/blob/main/figures/icd_alternating.png)

#### Getting Scan-adaptive masks:
![alt text](https://github.com/sidgautam95/adaptive-sampling-mri-suno/blob/main/figures/mri_train_pipeline.png)

#### Nearest Neighbor Search:
Given our collection of scan-adaptive ICD sampling patterns obtained from the training process, the task at the test time is to estimate the locations of high-frequency samples in $k$-space based on initially acquired low-frequency information. We use the nearest neighbor search to predict the sampling pattern from the collection of training scans. The nearest neighbor is found by comparing the adjoint reconstruction of the low-frequency test $k$-space and the corresponding low-frequency part of the training $k$-space as follows:

$$
d_i =  d(A^H y_test, A^H y^_{train_i}),
$$


![alt text](https://github.com/sidgautam95/adaptive-sampling-mri-suno/blob/main/figures/mri_testing_pipeline_nn.png)

**Datasets**: We used the publicly available fastMRI multi-coil knee and brain datasets (https://arxiv.org/abs/1811.08839) for our experiments, which can be downloaded at https://fastmri.med.nyu.edu/. 

**Saved Models**: The saved models for running the code can be found at https://drive.google.com/drive/folders/1Ppog0VikG06vLUk6F63gn79pdAeAYL-F.

The code is made up of three components: 
* ICD sampling algorithm
* Nearest Neighbor Search
* Training and Testing MoDL

## ICD Sampling algorithm.

Run `get_icd_mask.py` to get ICD mask for a given choice of reconstructor, metric, undersampling factor and initial mask. Default parameters are:
1. Reconstructor: UNet
2. Metric: NRMSE
3. Undersampling factor: 4x
4. Initial Mask: Variable Density Random Sampling (VDRS)

### ICD Output:

![alt text](https://github.com/sidgautam95/adaptive-sampling-mri-suno/blob/main/figures/icd_recon_4x.png)


## Nearest Neighbor Search
At inference time, a nearest neighbor search based on proximity to low-frequency k-space reconstructions is used to choose the best scan-specific mask from the training set.
Run `nearest_neighbor_search.py` to get the nearest neighbor predicted mask (also called the SUNO mask).

## Training and Testing MoDL
Run `train_modl.py` to train the MoDL network (https://ieeexplore.ieee.org/document/8434321) on a set of training images, undersampled by scan adaptive masks.

Run `test_modl.py` to test the trained MoDL on the SUNO-predicted mask.

`modl_cg_functions.py` contains the helper functions for running the conjugate gradient algorithm inside the MoDL framework.

The MoDL data preprocessing component is inspired by https://github.com/JeffFessler/BLIPSrecon.

### Results:
![alt text](https://github.com/sidgautam95/adaptive-sampling-mri-suno/blob/main/figures/icd_vs_baseline_masks_fastmri_4x.png)
![alt text](https://github.com/sidgautam95/adaptive-sampling-mri-suno/blob/main/figures/img_recons_modl_file1001668_slc20_4x_with_lf.png)


**Contact**
The code is provided to support reproducible research. If you have any questions with the code or have some trouble running any module of the framework, you can contact me at gautamsi@msu.edu.
