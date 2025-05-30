# Scan-adaptive Sampling for MRI (SUNO)
## Scan-Adaptive MRI Undersampling Using Neighbor-based Optimization (SUNO)

Python implementation for the paper:

Siddhant Gautam, Angqi Li, Nicole Seiberlich, Jeffrey A. Fessler, Saiprasad Ravishankar. "Scan-Adaptive MRI Undersampling Using Neighbor-based Optimization (SUNO)." arXiv preprint [arXiv:2501.09799](https://arxiv.org/abs/2501.09799) (2025).

### Alternating training framework:
In this paper, we propose a novel approach for jointly learning a set of scan-adaptive Cartesian undersampling patterns along with a reconstructor trained on such undersampling patterns. 
Using a training set consisting of fully sampled $k$-space and corresponding ground truth images, we learn a collection of scan-adaptive sampling masks and a reconstructor from the training data. The joint optimization problem can be formulated as:


<p align="center">
$$
\underset{\theta, M_i}{\min}
\sum_{i=1}^N
\big\|
  f_{\theta}\big(\mathbf{A}_i^H \,\mathbf{M}_i\, \mathbf{y}^{\mathrm{full}}_i\big)
  - \mathbf{x}^{\mathrm{gt}}_i
\big\|_2^2
$$
</p>

where:

- $\mathbf{M}_i$ is the $i$-th training mask that inserts zeros at non-sampled locations.
- $\mathbf{y}^{\mathrm{full}}_i$ and $\mathbf{x}^{\mathrm{gt}}_i$ are the fully-sampled multi-coil $k$-space data and the corresponding ground-truth image for the $i$-th scan, respectively.
- $N$ is the total number of training images.
- $\mathcal{C}$ is the set of all 1D Cartesian undersampling patterns with a specified sampling budget.
- $\mathbf{A}_i^H$ is the adjoint of the fully-sampled multi-coil MRI measurement operator for the $i$-th training scan.
- $f_{\theta}$ is the reconstruction network parameterized by $\theta$, trained on the set of sampling patterns $\{\mathbf{M}_i\}_{i=1}^N$.

<p align="center">
  <img src="https://raw.githubusercontent.com/sidgautam95/adaptive-sampling-mri-suno/main/figures/icd_alternating.png" alt="alt text" />
</p>

---
### Getting Scan-adaptive masks:

<p align="center">
  <img src="https://raw.githubusercontent.com/sidgautam95/adaptive-sampling-mri-suno/main/figures/mri_train_pipeline.png" alt="alt text" />
</p>

---
### Nearest Neighbor Search:
Given our collection of scan-adaptive ICD sampling patterns obtained from the training process, the task at the test time is to estimate the locations of high-frequency samples in $k$-space based on initially acquired low-frequency information. We use the nearest neighbor search to predict the sampling pattern from the collection of training scans. The nearest neighbor is found by comparing the adjoint reconstruction of the low-frequency test $k$-space and the corresponding low-frequency part of the training $k$-space as follows:

<p align="center">
  <img src="https://raw.githubusercontent.com/sidgautam95/adaptive-sampling-mri-suno/main/figures/neighbor_finding_equation.png" alt="alt text" />
</p>

The test-time pipeline uses the predicted sampling mask from nearest neighbor search, as illustrated below:

<p align="center">
  <img src="https://raw.githubusercontent.com/sidgautam95/adaptive-sampling-mri-suno/main/figures/mri_testing_pipeline_nn.png" alt="alt text" />
</p>

**Datasets**: We used the publicly available fastMRI multi-coil knee and brain datasets (https://arxiv.org/abs/1811.08839) for our experiments, which can be downloaded at https://fastmri.med.nyu.edu/. 

**Saved Models**: The trained model weights for running the code can be found at https://drive.google.com/drive/folders/1Ppog0VikG06vLUk6F63gn79pdAeAYL-F.

The code is made up of three components: 
* Sampling Optimization Algorithm
* Nearest Neighbor Search
* Training and Testing Reconstruction Networks (MoDL and U-Net)

## ICD-based Sampling Optimization algorithm

Run `get_icd_mask.py` to get the ICD mask for a given choice of reconstructor, metric, undersampling factor, and initial mask. The default parameters are:
1. Reconstructor: UNet
2. Metric: NRMSE
3. Undersampling factor: 4x
4. Initial Mask: Variable Density Random Sampling (VDRS)

### Output:

<p align="center">
  <img src="https://raw.githubusercontent.com/sidgautam95/adaptive-sampling-mri-suno/main/figures/icd_recon_4x.png" alt="alt text" />
</p>

## Nearest Neighbor Search
At inference time, a nearest neighbor search based on proximity to low-frequency k-space reconstructions is used to choose the best scan-specific mask from the training set.
Run `nearest_neighbor_search.py` to get the nearest neighbor predicted mask (also called the SUNO mask).

## Training and Testing MoDL

Run `data_preprocessing_modl.py` to preprocess the kspace, maps and ground truth data for MoDL and U-Net training the MoDL network.

Run `train_modl.py` to train the MoDL network (https://ieeexplore.ieee.org/document/8434321) on a set of training images, undersampled by scan adaptive masks.

Run `test_modl.py` to test the trained MoDL on the SUNO-predicted mask.

`modl_cg_functions.py` contains the helper functions for running the conjugate gradient algorithm inside the MoDL framework.

The MoDL data preprocessing component is inspired by https://github.com/JeffFessler/BLIPSrecon.

## Results
### SUNO mask with other baselines at 4x acceleration factor:

<p align="center">
  <img src="https://raw.githubusercontent.com/sidgautam95/adaptive-sampling-mri-suno/main/figures/icd_vs_baseline_masks_fastmri_4x.png" alt="alt text" />
</p>

### Reconstructed Images at 4x:

<p align="center">
  <img src="https://raw.githubusercontent.com/sidgautam95/adaptive-sampling-mri-suno/main/figures/img_recon_all_masks.png" alt="alt text" />
</p>

The SUNO masks outperform the baseline masks at 4x. The bottom row shows the region of interest.

**Contact**  
The code is provided to support reproducible research. If you have any questions about the code or have some trouble running any module of the framework, you can contact me at gautamsi@msu.edu.
