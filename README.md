# Scan-adaptive Sampling for MRI (SUNO)


Code for reproducing results for the paper:
Siddhant Gautam, Angqi Li, Nicole Seiberlich, Jeffrey A. Fessler, Saiprasad Ravishankar (2024)
"Scan-Adaptive MRI Undersampling Using Neighbor-based Optimization (SUNO)."
arxiv link: 

The data and saved models for running the code can be found at https://drive.google.com/drive/folders/1Ppog0VikG06vLUk6F63gn79pdAeAYL-F.

The MoDL data preprocessing component is inspired by https://github.com/JeffFessler/BLIPSrecon.

Dataset Used: fastMRI https://arxiv.org/abs/1811.08839.

The code is made up of three components: 
* ICD sampling algorithm
* Nearest Neighbor Search
* Testing/Inference

## ICD Sampling algorithm.

Run `get_icd_mask.py` to get ICD mask for a given choice of reconstructor, metric, undersampling factor and initial mask. Default parameters are:
1. Reconstructor: UNet
2. Metric: NRMSE
3. Undersampling factor: 4x
4. Initial Mask: Variable Density Random Sampling (VDRS)


## Nearest Neighbor Search
At inference time, a nearest neighbor search based on proximity to low-frequency k-space reconstructions is used to choose the best scan-specific mask from the training set.
Run `nearest_neighbor_search.py` to get the nearest neighbor predicted mask (also called the SUNO mask).

## Testing

