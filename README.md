# Scan-adaptive Sampling for MRI (SUNO)


Code for reproducing results for the paper:
Siddhant Gautam, Angqi Li, Nicole Seiberlich, Jeffrey A. Fessler, Saiprasad Ravishankar (2024)
"Scan-Adaptive MRI Undersampling Using Neighbor-based Optimization (SUNO)."
arxiv link: 


#### What this code does:
In this paper, we propose a novel approach for jointly learning a set of scan-adaptive Cartesian undersampling patterns along with a reconstructor trained on such undersampling patterns. 
Using a training set consisting of fully sampled $k$-space and corresponding ground truth images, we learn a collection of scan-adaptive sampling masks and a reconstructor from the training data. The joint optimization problem can be formulated as:

    \underset{\bm{\theta},  \mathbf{M}_i \in \mathcal{C},\,i\in\{1,\cdots,N\}   }{\min} \sum_{i=1}^N \| f_{\bm{\theta}} (\mathbf{A}_i^H \mathbf{M}_i \mathbf{y}^{full}_i ) - \mathbf{x}^{gt}_i \|_2^2,
    \label{eq:alternate_min}



**Datasets**: We used the publicly available fastMRI multi-coil knee and brain datasets (https://arxiv.org/abs/1811.08839) which can be downloaded at https://fastmri.med.nyu.edu/. 

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


## Nearest Neighbor Search
At inference time, a nearest neighbor search based on proximity to low-frequency k-space reconstructions is used to choose the best scan-specific mask from the training set.
Run `nearest_neighbor_search.py` to get the nearest neighbor predicted mask (also called the SUNO mask).

## Training and Testing MoDL
Run `train_modl.py` to train MoDL network on a set of training image, undersampled by scan adaptive masks.

Run `test_modl.py` to test the trained MoDL on the SUNO predicted mask.

The MoDL data preprocessing component is inspired by https://github.com/JeffFessler/BLIPSrecon.

**Contact**
The code is provided to support reproducible research. If you have any questions with the code or have some trouble running any module of the framework, you can contact me at gautamsi@msu.edu.
