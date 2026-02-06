# SUNO: Scan-Adaptive MRI Undersampling Using Neighbor-based Optimization

[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-Paper-blue.svg)](https://ieeexplore.ieee.org/document/11346972)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

> **Official Python Implementation** for the paper  
> **“Scan-Adaptive MRI Undersampling Using Neighbor-based Optimization (SUNO)”**  
> *IEEE Transactions on Computational Imaging, 2026*

**Siddhant Gautam**, Angqi Li, Nicole Seiberlich, Jeffrey A. Fessler, Saiprasad Ravishankar  

📄 **IEEE Xplore:** https://ieeexplore.ieee.org/document/11346972  
🔗 **DOI:** 10.1109/TCI.2026.3653330  

---

## 📖 Overview

This repository implements **SUNO**, a framework for **scan-adaptive MRI undersampling**.

Unlike conventional fixed undersampling strategies, SUNO learns a collection of optimized undersampling masks and selects the most suitable mask for each test scan using a nearest-neighbor search in low-frequency k-space.

SUNO enables:

- **Adaptive undersampling tailored to individual scans**
- Joint optimization of sampling patterns and reconstruction networks
- No retraining required at inference time
- Compatibility with standard deep MRI reconstruction models

---

## ✨ Key Features

- **ICD-based Sampling Optimization** for learning scan-specific masks  
- **Nearest Neighbor Search** for test-time mask prediction  
- Support for multiple reconstruction networks:
  - MoDL  
  - U-Net  
- Complete pipeline for training, mask learning, and inference  

---

## 🧠 Alternating Training Framework

SUNO jointly learns a set of scan-adaptive undersampling masks and a reconstruction network using the following optimization formulation:

\[
\underset{\theta, M_i}{\min}
\sum_{i=1}^N
\big\|
  f_{\theta}\big(\mathbf{A}_i^H \,\mathbf{M}_i\, \mathbf{y}^{\mathrm{full}}_i\big)
  - \mathbf{x}^{\mathrm{gt}}_i
\big\|_2^2
\]

where:

- \( \mathbf{M}_i \) – learned undersampling mask for scan \(i\)  
- \( \mathbf{y}^{\mathrm{full}}_i \) – fully sampled multi-coil k-space  
- \( \mathbf{x}^{\mathrm{gt}}_i \) – ground-truth image  
- \( \mathbf{A}_i^H \) – adjoint MRI measurement operator  
- \( f_{\theta} \) – reconstruction network  

The framework alternates between:

1. Optimizing masks using **iterative coordinate descent (ICD)**  
2. Training the reconstruction network with the updated masks  

---

## 🧩 Test-Time Scan Adaptation

### Nearest-Neighbor Mask Prediction

At inference time, SUNO predicts a scan-specific sampling mask by comparing low-frequency k-space information from the test scan with that of training scans.

The selected mask is then used with a pretrained reconstruction network—**no additional training is required.**

This enables fully adaptive MRI acquisition at test time with minimal overhead.

---

## 📂 Repository Components

The codebase consists of three main modules:

1. **Sampling Optimization Algorithm**  
2. **Nearest Neighbor Search for Mask Selection**  
3. **Training and Testing Pipelines**
   - MoDL-based reconstruction  
   - U-Net-based reconstruction  

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- SigPy

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Datasets and Pretrained Models

### Datasets

Experiments were conducted using the publicly available **fastMRI multi-coil knee and brain datasets**.

- fastMRI paper: https://arxiv.org/abs/1811.08839  
- Dataset download: https://fastmri.med.nyu.edu/

### Pretrained Models

Pretrained models for reproducing the results in the paper are available at:

https://drive.google.com/drive/folders/1Ppog0VikG06vLUk6F63gn79pdAeAYL-F

---

## 🏃 Usage

### 1. ICD-based Sampling Optimization

To compute optimized sampling masks:

```bash
python get_icd_mask.py
```

Default configuration:

- Reconstructor: **U-Net**
- Metric: **NRMSE**
- Undersampling factor: **4x**
- Initialization: **Variable Density Random Sampling (VDRS)**

---

### 2. Nearest Neighbor Search

To predict SUNO masks at test time:

```bash
python nearest_neighbor_search.py
```

This script selects the most appropriate mask from the learned collection using low-frequency k-space similarity.

---

### 3. Training and Testing MoDL

#### Data Preprocessing

```bash
python data_preprocessing_modl.py
```

#### Train MoDL

```bash
python train_modl.py
```

#### Test with SUNO Masks

```bash
python test_modl.py
```

Helper functions for the conjugate gradient solver used inside MoDL are provided in:

```
modl_cg_functions.py
```

The MoDL data preprocessing module is inspired by:

https://github.com/JeffFessler/BLIPSrecon

---

## 📊 Results

SUNO masks consistently outperform baseline fixed sampling strategies such as:

- Equispaced undersampling  
- Variable Density Random Sampling (VDRS)  
- VISTA  

across multiple datasets and reconstruction models.

Please refer to the paper for detailed quantitative and qualitative comparisons.

---

## 📝 Citation

If you use this code or the SUNO framework in your research, please cite:

```bibtex
@article{gautam2026suno,
  title={Scan-Adaptive MRI Undersampling Using Neighbor-based Optimization (SUNO)},
  author={Gautam, Siddhant and Li, Angqi and Seiberlich, Nicole and Fessler, Jeffrey A. and Ravishankar, Saiprasad},
  journal={IEEE Transactions on Computational Imaging},
  year={2026},
  pages={1--13},
  doi={10.1109/TCI.2026.3653330}
}
```

---

## 📬 Contact

For questions regarding the code or paper, please contact:

- **Siddhant Gautam** – gautamsi@msu.edu  
- **Saiprasad Ravishankar** – ravisha3@msu.edu  

---

## License

This project is released under the **MIT License**.

---

This repository is provided to support **reproducible research** in adaptive MRI sampling.
