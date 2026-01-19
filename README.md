# High-dimensional_scattering_photodetector
The code of high-dimensional scattering photodetector is available.

Citation for this code and algorithm: 
- Zhengzhong Huang, Xiangcong Xu, Zhen Mu, Junle Qu, Xiaogang Liu, "On-chip Single-shot High-Dimensional Light Decoding".

This repository contains **MATLAB** and **Python** implementations for high-dimensional light detection and computational reconstruction. The framework is designed to simultaneously encode and decode multiple optical degrees of freedom (DOFs). 

The methods leverage scattering-based light encoding combined with computational reconstruction, enabling compact, scalable, and non-interferometric multidimensional sensing. The core idea is to treat complex scattering as a high-dimensional optical encoder, mapping intertwined optical DOFs into spatially multiplexed intensity patterns. Computational algorithms are then used to reconstruct the original multidimensional light field from the measured signals.

**Setup requirement**: 

- MATLAB R2024a with Image Processing Toolbox (Recommended)
  
- Python ≥ 3.8, torch ≥ 1.13

## Data Information

- broadband_dataset.m: Generate scattering datasets with random spectral distribution. Customizing save folder and save tags are required.

- Full_stoke_polarization.m: Generate scattering datasets with random Stokes polarization distribution. Customizing save folder and save tags are required.

- FT.m, IFT.m: 2D Fourier transform.

- polarization_propagation.m: Diffraction of polarized light field.

- Propagator.m: Angular spectrum diffraction.

- polarization_scatter.mat: Complex functions of scattering layer.

- scatter_sphere.mat: Isotropic function of scattering layer.

- mat2txt.py: Transform .mat tags to .txt.

- train.py: Train HSD network. The folders of datasets and corresponding tags(.txt) need customization.

- test.py: Test HSD network. The folders of datasets need customization.

## Data availability statement

The code and datasets are made available to the editor and reviewers during the review period. Access to the dataset by other individuals or institutions may be granted upon reasonable request. Interested parties are kindly requested to contact the data owner via email to coordinate data access.


## Contact Information

If you have any questions, please feel free to contact us (huangzz1098@gmail.com, chmlx@nus.edu.sg).
