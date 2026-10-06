# High-dimensional_scattering_photodetector
The code of high-dimensional scattering photodetector is available.

Citation for this code and algorithm: 
- Zhengzhong Huang, Xiangcong Xu, Zhen Mu, Junle Qu, Xiaogang Liu, "Scattering-assisted single-shot decoding of high-dimensional light".

This repository contains **MATLAB** and **Python** implementations for high-dimensional light detection and computational reconstruction. The framework is designed to simultaneously encode and decode multiple optical degrees of freedom (DOFs). 

The methods leverage scattering-based light encoding combined with computational reconstruction, enabling compact, scalable, and non-interferometric multidimensional sensing. The core idea is to treat complex scattering as a high-dimensional optical encoder, mapping intertwined optical DOFs into spatially multiplexed intensity patterns. Computational algorithms are then used to reconstruct the original multidimensional light field from the measured signals.

**Setup requirements**: 

- MATLAB R2024a with Image Processing Toolbox (Recommended)
  
- Python ≥ 3.8, torch ≥ 1.13

## Data Information (Single-pixel reconstruction)

A intensity measurement of size X × Y is mapped to a C-dimensional output vector, where C denotes the number of reconstructed channels.

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

The current scripts expect the following directory structure, relative to the working directory:

Train images folder: dataset/(datasets name)/train  
Test images folder: dataset/(datasets name)/test  
Train label folder: dataset/(datasets name)/train.mat(train.txt)  
Test label folder: dataset/(datasets name)/test.mat(test.txt)  

All train file need to be the same path with 'dataset' folder. Run the Python scripts from the directory containing the 'dataset' folder. Measurement images must be single-channel grayscale images, such as **PNG** or **TIFF**. The label-conversion scripts generate 12-digit, zero-padded identifiers, so image filenames should follow the pattern 000000000001.png through the corresponding final sample identifier. Each identifier in the TXT label file must match the image filename without its extension. MATLAB labels are stored as an array of size **N × C**, where **N** is the number of samples. Use mat2txt.py to convert the labels to TXT files after configuring the dataset path and sample counts. Labels are flattened in NumPy **C** order, with the **C** channel values for each spatial pixel stored consecutively.


## Data Information (Snapshot Reconstruction Network) 

The whole HSD-Snapshot code is in 'HSD-Snapshot' folder. The Python snapshot reconstruction network decodes a single scattering-encoded intensity image into a spatially resolved high-dimensional light-field representation. A intensity measurement of size 3840 × 3840 is mapped to a reconstructed cube of size 64 × 64 × C, where C denotes the number of reconstructed channels. An overall downsampling factor is 60.

- network_unet.py: Snapshot reconstruction model, including PixelUnshuffle downsampling, residual blocks, and channel attention.

- dataload.py: Loads single-channel measurements and corresponding flattened labels, normalizes each input image, and returns PyTorch tensors.

- train.py: Train HSD-Snapshot network. The folders of datasets and corresponding tags(.txt) need customization.

- test.py: Test HSD-Snapshot network. The folders of datasets need customization.

- mat2txt.py: Converts supported ordinary MAT and MATLAB v7.3/HDF5 label arrays to flattened text labels.

- mat2txt_73.py: Converts MATLAB v7.3/HDF5 label arrays to flattened text labels.

The current scripts expect the following directory structure, relative to the working directory:

Train images folder: dataset/(datasets name)/train  
Test images folder: dataset/(datasets name)/test  
Train label folder: dataset/(datasets name)/train.mat(train.txt)  
Test label folder: dataset/(datasets name)/test.mat(test.txt)  

All train file need to be the same path with 'dataset' folder. Run the Python scripts from the directory containing the 'dataset' folder. Measurement images must be single-channel grayscale images, such as **PNG** or **TIFF**. The label-conversion scripts generate 12-digit, zero-padded identifiers, so image filenames should follow the pattern 000000000001.png through the corresponding final sample identifier. Each identifier in the TXT label file must match the image filename without its extension. MATLAB labels are stored as an array of size **N × 64 × 64 × C**, where **N** is the number of samples. Use mat2txt.py to convert the labels to TXT files after configuring the dataset path and sample counts. Labels are flattened in NumPy **C** order, with the **C** channel values for each spatial pixel stored consecutively.

## Data availability statement

The code and datasets are made available to the editor and reviewers during the review period. Access to the dataset by other individuals or institutions may be granted upon reasonable request. Interested parties are kindly requested to contact the data owner via email to coordinate data access.


## Contact Information

If you have any questions, please feel free to contact us (huangzz1098@gmail.com, chmlx@nus.edu.sg).
