# Cardiac motion estimation using heart models

Overleaf link for project presentation:
https://www.overleaf.com/7848481722jvdzxhrnrmpy

Utilise (dynamic) heart models to estimate the anatomical changes of the heart during the cardiac cycle. An approach would be developed to accurately estimate cardiac motion from 3D cardiac MR acquisitions. The final aim is to integrate the estimated cardiac motion in the MR image reconstruction algorithm to correct for it.

<img width="1248" alt="Bildschirmfoto 2022-03-04 um 21 06 51" src="https://user-images.githubusercontent.com/17784338/156834100-9cf8d877-fa4a-40c3-bd7d-78f1f53d919b.png">

> **_Note:_** Make sure to used the newest verson of [Morphomatics](morphomatics.github.io):
```bash
pip install -U git+https://github.com/morphomatics/morphomatics.git#egg=morphomatics
```

## Useful links:
Trained network for cardiac segmentation (parameter for trained network are available on the stfc servers): https://github.com/anikpram/cardiac_cine_mri_segmentation For this code SimpleITK 2.0.2 is needed. Use: 'pip install SimpleITK==2.0.2'

ACDC challange (data is also already available on the stfc servers): https://acdc.creatis.insa-lyon.fr/

Usefule tool to visualise multi-dimensional (nifti) data: http://www.itksnap.org/pmwiki/pmwiki.php

## Literature

> Martin Hanik, Hans-Christian Hege, Christoph von Tycowicz:  
> **[A Nonlinear Hierarchical Model for Longitudinal Data on Manifolds.](https://opus4.kobv.de/opus4-zib/files/6117/ZIBReport_16-69.pdf)**  
> IEEE 19th International Symposium on Biomedical Imaging (ISBI), 2022.</br>
> [![Preprint](https://img.shields.io/badge/arXiv-2202.01180-red)](http://arxiv.org/abs/2202.01180)
