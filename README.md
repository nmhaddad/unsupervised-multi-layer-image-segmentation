# Coupled Deep Image Prior Systems

- Nathaniel Haddad
- haddad.na@northeastern.edu
- Northeastern University
- Disclosure: this is an academic project
---

## Abstract

It is well known that deep image priors are capable of learning details of low-level statistics of images and 
videos. Using these statistics under the assumption that images and video are mixtures of multiple layers of 
sub-images, deep image priors are able to successfully decompose images into multiple layers. When coupled 
together, these randomly initialized neural networks are able to perform a number of tasks including denoising, 
image segmentation, and watermark removal. However, it is possible to solve these problems and possibly others 
using many connected deep image priors. In this project, I show that it is possible to create systems of deep 
image priors (more than two) that can be used to break images and videos into many separate layers. Deep image 
prior systems prove to be useful in inverse problems and show some promise in image segmentation problems. 
Furthermore, my work shows that deep image priors can be chained together into systems to capture low-level 
statistics about an image or video that are not captured by single or coupled deep image priors.

[Read the paper](coupled-deep-image-prior-systems.pdf)

## Install

1. Install libaries
    - PyTorch
    - OpenCV
    - Scikit-Image
    - scipy
    
![sketch](media/godzilla.png)

2. To run segmentation experiments, run `python segmentation.py`

![sketch](media/qatar.png)

4. To run dehazing system experiments, run `python dehazing.py`

## References

[1] Gandelsman, Yossi and Shocher, Assaf and Irani, Michal, "Double-DIP": Unsupervised Image Decomposition via Coupled Deep-Image-Priors", 2019 The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

[2] Wang, Jianhua & Wang, Huibin & Gao, Guowei & Lu, Huimin & Zhang, Zhen. (2019). Single Underwater Image Enhancement Based on Lp-norm Decomposition. IEEE Access. PP. 1-1. 10.1109/ACCESS.2019.2945576. 

[3] J. Yang, X. Wang, H. Yue, X. Fu and C. Hou, "Underwater image enhancement based on structure-texture decomposition," 2017 IEEE International Conference on Image Processing (ICIP), Beijing, 2017, pp. 1207-1211.
