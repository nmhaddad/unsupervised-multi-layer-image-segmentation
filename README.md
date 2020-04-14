# Coupled Deep Image Prior Systems

- Nathaniel Haddad
- haddad.na@northeastern.edu
- Northeastern University
- Disclosure: this is an academic project
---

## Abstract

It is well known that deep image priors are capable of learning low-level statistics of images and videos and can be 
used as a handcrafted prior for standard inverse problems. Using these statistics under the assumption that images are 
mixtures of multiple layers of sub-images, deep image priors are able to successfully de- compose images into separate 
layers. When coupled together (Double-DIP), deep image priors become even more versatile and can be used for tasks such 
as image dehazing and image segmentation. In this paper, I show that coupled deep image priors can be linked together 
into systems that are able to produce iterative solutions to problems. I introduce two tasks to demonstrate that it is 
possible to create more complex systems of deep image priors via iterative methods and also by increasing the number of 
deep image priors in an existing Double-DIP model. I show that together, these systems can be used solve problems in 
which the resulting layers of image decomposition are many. Furthermore, this paper reinforces claims made about the 
inductive bias of deep image priors and their ability to generalize well to new problems.

[Read the paper](coupled-deep-image-prior-systems.pdf)

![sketch](media/godzilla.png)

![sketch](media/qatar.png)

## Install

1. Install libaries
    - PyTorch
    - OpenCV
    - Scikit-Image
    - scipy
2. To run segmentation experiments, run `python segmentation.py`
4. To run dehazing system experiments, run `python dehazing_system.py`

## References

[1] Gandelsman, Yossi and Shocher, Assaf and Irani, Michal, "Double-DIP": Unsupervised Image Decomposition via Coupled Deep-Image-Priors", 2019 The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

[2] Wang, Jianhua & Wang, Huibin & Gao, Guowei & Lu, Huimin & Zhang, Zhen. (2019). Single Underwater Image Enhancement Based on Lp-norm Decomposition. IEEE Access. PP. 1-1. 10.1109/ACCESS.2019.2945576. 

[3] J. Yang, X. Wang, H. Yue, X. Fu and C. Hou, "Underwater image enhancement based on structure-texture decomposition," 2017 IEEE International Conference on Image Processing (ICIP), Beijing, 2017, pp. 1207-1211.
