# Physics Constrained Unsupervised Deep Learning for Rapid, High Resolution Scanning Coherent Diffraction Reconstruction

This repository contains the codebase for the methods presented in the paper "[Physics Constrained Unsupervised Deep Learning for Rapid, High Resolution Scanning Coherent Diffraction Reconstruction](https://arxiv.org/pdf/2306.11014.pdf)". 

## Overview
The repository hosts the TensorFlow 2 implementation of PtychoPINN, an unsupervised physics-informed neural network reconstruction method designed to facilitate real-time imaging by bypassing the resolution limitations of optics. PtychoPINN leverages deep learning for faster reconstruction and augments its performance by integrating the diffraction forward map with real-space constraints from overlapping measurements.

## Key Features
- **Unsupervised / self-supervised learning**: There is no need for extensive labeled training data, making the model more practical to train.
- **Enhanced performance**: PtychoPINN outperforms existing deep learning models for ptychographic reconstruction in terms of image quality, evidenced by a 10 dB PSNR increase and a 3- to 6-fold gain in linear resolution.
- **Scalability and Speed**: PtychoPINN is three orders of magnitude faster than iterative scanning CDI reconstruction.


TODOs

* subpixel convolution (Depth-to-space)
* make the model robust to arbitrary scaling/incorrect normalization of the diffracted intensity
* other ideas: fft based loss, gradient loss, vq-vae https://www.tensorflow.org/tutorials/generative/style_transfer#define_content_and_style_representations
* probe-based vs reconstruction-based support?

* consistency check on magnitude constraint (done)
* Fully Convolutional Networks for Semantic Segmentation, explore and discuss. Make a slide explaining the idea.
* Try MC Dropout https://arxiv.org/pdf/1511.02680.pdf
* Put poisson likelihood into data (done)
* read deep ensembles https://arxiv.org/pdf/1612.01474.pdf

* sub-pixel shifting
* hard constraint on diffraction norm using projection, consider tf.keras.constraints.MinMaxNorm
* stochastic probe
* probe symmetry consequences
* add an object normalization layer that uses the L2 norm
* how do super resolution models handle high resolutions?
* shift invariance
* grid permutation
* fourier ring correlation

* characterize robustness impact of Poisson likelihood vs. MAE