# Physics constrained machine learning for rapid, high resolution diffractive imaging

This repository contains the codebase for the methods presented in the paper "[Physics Constrained Unsupervised Deep Learning for Rapid, High Resolution Scanning Coherent Diffraction Reconstruction](https://www.nature.com/articles/s41598-023-48351-7)". 

## Overview
PtychoPINN is an unsupervised physics-informed neural network reconstruction method for scanning CDI designed to improve upon the speed of conventional reconstruction methods without sacrificing image quality. Compared to prior NN approaches, the main source of improvements in image quality are its combination of the diffraction forward map with real-space overlap constraints.

## Features
- **Unsupervised / self-supervised learning**: There is no need for extensive labeled training data, making the model more practical to train.
- **Resolution**: PtychoPINN outperforms existing deep learning models for ptychographic reconstruction in terms of image quality, with a 10 dB PSNR increase and a 3- to 6-fold gain in linear resolution. Generalizability and robustness are also improved.
- **Scalability and Speed**: PtychoPINN is two or three orders of magnitude as fast as iterative scanning CDI reconstruction.

![Architecture diagram](diagram/lett.png)
<!---
*Fig. 1: Caption for the figure.*
 -->


## Installation
`python -m pip install .`

## Usage
```
$ train.py

usage: PtychoPINN [-h] [--model_type MODEL_TYPE] [--label LABEL]
                  [--positions_provided POSITIONS_PROVIDED] [--data_source DATA_SOURCE] [--set_phi]
                  [--nepochs NEPOCHS] [--offset OFFSET] [--max_position_jitter MAX_POSITION_JITTER]
                  [--output_prefix OUTPUT_PREFIX] [--gridsize GRIDSIZE]
                  [--n_filters_scale N_FILTERS_SCALE] [--object_big OBJECT_BIG]
                  [--intensity_scale_trainable INTENSITY_SCALE_TRAINABLE] [--nll_weight NLL_WEIGHT]
                  [--mae_weight MAE_WEIGHT] [--nimgs_train NIMGS_TRAIN] [--nimgs_test NIMGS_TEST]
                  [--outer_offset_train OUTER_OFFSET_TRAIN] [--outer_offset_test OUTER_OFFSET_TEST]
```

For interactive usage, see `notebooks/ptycho_lines.ipynb` and `notebooks/non_grid_CDI_example.ipynb`. These demonstrate reconstruction with scanning CDI + grid scan pattern + simulated data and fresnel CDI + random scan pattern + experimental data, respectively.

### Checklist
| Status | Task |
|--------|------|
| 🟢 | Reconstruction with non-grid scan patterns |
| 🟢 | Workflow for experimental data |
| 🟡 | Position correction in CDI mode |
| 🟡 | Probe fitting |
| 🔴 | Stochastic probe model |
| 🔴 | 128 x 128 resolution |

<!-- 
* subpixel convolution (Depth-to-space)
* make the model robust to arbitrary scaling/incorrect normalization of the diffracted intensity
* other ideas: fft based loss, gradient loss, vq-vae https://www.tensorflow.org/tutorials/generative/style_transfer#define_content_and_style_representations
* probe-based vs reconstruction-based support?

* Fully Convolutional Networks for Semantic Segmentation, explore and discuss. Make a slide explaining the idea.
* Try MC Dropout https://arxiv.org/pdf/1511.02680.pdf
* read deep ensembles https://arxiv.org/pdf/1612.01474.pdf

* hard constraint on diffraction norm using projection, consider tf.keras.constraints.MinMaxNorm
* stochastic probe
* probe symmetry consequences
* add an object normalization layer that uses the L2 norm
* how do super resolution models handle high resolutions?
* shift invariance
* grid permutation
* fourier ring correlation

* characterize robustness impact of Poisson likelihood vs. MAE
 -->

