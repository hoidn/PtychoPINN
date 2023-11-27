# Physics Constrained Unsupervised Deep Learning for Rapid, High Resolution Scanning Coherent Diffraction Reconstruction

This repository contains the codebase for the methods presented in the paper "[Physics Constrained Unsupervised Deep Learning for Rapid, High Resolution Scanning Coherent Diffraction Reconstruction](https://arxiv.org/pdf/2306.11014.pdf)". 

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


## Usage
`python setup.py install`

```
train.py

usage: PtychoPINN [-h] [--model_type MODEL_TYPE] [--label LABEL]
                  [--positions_provided POSITIONS_PROVIDED] [--data_source DATA_SOURCE] [--set_phi]
                  [--nepochs NEPOCHS] [--offset OFFSET] [--max_position_jitter MAX_POSITION_JITTER]
                  [--output_prefix OUTPUT_PREFIX] [--gridsize GRIDSIZE]
                  [--n_filters_scale N_FILTERS_SCALE] [--object_big OBJECT_BIG]
                  [--intensity_scale_trainable INTENSITY_SCALE_TRAINABLE] [--nll_weight NLL_WEIGHT]
                  [--mae_weight MAE_WEIGHT] [--nimgs_train NIMGS_TRAIN] [--nimgs_test NIMGS_TEST]
                  [--outer_offset_train OUTER_OFFSET_TRAIN] [--outer_offset_test OUTER_OFFSET_TEST]

Generate / load data and train the model

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        model type (pinn or supervised)
  --label LABEL         Name of this run (output directory prefix)
  --positions_provided POSITIONS_PROVIDED
                        [deprecated] Whether nominal or true (nominal + jitter) positions are provided in simulation runs
  --data_source DATA_SOURCE
                        Dataset specification
  --set_phi             If true, simulated objects are given non-zero phase
  --nepochs NEPOCHS     Number of epochs
  --offset OFFSET       Offset
  --max_position_jitter MAX_POSITION_JITTER
                        Solution region is expanded around the edges by this amount
  --output_prefix OUTPUT_PREFIX
                        Output prefix
  --gridsize GRIDSIZE   Grid size
  --n_filters_scale N_FILTERS_SCALE
                        Number of filters scale
  --object_big OBJECT_BIG
                        If true, reconstruct the entire solution region for each set of patterns, instead of just the central N x N region.
  --intensity_scale_trainable INTENSITY_SCALE_TRAINABLE
                        Whether intensity scale is trainable or not
  --nll_weight NLL_WEIGHT
                        Diffraction reconstruction NLL loss weight
  --mae_weight MAE_WEIGHT
                        Diffraction reconstruction MAE loss weight
  --nimgs_train NIMGS_TRAIN
                        Number of generated training images
  --nimgs_test NIMGS_TEST
                        Number of generated testing images
  --outer_offset_train OUTER_OFFSET_TRAIN
                        Scan point grid offset for (generated) training datasets
  --outer_offset_test OUTER_OFFSET_TEST
                        Scan point grid offset for (generated) testing datasets
```

For sample usage, see `scripts/example.sh` or `notebooks/`.

### Checklist
| Status | Task |
|--------|------|
| 🟡 | Reconstruction with non-grid scan patterns |
| 🟡 | Position correction in CDI mode |
| 🟡 | Workflow for experimental data |
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

