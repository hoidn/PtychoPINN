# Physics constrained machine learning for rapid, high resolution diffractive imaging

This repository has been taken private.

It contained the codebase for the methods presented in the paper "[Physics Constrained Unsupervised Deep Learning for Rapid, High Resolution Scanning Coherent Diffraction Reconstruction](https://www.nature.com/articles/s41598-023-48351-7)". 

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
`pip install .`

## Usage
### Training
`ptycho_train -h `

```
usage: ptycho_train [-h] [--config CONFIG] [--N {64,128,256}] [--gridsize GRIDSIZE] [--n_filters_scale N_FILTERS_SCALE] [--model_type {pinn,supervised}] [--amp_activation {sigmoid,swish,softplus,relu}]
                    [--object_big OBJECT_BIG] [--probe_big PROBE_BIG] [--probe_mask PROBE_MASK] [--pad_object PAD_OBJECT] [--probe_scale PROBE_SCALE]
                    [--gaussian_smoothing_sigma GAUSSIAN_SMOOTHING_SIGMA] [--train_data_file TRAIN_DATA_FILE] [--test_data_file TEST_DATA_FILE] [--batch_size BATCH_SIZE] [--nepochs NEPOCHS]
                    [--mae_weight MAE_WEIGHT] [--nll_weight NLL_WEIGHT] [--realspace_mae_weight REALSPACE_MAE_WEIGHT] [--realspace_weight REALSPACE_WEIGHT] [--nphotons NPHOTONS]
                    [--positions_provided POSITIONS_PROVIDED] [--probe_trainable PROBE_TRAINABLE] [--intensity_scale_trainable INTENSITY_SCALE_TRAINABLE] [--output_dir OUTPUT_DIR]
```

### Inference 
`ptycho_inference -h`

```
usage: ptycho_inference [-h] --model_path MODEL_PATH --test_data TEST_DATA [--config CONFIG] [--output_dir OUTPUT_DIR] [--debug]

Ptychography Inference Script

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to the saved model
  --test_data TEST_DATA
                        Path to the test data file
  --config CONFIG       Optional path to YAML configuration file to override defaults
  --output_dir OUTPUT_DIR
                        Directory for saving output files and images
  --debug               Enable debug mode 
  ```

See examples and READMEs under scripts/.

For an example of interactive (Jupyter) usage, see notebooks/nongrid_simulations.ipynb. If you don't have inputs in the right .npz format you can simulate data.

non_grid_CDI_example.ipynb shows interactive usage using a dataset that is provided with the repo.

### Checklist
| Status | Task |
|--------|------|
| 🟢 | Reconstruction with non-grid scan patterns |
| 🟢 | 128 x 128 resolution |
| 🔴 | Position correction |
| 🔴 | Stochastic probe model |

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

