# Physics constrained machine learning for rapid, high resolution diffractive imaging

This repository contains the codebase for the methods presented in the paper "[Physics Constrained Unsupervised Deep Learning for Rapid, High Resolution Scanning Coherent Diffraction Reconstruction](https://www.nature.com/articles/s41598-023-48351-7)". 

## Overview
PtychoPINN is an unsupervised physics-informed neural network reconstruction method for scanning CDI designed to improve upon the speed of conventional reconstruction methods without sacrificing image quality. Compared to prior NN approaches, the main source of improvements in image quality are its combination of the diffraction forward map with real-space overlap constraints.

## For Developers

Developers looking to contribute to the codebase or understand its deeper architectural principles should first read the **[Unified Developer Guide](./docs/DEVELOPER_GUIDE.md)**. It contains critical information on the project's design, data pipeline, and best practices.

## Features
- **Unsupervised / self-supervised learning**: There is no need for extensive labeled training data, making the model more practical to train.
- **Resolution**: PtychoPINN outperforms existing deep learning models for ptychographic reconstruction in terms of image quality, with a 10 dB PSNR increase and a 3- to 6-fold gain in linear resolution. Generalizability and robustness are also improved.
- **Scalability and Speed**: PtychoPINN is two or three orders of magnitude as fast as iterative scanning CDI reconstruction.

![Architecture diagram](diagram/lett.png)
<!---
*Fig. 1: Caption for the figure.*
 -->


## Installation
`conda create -n ptycho python=3.10`

`conda activate ptycho`

`pip install .`

## Usage
### Training
`ptycho_train --train_data_file <train_path.npz> --test_data_file <test_path.npz> --output_dir <my_run>`

### Evaluation
`ptycho_evaluate --model-dir <my_run> --test-data <test_path.npz> --output-dir <eval_results>`

### Inference 
`ptycho_inference --model_path <my_run> --test_data <test_path.npz> --output_dir <inference_out>`

See examples and READMEs under scripts/.

For an example of interactive (Jupyter) usage, see notebooks/nongrid_simulations.ipynb. If you don't have inputs in the right .npz format you can simulate data.

non_grid_CDI_example.ipynb shows interactive usage using a dataset that is provided with the repo.

### Model Evaluation & Generalization Studies

Run comprehensive generalization studies with statistical robustness:
```bash
# Multi-trial study with uncertainty quantification
./scripts/studies/run_complete_generalization_study.sh \
    --train-sizes "512 1024 2048" \
    --num-trials 3 \
    --output-dir robust_study
```

See `scripts/studies/QUICK_REFERENCE.md` for detailed usage and options.


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

