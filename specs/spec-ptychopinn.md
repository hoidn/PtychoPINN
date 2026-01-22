# PtychoPINN Spec

PtychoPINN reconstructs complex-valued objects from diffraction patterns using
physics-informed neural networks. The core implementation is TensorFlow-based
and is located under `ptycho/`.

Key characteristics:
- Unsupervised or self-supervised training via diffraction loss.
- Optional supervised training mode.
- Reconstruction output includes amplitude and phase.
