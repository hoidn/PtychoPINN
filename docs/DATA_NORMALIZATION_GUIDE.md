# Data Normalization Guide

## Photon Scaling

`ptycho/diffsim.py` defines `scale_nphotons()` which scales amplitude so the expected
photon count matches `params.cfg['nphotons']`.

This scale is used when generating diffraction patterns and when normalizing inputs
in the loader and training loops.
