"""
Physics namespace providing access to parameters and TensorFlow operations.

Interface hub for physics-based operations in PtychoPINN. Delegates actual physics
implementations to specialized modules (diffsim, train_pinn) while providing unified
access to physics parameters and TF operations for PINN training workflows.

Public Interface:
    `p.get('nphotons')` - Access physics simulation parameters
    `hh.*` - Physics-aware TensorFlow operations

Architecture: Core physics in diffsim.py, PINN constraints in train_pinn.py
"""
from . import params as p
from . import tf_helper as hh
import tensorflow as tf
import numpy as np
import pdb
