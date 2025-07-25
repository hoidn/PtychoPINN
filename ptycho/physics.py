"""Physics operations and constraints for physics-informed neural networks.

This module provides the interface and utilities for physics-based operations that are 
fundamental to the physics-informed neural network (PINN) approach in ptychographic 
reconstruction. While many core physics functions have been migrated to specialized 
modules, this module maintains the central physics namespace and imports.

Architecture Role
-----------------
The physics module serves as the central hub for physics-related operations in the PINN 
framework, providing:

- **Physics Namespace**: Central import point for physics-related functionality
- **PINN Integration**: Interface between neural network training and physics constraints
- **Physics Utilities**: Access to core physics parameters and helper functions
- **Legacy Compatibility**: Maintains backward compatibility for existing workflows

Key Physics Concepts
--------------------
The module supports fundamental ptychography physics operations:

**Photon Counting and Noise**:
- Poisson noise modeling for realistic diffraction patterns
- Photon counting operations for intensity normalization
- Statistical sampling from photon distributions

**Forward Physics Model**:
- Integration with differentiable physics simulation (diffsim.py)
- Object illumination and diffraction modeling  
- Complex wave propagation through the sample

**Physics Constraints**:
- Physical parameter bounds and relationships
- Conservation laws in wave propagation
- Coherent imaging constraints

Migration History
-----------------
Physics functions have been reorganized across specialized modules:
- Core physics simulation: `ptycho.diffsim` module
- PINN-specific physics: `ptycho.train_pinn` module  
- Wave propagation helpers: `ptycho.tf_helper` module

This architecture allows for better separation of concerns while maintaining
a unified physics interface.

Integration with PINN Training
------------------------------
The physics module integrates with the PINN training process by:

1. **Forward Model**: Providing access to the differentiable forward physics model
2. **Loss Functions**: Supporting physics-informed loss terms in training
3. **Constraints**: Enforcing physical constraints during optimization
4. **Validation**: Enabling physics-based validation of reconstructions

Usage Examples
--------------
Basic physics module usage:

>>> from ptycho import physics
>>> from ptycho import params as p
>>> 
>>> # Access physics parameters
>>> nphotons = p.get('sim_nphotons')
>>> print(f"Simulated photon count: {nphotons}")
>>>
>>> # Physics module provides namespace for physics operations
>>> # Actual implementations are in specialized modules:
>>> from ptycho.diffsim import observe_amplitude, count_photons
>>> from ptycho.train_pinn import scale_nphotons

Integration with PINN workflow:

>>> import tensorflow as tf
>>> from ptycho import physics
>>> from ptycho.diffsim import simulate_diffraction
>>> from ptycho.train_pinn import physics_loss
>>>
>>> # Physics-informed training integrates physical constraints
>>> def pinn_training_step(model, data, physics_weight=1.0):
>>>     with tf.GradientTape() as tape:
>>>         reconstruction = model(data['diffraction'])
>>>         
>>>         # Standard reconstruction loss
>>>         recon_loss = tf.losses.mse(data['target'], reconstruction)
>>>         
>>>         # Physics-informed loss incorporating forward model
>>>         physics_loss_val = physics_loss(reconstruction, data)
>>>         
>>>         total_loss = recon_loss + physics_weight * physics_loss_val
>>>     
>>>     return total_loss

See Also
--------
- `ptycho.diffsim` : Core physics simulation and forward modeling
- `ptycho.train_pinn` : PINN-specific physics implementations  
- `ptycho.tf_helper` : TensorFlow utilities for physics operations
- `ptycho.model` : Neural network architecture with physics layers
"""
from . import params as p
from . import tf_helper as hh
import tensorflow as tf
import numpy as np
import pdb
