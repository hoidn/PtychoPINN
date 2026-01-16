"""
Custom Loss Function Definitions for PtychoPINN Training

This module is intended to contain specialized loss functions for ptychographic reconstruction 
training, including physics-aware losses, symmetry-invariant losses, and domain-specific 
evaluation metrics. Currently contains prototype implementations in commented form.

Intended Functions:
  - I_channel_MAE(): Mean absolute error loss for intensity (amplitude) channels
  - symmetrized_loss(): Coordinate-invariant loss accounting for reconstruction ambiguities
  - Physics-informed losses incorporating ptychographic constraints
  - Multi-scale losses for different reconstruction scales

Architecture Integration:
  This module is designed to provide custom loss functions for integration with the TensorFlow 
  model training pipeline in ptycho.model and ptycho.train_pinn. These losses would complement 
  the standard reconstruction losses with domain-specific knowledge about ptychographic 
  reconstruction constraints and symmetries.

Development Status:
  Currently contains only commented prototype code. The module serves as a placeholder for 
  future loss function development and experimentation. Active loss functions are currently 
  implemented directly in the model architecture and training loops.

Intended Usage Pattern:
  ```python
  # Future intended usage
  from ptycho.losses import symmetrized_loss, I_channel_MAE
  
  # Configure model with custom losses
  model.compile(
      loss={'reconstruction': symmetrized_loss, 
            'intensity': I_channel_MAE},
      optimizer=optimizer
  )
  ```

Symmetry-Aware Loss Design:
  Ptychographic reconstructions suffer from inherent ambiguities including coordinate 
  inversion symmetries. The commented symmetrized_loss() function represents an approach 
  to make training invariant to these physical symmetries by computing losses across 
  multiple symmetric variants and selecting the minimum.

Notes:
  - Module currently contains no active code
  - Commented implementations show intended design patterns
  - Future development should consider integration with existing evaluation metrics
  - Loss functions should be compatible with both PINN and supervised training modes
"""

#def I_channel_MAE(y_true,y_pred, center_target = True):
#    if center_target:
#        y_true = center_channels(y_true
#    return tf.reduce_mean(tf.keras.losses.MeanAbsoluteError(y_true,y_pred))

#def symmetrized_loss(target, pred, loss_fn):
#    """
#    Calculate loss function on an image, taking into account that the
#    prediction may be coordinate-inverted relative to the target
#    """
#    abs1 = (target)
#    abs2 = (pred)
#    abs3 = abs2[:, ::-1, ::-1, :]
#    target_sym = (symmetrize_3d(target))
#    a, b, c = loss_fn(abs1, abs2), loss_fn(abs1, abs3), loss_fn(target_sym, pred)
#    return tf.minimum(a,
#                      tf.minimum(b, c))
