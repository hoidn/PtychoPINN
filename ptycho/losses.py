"""
Experimental loss function prototypes for ptychographic reconstruction.

This module contains commented prototype implementations of specialized loss functions
for future ptychographic training workflows. Currently no active functions are exported.

Architecture Role:
    Development workspace for custom loss functions that will integrate with ptycho.model
    training pipelines once implemented and tested.

Current Contents:
    - Commented prototypes for intensity-based and symmetry-aware loss functions
    - No active public API (all functions are commented out)
    - Placeholder for future physics-informed loss implementations

Note: This module currently serves as a development workspace and does not export
any usable functions. Loss computation is handled elsewhere in the codebase.
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
