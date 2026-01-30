"""
CNN generator for TensorFlow PINN architecture.

This is the default generator using a U-Net-based CNN for object/probe reconstruction.

See also:
    - ptycho/generators/README.md for adding new generators
    - ptycho/model.py for the underlying model implementation
"""


class CnnGenerator:
    """
    CNN-based generator for PINN reconstruction.

    The CNN generator uses a U-Net architecture to reconstruct object and probe
    from diffraction patterns in an unsupervised manner.
    """
    name = 'cnn'

    def __init__(self, config):
        """
        Initialize the CNN generator.

        Args:
            config: TrainingConfig or InferenceConfig with model settings
        """
        self.config = config

    def build_models(self):
        """
        Build the CNN model for training.

        Returns:
            Tuple of (model_instance, diffraction_to_obj) from ptycho.model

        Note:
            This requires update_legacy_dict(params.cfg, config) to have been
            called before invocation to ensure params.cfg is populated.
        """
        from ptycho import model
        return model.create_compiled_model()
