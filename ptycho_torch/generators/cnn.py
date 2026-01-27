"""
CNN generator for PyTorch PINN architecture.

This is the default generator using a U-Net-based CNN for object/probe reconstruction.

See also:
    - ptycho_torch/generators/README.md for adding new generators
    - ptycho_torch/model.py for the underlying model implementation
"""


class CnnGenerator:
    """
    CNN-based generator for PINN reconstruction (PyTorch/Lightning).

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

    def build_model(self, pt_configs):
        """
        Build the CNN Lightning module for training.

        Args:
            pt_configs: Dict containing PyTorch config objects:
                - model_config: PTModelConfig
                - data_config: PTDataConfig
                - training_config: PTTrainingConfig
                - inference_config: PTInferenceConfig

        Returns:
            PtychoPINN_Lightning model instance
        """
        from ptycho_torch.model import PtychoPINN_Lightning
        return PtychoPINN_Lightning(**pt_configs)
