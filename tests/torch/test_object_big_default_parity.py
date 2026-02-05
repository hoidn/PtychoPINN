from ptycho.config.config import ModelConfig as TFModelConfig
from ptycho_torch.config_params import ModelConfig as PTModelConfig


def test_object_big_default_matches_tf():
    assert PTModelConfig().object_big == TFModelConfig().object_big
