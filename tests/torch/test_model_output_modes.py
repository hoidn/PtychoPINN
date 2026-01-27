import math
import torch

from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig
from ptycho_torch.model import PtychoPINN


class DummyTwoChannelGenerator(torch.nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        return torch.randn(b, h, w, c, 2, device=x.device, dtype=x.dtype)


def test_amp_phase_logits_bounds():
    model_config = ModelConfig(
        architecture='fno',
        generator_output_mode='amp_phase_logits',
    )
    data_config = DataConfig()
    training_config = TrainingConfig()
    inference_config = InferenceConfig()

    model = PtychoPINN(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        generator=DummyTwoChannelGenerator(),
        generator_output='amp_phase_logits',
    )

    x = torch.randn(2, data_config.C, data_config.N, data_config.N)
    x_complex, amp, phase = model._predict_complex(x)

    assert torch.all(amp >= 0)
    assert torch.all(amp <= 1)
    assert torch.all(phase >= -math.pi)
    assert torch.all(phase <= math.pi)
