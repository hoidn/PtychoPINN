import math
import pytest
import torch

from ptycho_torch.config_params import DataConfig, ModelConfig, TrainingConfig, InferenceConfig
from ptycho_torch.model import Autoencoder, PtychoPINN, Ptycho_Supervised


class DummyTwoChannelGenerator(torch.nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        return torch.randn(b, h, w, c, 2, device=x.device, dtype=x.dtype)


class DummyAmpPhaseGenerator(torch.nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        amp = torch.sigmoid(torch.randn(b, c, h, w, device=x.device, dtype=x.dtype))
        phase = math.pi * torch.tanh(torch.randn(b, c, h, w, device=x.device, dtype=x.dtype))
        return amp, phase


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


def test_amp_phase_mode_accepts_tuple():
    model_config = ModelConfig(
        architecture='fno',
        generator_output_mode='amp_phase',
    )
    data_config = DataConfig()
    training_config = TrainingConfig()
    inference_config = InferenceConfig()

    model = PtychoPINN(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        generator=DummyAmpPhaseGenerator(),
        generator_output='amp_phase',
    )

    x = torch.randn(1, data_config.C, data_config.N, data_config.N)
    x_complex, amp, phase = model._predict_complex(x)

    assert amp.shape == phase.shape
    assert torch.is_complex(x_complex)


# --- Task 2.3 (B1): cnn_output_mode -----------------------------------------


def _cnn_configs(cnn_output_mode, mode='Unsupervised'):
    """CDI gridsize=1 configs (real/imag heads share one channel so torch.complex
    combines cleanly and the ScaledTanh box is observable on the primary path)."""
    model_config = ModelConfig(
        architecture='cnn',
        cnn_output_mode=cnn_output_mode,
        mode=mode,
        C_model=1,
        object_big=False,
        probe_big=False,  # Decoder_last returns the ScaledTanh-bounded primary path only
    )
    data_config = DataConfig(N=64, C=1, grid_size=(1, 1))
    training_config = TrainingConfig()
    return model_config, data_config, training_config


def test_cnn_default_output_mode_is_amp_phase():
    """Default CNN keeps fno-stable's amp/phase contract (defaults unchanged)."""
    model_config, data_config, training_config = _cnn_configs('amp_phase')
    assert model_config.cnn_output_mode == 'amp_phase'

    model = PtychoPINN(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
    )
    assert model.generator_output == 'amp_phase'

    x = torch.randn(1, data_config.C, data_config.N, data_config.N)
    x_complex, amp, phase = model._predict_complex(x)
    assert torch.is_complex(x_complex)


def test_cnn_real_imag_unsupervised_combines_torch_complex():
    """Opt-in real_imag: unsupervised CNN returns torch.complex(real, imag) and the
    ScaledTanh box (real in (-0.8, 1.2), imag in (-1.2, 1.2)) is gated in with it."""
    model_config, data_config, training_config = _cnn_configs('real_imag')
    model = PtychoPINN(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
    )
    model.eval()
    assert model.generator_output == 'real_imag'

    x = torch.randn(1, data_config.C, data_config.N, data_config.N)
    with torch.no_grad():
        real, imag = model.autoencoder(x)
        x_complex, amp, phase = model._predict_complex(x)

    assert torch.is_complex(x_complex)
    # x_complex must be exactly torch.complex(real, imag) of the CNN head tuple.
    assert torch.equal(x_complex.real, real)
    assert torch.equal(x_complex.imag, imag)

    # ScaledTanh gating (Amendment #11): amp branch -> real via tanh+0.2, phase
    # branch -> imag via 1.2*tanh. Padding contributes zeros (in range).
    assert torch.all(x_complex.real >= -0.8 - 1e-4)
    assert torch.all(x_complex.real <= 1.2 + 1e-4)
    assert torch.all(x_complex.imag >= -1.2 - 1e-4)
    assert torch.all(x_complex.imag <= 1.2 + 1e-4)


def test_cnn_real_imag_supervised_output_unaffected():
    """Amendment #4: real_imag is UNSUPERVISED-ONLY. The supervised path keeps its
    amp/phase combine and its output is byte-identical regardless of cnn_output_mode."""
    _, data_config, training_config = _cnn_configs('amp_phase', mode='Supervised')
    x = torch.randn(1, data_config.C, data_config.N, data_config.N)

    def build(cnn_output_mode):
        torch.manual_seed(20260702)
        model_config, dc, tc = _cnn_configs(cnn_output_mode, mode='Supervised')
        model = Ptycho_Supervised(
            model_config=model_config,
            data_config=dc,
            training_config=tc,
        )
        model.eval()
        return model

    model_amp_phase = build('amp_phase')
    model_real_imag = build('real_imag')

    # The knob is ignored in supervised mode: output contract stays amp_phase.
    assert model_amp_phase.generator_output == 'amp_phase'
    assert model_real_imag.generator_output == 'amp_phase'

    with torch.no_grad():
        c_amp_phase, _, _ = model_amp_phase._predict_complex(x)
        c_real_imag, _, _ = model_real_imag._predict_complex(x)

    # Seeded identical weights + unaffected knob -> byte-identical output.
    assert torch.equal(c_amp_phase, c_real_imag)


# --- Task 2.4 (B2): shared decoder (opt-in) ----------------------------------


def _shared_decoder_configs(cnn_output_mode, C, mode='Unsupervised'):
    """Shared-decoder CNN configs at C=1 (CDI, object_big=False) or C=4
    (ptychography, object_big=True). ``decoder_last_amp_channels=C`` keeps the
    shared head's 2*C split symmetric so both output tensors are (B, C, N, N)."""
    object_big = C > 1
    model_config = ModelConfig(
        architecture='cnn',
        use_shared_decoder=True,
        cnn_output_mode=cnn_output_mode,
        mode=mode,
        C_model=C,
        object_big=object_big,
        decoder_last_amp_channels=C,
        probe_big=False,
    )
    data_config = DataConfig(N=64, C=C, grid_size=(2, 2) if C > 1 else (1, 1))
    return model_config, data_config


def test_use_shared_decoder_defaults_to_false():
    assert ModelConfig().use_shared_decoder is False


def test_autoencoder_default_uses_separate_decoders():
    """Default False leaves the current (separate decoder_amp/decoder_phase)
    architecture untouched -- no decoder_shared submodule is built."""
    model_config = ModelConfig(architecture='cnn')
    data_config = DataConfig()

    autoencoder = Autoencoder(model_config, data_config)

    assert hasattr(autoencoder, 'decoder_amp')
    assert hasattr(autoencoder, 'decoder_phase')
    assert not hasattr(autoencoder, 'decoder_shared')


def test_autoencoder_shared_decoder_opt_in_builds_shared_decoder():
    model_config, data_config = _shared_decoder_configs('amp_phase', C=1)

    autoencoder = Autoencoder(model_config, data_config)

    assert hasattr(autoencoder, 'decoder_shared')
    assert not hasattr(autoencoder, 'decoder_amp')
    assert not hasattr(autoencoder, 'decoder_phase')


@pytest.mark.parametrize("C", [1, 4])
@pytest.mark.parametrize("cnn_output_mode", ["amp_phase", "real_imag"])
def test_shared_decoder_shape_contract(C, cnn_output_mode):
    """Shared decoder emits 2*C_out raw channels split into two (B, C_out, N, N)
    tensors, for both cnn_output_mode settings at C=1 (CDI) and C=4 (ptychography)."""
    model_config, data_config = _shared_decoder_configs(cnn_output_mode, C)
    autoencoder = Autoencoder(model_config, data_config)

    x = torch.randn(2, data_config.C, data_config.N, data_config.N)
    branch1, branch2 = autoencoder(x)

    assert branch1.shape == (2, C, data_config.N, data_config.N)
    assert branch2.shape == (2, C, data_config.N, data_config.N)
    assert torch.isfinite(branch1).all()
    assert torch.isfinite(branch2).all()


@pytest.mark.parametrize("C", [1, 4])
def test_shared_decoder_real_imag_matches_b1_scaledtanh_box(C):
    """Shared decoder + cnn_output_mode='real_imag' must apply the SAME ScaledTanh
    box Task 2.3 (B1) applies on the separate decoder path: real in (-0.8, 1.2),
    imag in (-1.2, 1.2) (Amendment #11)."""
    model_config, data_config = _shared_decoder_configs('real_imag', C)
    autoencoder = Autoencoder(model_config, data_config)
    autoencoder.eval()

    x = torch.randn(2, data_config.C, data_config.N, data_config.N)
    with torch.no_grad():
        real, imag = autoencoder(x)

    assert torch.all(real >= -0.8 - 1e-4)
    assert torch.all(real <= 1.2 + 1e-4)
    assert torch.all(imag >= -1.2 - 1e-4)
    assert torch.all(imag <= 1.2 + 1e-4)


@pytest.mark.parametrize("C", [1, 4])
def test_shared_decoder_amp_phase_matches_configured_activations(C):
    """Shared decoder + cnn_output_mode='amp_phase' (default) must keep the
    configured Decoder_amp/Decoder_phase activations: amp uses
    Amplitude_activation (unbounded-below via silu by default), phase uses
    pi*tanh (bounded to (-pi, pi))."""
    model_config, data_config = _shared_decoder_configs('amp_phase', C)
    autoencoder = Autoencoder(model_config, data_config)
    autoencoder.eval()

    x = torch.randn(2, data_config.C, data_config.N, data_config.N)
    with torch.no_grad():
        amp, phase = autoencoder(x)

    assert torch.all(phase >= -math.pi - 1e-4)
    assert torch.all(phase <= math.pi + 1e-4)
