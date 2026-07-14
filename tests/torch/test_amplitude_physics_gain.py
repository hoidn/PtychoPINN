"""Explicit amplitude physics gain (PROBE-RANK-001, design 2026-07-12 §3.3).

The banned flat (B, H, W) probe layout used to multiply the predicted
amplitude by the batch size — an accidental training gain that demonstrably
conditioned amplitude-mode training (amp SSIM 0.486 vs 0.896 at B=16). The
gain survives as ``ModelConfig.amplitude_physics_gain``: an explicit,
batch-size-independent, provenance-carrying constant, plumbed through
``create_training_payload`` -> Lightning hparams -> the amplitude-mode
forward, validated by the scaling contract (finite, > 0; exactly 1.0 for
rectangular_scaled/CI modes, fail-closed).

Contract: docs/specs/spec-ptycho-torch-probe-layout.md.
"""


import numpy as np
import pytest
import torch

from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)

N = 16


@pytest.fixture
def mock_train_npz(tmp_path):
    """Minimal DATA-001-compliant NPZ for create_training_payload."""
    n_images = 8
    npz_path = tmp_path / "mock_train.npz"
    np.savez(
        npz_path,
        diffraction=np.random.rand(n_images, 64, 64).astype(np.float32),
        probeGuess=np.random.rand(64, 64).astype(np.complex64),
        objectGuess=np.random.rand(128, 128).astype(np.complex64),
        xcoords=np.linspace(0, 1, n_images).astype(np.float64),
        ycoords=np.linspace(0, 1, n_images).astype(np.float64),
        scan_index=np.arange(n_images).astype(np.int32),
    )
    return npz_path


@pytest.mark.torch
class TestConfigPlumbing:
    def test_model_config_defaults_to_unit_gain(self):
        assert ModelConfig().amplitude_physics_gain == 1.0

    def test_training_payload_plumbs_gain_and_audits_it(self, mock_train_npz, tmp_path):
        from ptycho_torch.config_factory import create_training_payload

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=tmp_path / "out",
            overrides={"n_groups": 8, "amplitude_physics_gain": 16.0},
        )
        assert payload.pt_model_config.amplitude_physics_gain == 16.0
        assert payload.overrides_applied["amplitude_physics_gain"] == 16.0

    def test_training_payload_audits_default_gain(self, mock_train_npz, tmp_path):
        from ptycho_torch.config_factory import create_training_payload

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=tmp_path / "out",
            overrides={"n_groups": 8},
        )
        assert payload.pt_model_config.amplitude_physics_gain == 1.0
        assert payload.overrides_applied["amplitude_physics_gain"] == 1.0

    def test_gain_lands_in_lightning_hparams(self):
        from ptycho_torch.model import PtychoPINN_Lightning

        module = PtychoPINN_Lightning(
            ModelConfig(
                object_big=False,
                probe_big=False,
                C_model=1,
                C_forward=1,
                amplitude_physics_gain=16.0,
            ),
            DataConfig(N=64, C=1, grid_size=(1, 1)),  # CNN autoencoder needs N>=64
            TrainingConfig(device="cpu", torch_loss_mode="mae"),
            InferenceConfig(),
        )
        assert module.hparams["model_config"]["amplitude_physics_gain"] == 16.0


@pytest.mark.torch
class TestScalingContractValidation:
    """Design §3.3 + §8 case 4: finite, > 0 everywhere; exactly 1.0 for
    rectangular_scaled/CI modes (fail-closed)."""

    @staticmethod
    def _configs(gain, physics_forward_mode="amplitude", **data_overrides):
        data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1), **data_overrides)
        model_cfg = ModelConfig(
            physics_forward_mode=physics_forward_mode,
            amplitude_physics_gain=gain,
        )
        train_cfg = TrainingConfig(torch_loss_mode="poisson")
        return data_cfg, model_cfg, train_cfg

    def test_amplitude_mode_accepts_non_unit_gain(self):
        from ptycho_torch.scaling_contract import validate_scale_contract

        validate_scale_contract(*self._configs(16.0))  # must not raise

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_rejects_nonpositive_or_nonfinite_gain_in_amplitude_mode(self, bad):
        from ptycho_torch.scaling_contract import validate_scale_contract

        with pytest.raises(ValueError, match="amplitude_physics_gain"):
            validate_scale_contract(*self._configs(bad))

    def test_rejects_non_unit_gain_for_rectangular_ci_mode(self):
        from ptycho_torch.scaling_contract import validate_scale_contract

        with pytest.raises(ValueError, match="amplitude_physics_gain"):
            validate_scale_contract(
                *self._configs(16.0, physics_forward_mode="rectangular_scaled")
            )

    def test_rejects_non_unit_gain_for_rectangular_legacy_profile(self):
        from ptycho_torch.scaling_contract import validate_scale_contract

        with pytest.raises(ValueError, match="amplitude_physics_gain"):
            validate_scale_contract(
                *self._configs(
                    16.0,
                    physics_forward_mode="rectangular_scaled",
                    scale_contract_version="legacy_v1",
                    measurement_domain="normalized_amplitude",
                )
            )

    def test_unit_gain_passes_rectangular_mode(self):
        from ptycho_torch.scaling_contract import validate_scale_contract

        resolved = validate_scale_contract(
            *self._configs(1.0, physics_forward_mode="rectangular_scaled")
        )
        assert resolved is not None

    def test_missing_attribute_treated_as_unit_gain(self):
        """Duck-typed configs without the field (pre-fix checkpoints, test
        stand-ins) must resolve to the 1.0 default, not crash."""
        from types import SimpleNamespace

        from ptycho_torch.scaling_contract import validate_amplitude_physics_gain

        legacy = SimpleNamespace(physics_forward_mode="amplitude")
        assert validate_amplitude_physics_gain(legacy) == 1.0


@pytest.mark.torch
class TestForwardApplication:
    """Design §3.3: applied ONCE, multiplicatively, to the predicted
    amplitude inside the amplitude-mode forward; rectangular_scaled path
    untouched; inference (forward_predict) never applies it."""

    @staticmethod
    def _forward_model(gain, **model_overrides):
        from ptycho_torch.model import ForwardModel

        model_cfg = ModelConfig(
            object_big=False,
            C_model=1,
            C_forward=1,
            amplitude_physics_gain=gain,
            **model_overrides,
        )
        data_cfg = DataConfig(N=N, C=1, grid_size=(1, 1))
        return ForwardModel(model_cfg, data_cfg)

    @staticmethod
    def _inputs(batch=2):
        torch.manual_seed(0)
        x = (
            torch.randn(batch, 1, N, N) + 1j * torch.randn(batch, 1, N, N)
        ).to(torch.complex64)
        probe = (
            torch.randn(1, 1, 1, N, N) + 1j * torch.randn(1, 1, 1, N, N)
        ).to(torch.complex64)
        ones = torch.ones(batch, 1, 1, 1)
        eids = torch.zeros(batch, dtype=torch.long)
        return x, probe, ones, eids

    def test_gain_multiplies_amplitude_prediction_exactly_once(self):
        x, probe, ones, eids = self._inputs()
        with torch.no_grad():
            base = self._forward_model(1.0).forward(x, None, None, probe, ones, eids)
            gained = self._forward_model(16.0).forward(x, None, None, probe, ones, eids)
        # 16 is a power of two: the multiply is exact, so bit-equality holds.
        assert torch.equal(gained, 16.0 * base)

    def test_unit_gain_is_bit_identical_noop(self):
        x, probe, ones, eids = self._inputs()
        fm = self._forward_model(1.0)
        with torch.no_grad():
            out = fm.forward(x, None, None, probe, ones, eids)
            # Manual reference chain (probe mask disabled -> ones).
            illuminated = x.unsqueeze(2) * probe
            import ptycho_torch.helper as hh

            ref, _ = hh.pad_and_diffract(illuminated, pad=False)
            ref = fm.scaler.inv_scale(ref, ones)
        assert torch.equal(out, ref)

    def test_gain_is_read_live_from_model_config(self):
        """The sealed-checkpoint tie-back sets the gain on the loaded
        module's (shared) model_config; the forward must honor it without
        reconstruction."""
        x, probe, ones, eids = self._inputs()
        fm = self._forward_model(1.0)
        with torch.no_grad():
            base = fm.forward(x, None, None, probe, ones, eids)
            fm.model_config.amplitude_physics_gain = 4.0
            gained = fm.forward(x, None, None, probe, ones, eids)
        assert torch.equal(gained, 4.0 * base)

    def test_rectangular_scaled_forward_ignores_gain(self):
        """The gain application site must leave the rectangular_scaled chain
        untouched (its contract validator separately rejects non-1.0; this
        pins the forward-site isolation itself)."""
        x, probe, ones, eids = self._inputs()
        rect_kwargs = dict(physics_forward_mode="rectangular_scaled")
        with torch.no_grad():
            base = self._forward_model(1.0, **rect_kwargs).forward(
                x, None, None, probe, ones, eids
            )
            torch.manual_seed(0)  # rect_scaler params are deterministic anyway
            gained = self._forward_model(16.0, **rect_kwargs).forward(
                x, None, None, probe, ones, eids
            )
        assert torch.equal(gained, base)

    def test_inference_forward_predict_never_applies_gain(self):
        from ptycho_torch.model import PtychoPINN

        n_model = 64  # CNN autoencoder needs N>=64
        data_cfg = DataConfig(N=n_model, C=1, grid_size=(1, 1))
        train_cfg = TrainingConfig(device="cpu", torch_loss_mode="mae")

        def build(gain):
            torch.manual_seed(7)
            return PtychoPINN(
                ModelConfig(
                    object_big=False,
                    probe_big=False,
                    C_model=1,
                    C_forward=1,
                    amplitude_physics_gain=gain,
                ),
                data_cfg,
                train_cfg,
            ).eval()

        x = torch.rand(2, 1, n_model, n_model)
        positions = torch.zeros(2, 1, 1, 2)
        probe = torch.ones(1, 1, 1, n_model, n_model, dtype=torch.complex64)
        ones = torch.ones(2, 1, 1, 1)
        with torch.no_grad():
            pred_1 = build(1.0).forward_predict(x, positions, probe, ones)
            pred_16 = build(16.0).forward_predict(x, positions, probe, ones)
        assert torch.equal(pred_1, pred_16)
