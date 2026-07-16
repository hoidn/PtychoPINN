"""
Routing tests for the native inference CLI stitching/scaling knobs (Conformance D4).

The `python -m ptycho_torch.inference` CLI historically parsed/threaded
`patch_weighting` / `varpro_scaling` and then unconditionally stitched with the
uniform `helper.reassemble_patches_position_real` path, silently discarding both
knobs (docs/superpowers/plans/2026-07-14-ci-paper-conformance-audit.md, Theme 2.1).

These tests pin the fixed contract:
- knobs unset  -> legacy uniform path, unchanged (bit-identical back-compat);
- `--patch-weighting probe` and/or `--varpro-scaling` -> reconstruction routes
  through `ptycho_torch.reassembly.reconstruct_image_barycentric` with the knobs
  forwarded on the inference config;
- combinations the CLI path cannot satisfy raise ValueError naming the knob —
  never silent discard.

All heavy functions (factory, bundle loader, dataset build, reassembly) are
stubbed; assertions target routing, not physics. Stubbing patterns mirror
tests/torch/test_cli_inference_torch.py.
"""

import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cli_paths(tmp_path):
    """Model dir with dummy checkpoint, dummy test NPZ, and output dir."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "wts.h5.zip").touch()

    test_file = tmp_path / "test.npz"
    np.savez(test_file, diffraction=np.zeros((4, 8, 8), dtype=np.float32))

    output_dir = tmp_path / "inference_outputs"
    return {
        "model_dir": model_dir,
        "test_file": test_file,
        "output_dir": output_dir,
        "base_args": [
            "--model_path", str(model_dir),
            "--test_data", str(test_file),
            "--output_dir", str(output_dir),
        ],
    }


def _cli_stub_stack():
    """Standard mock set so cli_main runs to the routing decision without IO."""
    mock_factory = MagicMock()
    mock_factory.return_value = MagicMock(
        tf_inference_config=MagicMock(n_groups=32),
        pt_data_config=MagicMock(),
        pt_inference_config=MagicMock(log_patch_stats=False, patch_stats_limit=None),
        execution_config=MagicMock(accelerator="cpu", num_workers=0,
                                   inference_batch_size=None),
    )
    mock_bundle_loader = MagicMock(
        return_value=({"diffraction_to_obj": MagicMock()}, {})
    )
    mock_raw_data = MagicMock()
    return mock_factory, mock_bundle_loader, mock_raw_data


class _ModelStub:
    """Checkpoint-shaped model exposing torch configs (as Lightning bundles do)."""

    def __init__(self):
        from ptycho_torch.config_params import DataConfig, ModelConfig

        self.model_config = ModelConfig(C_model=1, C_forward=1)
        self.data_config = DataConfig(N=64, C=1, grid_size=(1, 1))

    def eval(self):
        return self

    def to(self, device):
        return self


class _BareModelStub:
    """Model without model_config/data_config (legacy/opaque checkpoint)."""

    def eval(self):
        return self

    def to(self, device):
        return self


# ---------------------------------------------------------------------------
# Routing decision unit tests
# ---------------------------------------------------------------------------

class TestResolveReassemblyRoute:
    def test_compatibility_wrapper_delegates_to_policy_resolver(self, monkeypatch):
        from ptycho_torch import inference

        calls = []

        class Policy:
            compatibility_route = "barycentric"

        def resolve(patch_weighting, varpro_scaling):
            calls.append((patch_weighting, varpro_scaling))
            return Policy()

        monkeypatch.setattr(inference, "resolve_cli_reconstruction_policy", resolve)

        assert inference._resolve_reassembly_route("uniform", True) == "barycentric"
        assert calls == [("uniform", True)]

    def test_knobs_unset_resolves_uniform(self):
        from ptycho_torch.inference import _resolve_reassembly_route

        assert _resolve_reassembly_route("uniform", False) == "uniform"

    def test_probe_weighting_resolves_barycentric(self):
        from ptycho_torch.inference import _resolve_reassembly_route

        assert _resolve_reassembly_route("probe", False) == "barycentric"

    def test_varpro_scaling_resolves_barycentric(self):
        from ptycho_torch.inference import _resolve_reassembly_route

        assert _resolve_reassembly_route("uniform", True) == "barycentric"

    def test_unknown_patch_weighting_raises(self):
        from ptycho_torch.inference import _resolve_reassembly_route

        with pytest.raises(ValueError, match="patch_weighting"):
            _resolve_reassembly_route("central_mask", False)

    def test_active_ci_requires_varpro_scaling(self):
        from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig
        from ptycho_torch.inference import _require_ci_varpro_scaling

        model = _ModelStub()
        model.model_config.physics_forward_mode = "rectangular_scaled"
        with pytest.raises(ValueError, match="varpro-scaling"):
            _require_ci_varpro_scaling(
                model,
                PTInferenceConfig(patch_weighting="probe", varpro_scaling=False),
            )

    def test_amplitude_model_does_not_require_varpro_scaling(self):
        from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig
        from ptycho_torch.inference import _require_ci_varpro_scaling

        assert (
            _require_ci_varpro_scaling(
                _ModelStub(),
                PTInferenceConfig(patch_weighting="probe", varpro_scaling=False),
            )
            is None
        )


# ---------------------------------------------------------------------------
# CLI-level routing (argparse/config surface end-to-end to routing decision)
# ---------------------------------------------------------------------------

class TestCliRouting:
    def _run_cli(self, argv, monkeypatch, uniform_helper, barycentric_helper):
        mock_factory, mock_bundle_loader, mock_raw_data = _cli_stub_stack()

        with patch("ptycho_torch.cli.shared.validate_paths", MagicMock()), \
             patch("ptycho_torch.config_factory.create_inference_payload", mock_factory), \
             patch("ptycho_torch.workflows.components.load_inference_bundle_torch", mock_bundle_loader), \
             patch("ptycho.raw_data.RawData.from_file", return_value=mock_raw_data), \
             patch("ptycho_torch.inference._run_inference_and_reconstruct", uniform_helper), \
             patch("ptycho_torch.inference._run_barycentric_inference_and_reconstruct", barycentric_helper), \
             patch("ptycho_torch.inference.save_individual_reconstructions", MagicMock()):
            from ptycho_torch.inference import cli_main

            monkeypatch.setattr("sys.argv", ["inference.py"] + argv)
            exit_code = cli_main()
        return exit_code, mock_factory

    def test_default_flags_route_uniform_path(self, cli_paths, monkeypatch):
        """Knobs unset: the legacy uniform helper runs; barycentric never does."""
        uniform_helper = MagicMock(
            return_value=(np.zeros((8, 8)), np.zeros((8, 8)))
        )
        barycentric_helper = MagicMock()

        argv = cli_paths["base_args"] + ["--n_images", "32", "--quiet"]
        exit_code, _ = self._run_cli(
            argv, monkeypatch, uniform_helper, barycentric_helper
        )

        assert exit_code == 0
        assert uniform_helper.called, "uniform helper must run when knobs are unset"
        assert not barycentric_helper.called, (
            "barycentric path must NOT run when knobs are unset"
        )

    def test_probe_weighting_routes_barycentric(self, cli_paths, monkeypatch):
        """--patch-weighting probe routes to the barycentric helper, knob threaded."""
        uniform_helper = MagicMock()
        barycentric_helper = MagicMock(
            return_value=(np.zeros((8, 8)), np.zeros((8, 8)))
        )

        argv = cli_paths["base_args"] + ["--patch-weighting", "probe", "--quiet"]
        exit_code, mock_factory = self._run_cli(
            argv, monkeypatch, uniform_helper, barycentric_helper
        )

        assert exit_code == 0
        assert barycentric_helper.called, (
            "--patch-weighting probe must route through the barycentric helper"
        )
        assert not uniform_helper.called
        overrides = mock_factory.call_args.kwargs["overrides"]
        assert overrides["patch_weighting"] == "probe"

    def test_varpro_scaling_routes_barycentric(self, cli_paths, monkeypatch):
        """--varpro-scaling routes to the barycentric helper, knob threaded."""
        uniform_helper = MagicMock()
        barycentric_helper = MagicMock(
            return_value=(np.zeros((8, 8)), np.zeros((8, 8)))
        )

        argv = cli_paths["base_args"] + ["--varpro-scaling", "--quiet"]
        exit_code, mock_factory = self._run_cli(
            argv, monkeypatch, uniform_helper, barycentric_helper
        )

        assert exit_code == 0
        assert barycentric_helper.called, (
            "--varpro-scaling must route through the barycentric helper"
        )
        assert not uniform_helper.called
        overrides = mock_factory.call_args.kwargs["overrides"]
        assert overrides["varpro_scaling"] is True

    def test_explicit_n_images_with_barycentric_raises(self, cli_paths, monkeypatch):
        """Explicit --n_images cannot be honored on the full-scan barycentric path."""
        from ptycho_torch.inference import cli_main

        argv = cli_paths["base_args"] + [
            "--patch-weighting", "probe", "--n_images", "16",
        ]
        monkeypatch.setattr("sys.argv", ["inference.py"] + argv)

        with pytest.raises(ValueError, match="n_images"):
            cli_main()


# ---------------------------------------------------------------------------
# Uniform helper still targets hh.reassemble_patches_position_real
# ---------------------------------------------------------------------------

class TestUniformPathUnchanged:
    def test_uniform_helper_calls_position_real(self):
        """The knobs-unset stitching target remains reassemble_patches_position_real."""
        from ptycho.config.config import (
            InferenceConfig,
            ModelConfig,
            PyTorchExecutionConfig,
        )
        from ptycho_torch.inference import _run_inference_and_reconstruct

        raw_data = MagicMock()
        raw_data.diff3d = np.random.rand(4, 64, 64).astype(np.float32)
        raw_data.probeGuess = np.random.rand(64, 64).astype(np.complex64)
        raw_data.xcoords = np.random.rand(4)
        raw_data.ycoords = np.random.rand(4)

        model = MagicMock()
        model.to = MagicMock(return_value=model)
        model.eval = MagicMock(return_value=model)
        patch_complex = torch.complex(
            torch.rand(4, 1, 64, 64), torch.rand(4, 1, 64, 64)
        )
        model.forward_predict = MagicMock(return_value=patch_complex)

        config = InferenceConfig(
            model=ModelConfig(N=64, gridsize=1),
            model_path=Path("outputs/test/bundle.zip"),
            test_data_file=Path("test.npz"),
            backend="pytorch",
            output_dir=Path("outputs/inference"),
            n_groups=4,
        )
        execution_config = PyTorchExecutionConfig(
            accelerator="cpu", num_workers=0, inference_batch_size=None
        )

        def fake_reassemble(patches, offsets, data_cfg, model_cfg, **_kwargs):
            imgs = torch.zeros((1, 64, 64), dtype=patches.dtype)
            return imgs, None, None

        presenter = MagicMock(
            return_value=(np.zeros((64, 64)), np.zeros((64, 64)))
        )
        with patch(
            "ptycho_torch.helper.reassemble_patches_position_real",
            side_effect=fake_reassemble,
        ) as mock_reassemble, patch(
            "ptycho_torch.inference.present_reconstruction_canvas",
            presenter,
        ):
            _run_inference_and_reconstruct(
                model, raw_data, config, execution_config, "cpu", quiet=True
            )

        assert mock_reassemble.called, (
            "knobs-unset path must stitch via hh.reassemble_patches_position_real"
        )
        assert presenter.call_count == 1


# ---------------------------------------------------------------------------
# Barycentric helper: knob forwarding and fail-fast preconditions
# ---------------------------------------------------------------------------

class TestBarycentricHelper:
    def _run_helper(self, model, pt_inference_config, cli_paths):
        from ptycho_torch.inference import (
            _run_barycentric_inference_and_reconstruct,
        )

        canvas = torch.complex(torch.zeros(1, 32, 32), torch.zeros(1, 32, 32))
        mock_recon = MagicMock(return_value=(canvas, MagicMock(), [0.0, 0.0, {}]))
        mock_dataset_cls = MagicMock(return_value=MagicMock())

        presenter = MagicMock(
            return_value=(np.zeros((32, 32)), np.zeros((32, 32)))
        )
        with patch(
            "ptycho_torch.reassembly.reconstruct_image_barycentric", mock_recon
        ), patch("ptycho_torch.dataloader.PtychoDataset", mock_dataset_cls), patch(
            "ptycho_torch.inference.present_reconstruction_canvas", presenter
        ):
            amplitude, phase = _run_barycentric_inference_and_reconstruct(
                model=model,
                test_data_path=cli_paths["test_file"],
                pt_inference_config=pt_inference_config,
                execution_config=None,
                device="cpu",
                output_dir=cli_paths["output_dir"],
                quiet=True,
            )
        return amplitude, phase, mock_recon, mock_dataset_cls, presenter

    def test_forwards_probe_patch_weighting(self, cli_paths):
        from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig

        pt_inference_config = PTInferenceConfig(
            patch_weighting="probe", varpro_scaling=False
        )
        amplitude, phase, mock_recon, _, presenter = self._run_helper(
            _ModelStub(), pt_inference_config, cli_paths
        )

        assert mock_recon.call_count == 1
        forwarded = mock_recon.call_args.args[5]
        assert forwarded.patch_weighting == "probe"
        assert forwarded.varpro_scaling is False
        assert amplitude.shape == (32, 32)
        assert phase.shape == (32, 32)
        assert presenter.call_count == 1

    def test_forwards_varpro_scaling(self, cli_paths):
        from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig

        pt_inference_config = PTInferenceConfig(
            patch_weighting="uniform", varpro_scaling=True
        )
        _, _, mock_recon, _, _ = self._run_helper(
            _ModelStub(), pt_inference_config, cli_paths
        )

        assert mock_recon.call_count == 1
        forwarded = mock_recon.call_args.args[5]
        assert forwarded.varpro_scaling is True

    def test_uses_checkpoint_configs_for_dataset(self, cli_paths):
        """The grouped dataset must be built from the checkpoint's own configs."""
        from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig

        model = _ModelStub()
        pt_inference_config = PTInferenceConfig(
            patch_weighting="probe", varpro_scaling=True
        )
        _, _, mock_recon, mock_dataset_cls, _ = self._run_helper(
            model, pt_inference_config, cli_paths
        )

        dataset_args = mock_dataset_cls.call_args
        assert dataset_args.args[1] is model.model_config
        assert dataset_args.args[2] is model.data_config
        # Staged NPZ directory contains exactly the test file copy.
        staged_dir = Path(dataset_args.args[0])
        assert [p.name for p in staged_dir.iterdir()] == [
            cli_paths["test_file"].name
        ]
        # And the reassembly call receives the same checkpoint configs.
        assert mock_recon.call_args.args[3] is model.data_config
        assert mock_recon.call_args.args[4] is model.model_config

    def test_missing_checkpoint_configs_raises_valueerror(self, cli_paths):
        """Opaque checkpoints cannot satisfy the knobs: fail fast, name the knob."""
        from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig

        pt_inference_config = PTInferenceConfig(
            patch_weighting="uniform", varpro_scaling=True
        )
        with pytest.raises(ValueError, match="varpro_scaling"):
            self._run_helper(_BareModelStub(), pt_inference_config, cli_paths)

    def test_missing_configs_error_names_patch_weighting(self, cli_paths):
        from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig

        pt_inference_config = PTInferenceConfig(
            patch_weighting="probe", varpro_scaling=False
        )
        with pytest.raises(ValueError, match="patch_weighting"):
            self._run_helper(_BareModelStub(), pt_inference_config, cli_paths)
