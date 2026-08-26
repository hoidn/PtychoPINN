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
from types import SimpleNamespace
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

        self.model_config = ModelConfig()
        self.data_config = DataConfig(N=64, gridsize=1)

    def eval(self):
        return self

    def to(self, device):
        return self


# ---------------------------------------------------------------------------
# CI VarPro scaling guard
# ---------------------------------------------------------------------------

class TestCiVarproScalingGuard:
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
    def _run_cli(self, argv, monkeypatch, barycentric_helper):
        mock_factory, _, _ = _cli_stub_stack()

        with patch("ptycho_torch.cli.shared.validate_paths", MagicMock()), \
             patch("ptycho_torch.config_factory.create_inference_payload", mock_factory), \
             patch("ptycho_torch.inference.reconstruct", barycentric_helper), \
             patch("ptycho_torch.inference.save_individual_reconstructions", MagicMock()):
            from ptycho_torch.inference import cli_main

            monkeypatch.setattr("sys.argv", ["inference.py"] + argv)
            exit_code = cli_main()
        return exit_code, mock_factory

    def test_default_routes_barycentric_kernel(self, cli_paths, monkeypatch):
        """Knobs unset: the barycentric kernel runs by default."""
        barycentric_helper = MagicMock(
            return_value=SimpleNamespace(
                amplitude=np.zeros((8, 8)), phase=np.zeros((8, 8))
            )
        )

        argv = cli_paths["base_args"] + ["--quiet"]
        exit_code, _ = self._run_cli(argv, monkeypatch, barycentric_helper)

        assert exit_code == 0
        assert barycentric_helper.called, (
            "the barycentric kernel must run by default"
        )

    def test_probe_weighting_routes_barycentric(self, cli_paths, monkeypatch):
        """--patch-weighting probe routes to the barycentric helper, knob threaded."""
        barycentric_helper = MagicMock(
            return_value=SimpleNamespace(
                amplitude=np.zeros((8, 8)), phase=np.zeros((8, 8))
            )
        )

        argv = cli_paths["base_args"] + ["--patch-weighting", "probe", "--quiet"]
        exit_code, mock_factory = self._run_cli(argv, monkeypatch, barycentric_helper)

        assert exit_code == 0
        assert barycentric_helper.called, (
            "--patch-weighting probe must route through the barycentric helper"
        )
        overrides = mock_factory.call_args.kwargs["overrides"]
        assert overrides["patch_weighting"] == "probe"

    def test_varpro_scaling_routes_barycentric(self, cli_paths, monkeypatch):
        """--varpro-scaling routes to the barycentric helper, knob threaded."""
        barycentric_helper = MagicMock(
            return_value=SimpleNamespace(
                amplitude=np.zeros((8, 8)), phase=np.zeros((8, 8))
            )
        )

        argv = cli_paths["base_args"] + ["--varpro-scaling", "--quiet"]
        exit_code, mock_factory = self._run_cli(argv, monkeypatch, barycentric_helper)

        assert exit_code == 0
        assert barycentric_helper.called, (
            "--varpro-scaling must route through the barycentric helper"
        )
        overrides = mock_factory.call_args.kwargs["overrides"]
        assert overrides["varpro_scaling"] is True


class TestPublicWorkflowCliDelegation:
    def test_unified_parser_exposes_barycentric_runtime_knobs(self, monkeypatch):
        from scripts.inference import inference

        monkeypatch.setattr(
            "sys.argv",
            [
                "inference.py",
                "--patch-weighting",
                "probe",
                "--varpro-scaling",
                "--groups-per-center",
                "3",
            ],
        )
        args = inference.parse_arguments()
        assert args.patch_weighting == "probe"
        assert args.varpro_scaling is True
        assert args.groups_per_center == 3

    def test_native_barycentric_cli_delegates_without_preloading_bundle(
        self, cli_paths, monkeypatch
    ):
        """The public workflow owns the one strict model load."""
        from types import SimpleNamespace
        from ptycho_torch import inference

        mock_factory, _, _ = _cli_stub_stack()
        public = MagicMock(
            return_value=SimpleNamespace(
                amplitude=np.zeros((8, 8)),
                phase=np.zeros((8, 8)),
            )
        )
        monkeypatch.setattr(
            inference, "reconstruct", public, raising=False
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "inference.py",
                *cli_paths["base_args"],
                "--patch-weighting",
                "probe",
                "--scale-contract-version",
                "legacy_v1",
                "--measurement-domain",
                "normalized_amplitude",
                "--quiet",
            ],
        )
        with patch(
            "ptycho_torch.cli.shared.validate_paths", MagicMock()
        ), patch(
            "ptycho_torch.config_factory.create_inference_payload", mock_factory
        ), patch(
            "ptycho_torch.workflows.bundle_io.load_inference_bundle_torch",
            side_effect=AssertionError("CLI preloaded the bundle"),
        ), patch(
            "ptycho_torch.inference.save_individual_reconstructions", MagicMock()
        ):
            assert inference.cli_main() == 0

        public.assert_called_once()
        assert public.call_args.args[:2] == (
            cli_paths["model_dir"],
            cli_paths["test_file"],
        )
        assert public.call_args.kwargs["scale_contract_version"] == "legacy_v1"
        assert (
            public.call_args.kwargs["measurement_domain"]
            == "normalized_amplitude"
        )

    def test_unified_barycentric_cli_delegates_without_legacy_loader(
        self, tmp_path, monkeypatch
    ):
        from types import SimpleNamespace
        from scripts.inference import inference
        from ptycho_torch import inference as torch_inference

        model_dir = tmp_path / "training"
        model_dir.mkdir()
        test_npz = tmp_path / "test.npz"
        test_npz.touch()
        output_dir = tmp_path / "output"
        config = SimpleNamespace(
            backend="pytorch",
            model_path=model_dir,
            test_data_file=test_npz,
            output_dir=output_dir,
            inference_groups=None,
        )
        args = SimpleNamespace(
            config=None,
            debug_dump=None,
            comparison_plot=False,
            phase_vmin=None,
            phase_vmax=None,
            patch_weighting="probe",
            varpro_scaling=False,
            groups_per_center=1,
            torch_accelerator="cpu",
            torch_num_workers=0,
            torch_inference_batch_size=2,
        )
        execution_request = SimpleNamespace(explicit_fields=frozenset())
        execution_config = SimpleNamespace(
            accelerator="cpu", num_workers=0, inference_batch_size=2,
            precision="32-true",
        )
        runtime = SimpleNamespace(
            config=execution_config,
            notices=(),
            audit_dict=lambda: {},
        )
        public = MagicMock(
            return_value=SimpleNamespace(
                amplitude=np.ones((8, 8)),
                phase=np.zeros((8, 8)),
            )
        )
        monkeypatch.setattr(inference, "parse_arguments", lambda: args)
        monkeypatch.setattr(
            inference, "setup_inference_configuration", lambda *_args: config
        )
        monkeypatch.setattr(
            torch_inference, "reconstruct", public, raising=False
        )
        monkeypatch.setattr(
            "ptycho_torch.cli.shared.build_execution_request_from_args",
            lambda *_args, **_kwargs: execution_request,
        )
        monkeypatch.setattr(
            "ptycho_torch.execution_request.resolve_runtime_execution_request",
            lambda *_args, **_kwargs: runtime,
        )
        from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig
        monkeypatch.setattr(
            "ptycho_torch.config_factory.create_inference_payload",
            lambda **_kwargs: SimpleNamespace(
                pt_inference_config=PTInferenceConfig(
                    patch_weighting="probe", varpro_scaling=False
                ),
                execution_config=execution_config,
            ),
        )

        with patch.object(
            inference,
            "load_inference_bundle_with_backend",
            side_effect=AssertionError("unified CLI preloaded the bundle"),
        ), patch.object(
            inference,
            "load_data",
            side_effect=AssertionError("unified CLI loaded legacy RawData"),
        ), patch.object(
            inference, "save_reconstruction_images"
        ) as save, pytest.raises(SystemExit) as exit_info:
            inference.main()

        assert exit_info.value.code == 0
        public.assert_called_once()
        save.assert_called_once()

    def _installed_door_env(self, tmp_path, monkeypatch, *, config_extra=None,
                            args_extra=None):
        """Common installed-door harness: real dispatcher, stubbed factory IO."""
        from types import SimpleNamespace
        from scripts.inference import inference
        from ptycho_torch import inference as torch_inference
        from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig

        model_dir = tmp_path / "training"
        model_dir.mkdir(exist_ok=True)
        test_npz = tmp_path / "test.npz"
        test_npz.touch()
        config_fields = {
            "backend": "pytorch",
            "model_path": model_dir,
            "test_data_file": test_npz,
            "output_dir": tmp_path / "output",
            "inference_groups": None,
        }
        config_fields.update(config_extra or {})
        config = SimpleNamespace(**config_fields)
        args = SimpleNamespace(
            config=None, debug_dump=None, comparison_plot=False,
            phase_vmin=None, phase_vmax=None,
            patch_weighting="uniform", varpro_scaling=False,
            groups_per_center=1, torch_accelerator="cpu",
            torch_num_workers=0, torch_inference_batch_size=2,
            **(args_extra or {}),
        )
        execution_config = SimpleNamespace(
            accelerator="cpu", num_workers=0, inference_batch_size=2,
            precision="32-true",
        )
        factory = MagicMock(
            return_value=SimpleNamespace(
                pt_inference_config=PTInferenceConfig(),
                execution_config=execution_config,
            )
        )
        monkeypatch.setattr(inference, "parse_arguments", lambda: args)
        monkeypatch.setattr(
            inference, "setup_inference_configuration", lambda *_a: config
        )
        monkeypatch.setattr(
            "ptycho_torch.cli.shared.build_execution_request_from_args",
            lambda *_a, **_k: SimpleNamespace(explicit_fields=frozenset()),
        )
        monkeypatch.setattr(
            "ptycho_torch.config_factory.create_inference_payload", factory
        )
        monkeypatch.setattr(
            torch_inference,
            "reconstruct",
            MagicMock(return_value=SimpleNamespace(
                amplitude=np.ones((8, 8)), phase=np.zeros((8, 8)))),
            raising=False,
        )
        return inference, factory, config

    def test_installed_door_forwards_required_n_groups_default(
        self, tmp_path, monkeypatch
    ):
        """The factory hard-requires n_groups; the dispatcher must supply it.

        Pins the P0 review finding: overrides without n_groups make every
        real --backend pytorch run fail inside create_inference_payload.
        """
        inference, factory, _ = self._installed_door_env(tmp_path, monkeypatch)
        with patch.object(inference, "save_reconstruction_images"), \
             pytest.raises(SystemExit) as exit_info:
            inference.main()
        assert exit_info.value.code == 0
        overrides = factory.call_args.kwargs["overrides"]
        assert overrides["inference_groups"] == 32  # native-door default

    def test_installed_door_forwards_explicit_n_groups(self, tmp_path, monkeypatch):
        inference, factory, _ = self._installed_door_env(
            tmp_path, monkeypatch, config_extra={"inference_groups": 7}
        )
        with patch.object(inference, "save_reconstruction_images"), \
             pytest.raises(SystemExit) as exit_info:
            inference.main()
        assert exit_info.value.code == 0
        assert factory.call_args.kwargs["overrides"]["inference_groups"] == 7

    def test_installed_door_rejects_tf_sampling_flags_on_pytorch(
        self, tmp_path, monkeypatch
    ):
        """--n_subsample/--subsample_seed are TF-door semantics: loud, not dropped."""
        inference, factory, _ = self._installed_door_env(
            tmp_path, monkeypatch, args_extra={"n_subsample": 16}
        )
        with pytest.raises(SystemExit) as exit_info:
            inference.main()
        assert exit_info.value.code == 1
        factory.assert_not_called()
