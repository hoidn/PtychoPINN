"""TDD coverage for the grid-lines reference-performance qualification plumbing.

The reference condition is the executable definition in
tests/torch/test_grid_lines_hybrid_resnet_integration.py: N=128, gridsize/C=1,
Run1084 probe (smoothing 0.5, pad_extrapolate), no mask, Hybrid conv hidden
scale 2.0, central-mask weighting, legacy amplitude physics, MAE, seed 3, five
epochs, historical grid-lines stitching. These tests drive the plumbing with
CPU stubs monkeypatched onto the exact dictionary-loader entry points; no GPU,
no real N=128 data generation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.studies.ablation import runtime, runtime_reference
from scripts.studies.ablation.dataset_provenance import canonical_array_sha256
from scripts.studies.ablation.datasets import DatasetError
from scripts.studies.ablation.runtime_execution import RuntimeExecutionError
from scripts.studies.ablation.runtime_planning import StudyRequestError
from scripts.studies.ablation.verdicts import (
    INTEGRATION_BRIDGE_CONTRACT_SCHEMA_VERSION,
    INTEGRATION_BRIDGE_EVIDENCE_SCHEMA_VERSION,
    IntegrationBridgeEvidence,
    IntegrationBridgeRequirement,
    Verdict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKED_SPEC = REPO_ROOT / "scripts/studies/specs/grid_lines_reference_performance.toml"

HYBRID_ID = "grid_lines_hybrid_resnet_reference"
CNN_ID = "grid_lines_cnn_reference"

RUN1084_ARCHIVE_SHA = "9f82cb9eb2c5a853764b98c1657b778600c0e90425296a7d1fdc6e8fdb53c906"
RUN1084_RAW_ARRAY_SHA = (
    "de564a3ed5e70118fde70d8b65214ddfb3f00364228ef8b7c61a3f31a56c309a"
)
RUN1084_TRANSFORMED_SHA = (
    "eeccb1c92eae6dce36f4102bccda3f814b3eaa16e03e5c805f786edc628d4cd2"
)
# Pinned SHA-256 of the Task 27 gain-16 run2 frozen CNN floors artifact.
CNN_FROZEN_ARTIFACT_SHA = (
    "7b4b1d5b319031094979faa4aeb0c8ef884b23e8b2d21302235795b25caa7346"
)


# ---------------------------------------------------------------------------
# Synthetic miniature recipe + spec fixtures (no GPU, no real data generation)
# ---------------------------------------------------------------------------

N_SMALL = 16
GT_SHAPE = (24, 24)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise AssertionError(f"unsupported TOML test value {value!r}")


def _toml_table(
    name: str, table: dict[str, Any], *, array_of_tables: bool = False
) -> str:
    header = f"[[{name}]]" if array_of_tables else f"[{name}]"
    lines = [header]
    for key, value in table.items():
        if value is None:
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def _write_dataset_files(tmp_path: Path) -> dict[str, Any]:
    """Write a miniature probe archive plus train/test NPZs; return identity."""
    rng = np.random.default_rng(7)
    raw_probe = (rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))).astype(
        np.complex64
    )
    archive = tmp_path / "probe_archive.npz"
    np.savez(archive, probeGuess=raw_probe)

    probe = (
        rng.normal(size=(N_SMALL, N_SMALL)) + 1j * rng.normal(size=(N_SMALL, N_SMALL))
    ).astype(np.complex64)

    def _split(path: Path, count: int, with_truth: bool) -> None:
        payload = {
            "diffraction": rng.random(
                (count, N_SMALL, N_SMALL, 1), dtype=np.float64
            ).astype(np.float32),
            "Y_I": rng.random((count, N_SMALL, N_SMALL, 1)).astype(np.float32),
            "Y_phi": rng.random((count, N_SMALL, N_SMALL, 1)).astype(np.float32),
            "coords_nominal": rng.random((count, 1, 2, 1)).astype(np.float32),
            "probeGuess": probe,
        }
        if with_truth:
            payload["YY_ground_truth"] = (
                rng.normal(size=(1, *GT_SHAPE)) + 1j * rng.normal(size=(1, *GT_SHAPE))
            ).astype(np.complex64)
        np.savez(path, **payload)

    train = tmp_path / "train.npz"
    test = tmp_path / "test.npz"
    _split(train, 6, with_truth=False)
    _split(test, 3, with_truth=True)
    return {
        "archive": archive,
        "archive_sha256": _file_sha256(archive),
        "raw_probe_array_sha256": canonical_array_sha256(raw_probe),
        "transformed_probe_sha256": canonical_array_sha256(probe),
        "train": train,
        "test": test,
        "probe": probe,
    }


def _recipe_table(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "grid_lines_reference_miniature",
        "generator": "grid_lines_workflow_v1",
        "probe_archive": identity["archive"].name,
        "probe_archive_sha256": identity["archive_sha256"],
        "raw_probe_array_sha256": identity["raw_probe_array_sha256"],
        "transformed_probe_sha256": identity["transformed_probe_sha256"],
        "probe_scale_mode": "pad_extrapolate",
        "probe_smoothing_sigma": 0.5,
        "set_phi": True,
        "N": N_SMALL,
        "gridsize": 1,
        "size": 392,
        "offset": 4,
        "outer_offset_train": 8,
        "outer_offset_test": 20,
        "nimgs_train": 2,
        "nimgs_test": 1,
        "nphotons": 1e9,
    }


def _runner_table(architecture: str = "hybrid_resnet") -> dict[str, Any]:
    return {
        "architecture": architecture,
        "N": N_SMALL,
        "gridsize": 1,
        "seed": 3,
        "epochs": 5,
        "batch_size": 16,
        "infer_batch_size": 16,
        "learning_rate": 2e-4,
        "scheduler": "ReduceLROnPlateau",
        "plateau_factor": 0.5,
        "plateau_patience": 2,
        "plateau_min_lr": 1e-4,
        "plateau_threshold": 0.0,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "torch_loss_mode": "mae",
        "generator_output_mode": "real_imag",
        "probe_source": "custom",
        "training_patch_weighting": "central_mask",
        "physics_forward_mode": "amplitude",
        "amplitude_physics_gain": 16.0,
        "hybrid_encoder_conv_hidden_scale": 2.0,
        "position_crop_border": 3,
        "enable_checkpointing": True,
        "logger_backend": "csv",
    }


def _bridge_table(identity: dict[str, Any], **changes: Any) -> dict[str, Any]:
    table = {
        "schema_version": "hybrid_resnet_integration_bridge_v4",
        "bridge_id": HYBRID_ID,
        "ssim_floor_source": "fixture",
        "probe_source_kind": "run1084",
        "loader_kind": "dictionary",
        "raw_probe_archive_sha256": identity["archive_sha256"],
        "raw_probe_array_sha256": identity["raw_probe_array_sha256"],
        "transform_order": ["pad_extrapolate_complex", "smooth_complex"],
        "probe_smoothing_sigma": 0.5,
        "probe_target_n": N_SMALL,
        "transformed_probe_sha256": identity["transformed_probe_sha256"],
        "probe_normalize": True,
        "probe_scale": 4.0,
        "dictionary_probe_normalization_policy": "legacy_passthrough_config_inactive",
        "dictionary_effective_probe_sha256": identity["transformed_probe_sha256"],
        "mmap_probe_normalization_policy": "normalize_probe_like_tf_when_enabled",
        "mmap_same_config_equivalent": False,
        "probe_mask": False,
        "probe_mask_tensor_is_none": True,
        "resolved_probe_mask_kind": "identity",
        "resolved_probe_mask_sha256": canonical_array_sha256(
            np.ones((N_SMALL, N_SMALL), dtype=np.float32)
        ),
        "probe_mask_sigma": 1.0,
        "model_edge_pad": 10,
        "grid_lines_size": 392,
        "grid_lines_n": N_SMALL,
        "grid_lines_gridsize": 1,
        "grid_lines_offset": 4,
        "outer_offset_train": 8,
        "outer_offset_test": 20,
        "nimgs_train": 2,
        "nimgs_test": 1,
        "historical_crop_border": 3,
        "historical_effective_patch_n": 10,
        "generic_position_crop_border": 3,
        "generic_effective_patch_n": 10,
        "crop_boundaries_equivalent": True,
        "hybrid_encoder_conv_hidden_scale": 2.0,
        "architecture": "hybrid_resnet",
        "generator_output_mode": "real_imag",
        "training_patch_weighting": "central_mask",
        "physics_forward_mode": "amplitude",
        "amplitude_physics_gain": 16.0,
        "torch_loss_mode": "mae",
        "seed": 3,
        "epochs": 5,
        "fixture_amp_mae_max": 0.09668590068817139,
        "fixture_phase_mae_max": 0.15318376669684494,
        "fixture_amp_ssim_min": 0.8508652644013688,
        "fixture_phase_ssim_min": 0.9468665959387648,
    }
    table.update(changes)
    return {key: value for key, value in table.items() if value is not None}


def _spec_text(
    identity: dict[str, Any],
    *,
    references: list[dict[str, Any]],
    kind: str = "grid_lines_reference_performance_v1",
) -> str:
    parts = [
        _toml_table("schema", {"kind": kind, "version": 1}),
        _toml_table("study", {"id": "grid_lines_reference_miniature_study"}),
        _toml_table("dataset", _recipe_table(identity)),
    ]
    for reference in references:
        top = {"id": reference["id"]}
        if reference.get("frozen_reference_artifact") is not None:
            top["frozen_reference_artifact"] = reference["frozen_reference_artifact"]
        parts.append(_toml_table("references", top, array_of_tables=True))
        parts.append(_toml_table("references.runner", reference["runner"]))
        parts.append(_toml_table("references.bridge", reference["bridge"]))
        for field, entry in reference.get("expected_differences", {}).items():
            parts.append(_toml_table(f"references.expected_differences.{field}", entry))
    return "\n".join(parts)


def _spec_text_v2(identity: dict[str, Any], reference: dict[str, Any]) -> str:
    parts = [
        _toml_table(
            "schema",
            {"kind": "grid_lines_reference_performance_v2", "version": 2},
        ),
        _toml_table("study", {"id": "grid_lines_reference_miniature_study"}),
        _toml_table(
            "dataset",
            {
                "id": "grid_lines_reference_miniature",
                "generator": "grid_lines_workflow_v1",
                "probe_archive_sha256": identity["archive_sha256"],
                "raw_probe_array_sha256": identity["raw_probe_array_sha256"],
                "transformed_probe_sha256": identity["transformed_probe_sha256"],
            },
        ),
        _toml_table("simulation", {"N": N_SMALL, "seed": 3}),
        _toml_table(
            "simulation.probe",
            {
                "source": "custom",
                "source_path": identity["archive"].name,
                "transform_pipeline": f"pad_extrapolate:{N_SMALL}|smooth:0.5",
            },
        ),
        _toml_table(
            "simulation.object",
            {
                "kind": "lines",
                "image_size": [392, 392],
                "objects_per_probe": 4,
                "diffractions_per_object": 7000,
                "set_phi": True,
            },
        ),
        _toml_table(
            "simulation.scan",
            {
                "kind": "grid",
                "grid_size": [1, 1],
                "offset": 4,
                "outer_offset_train": 8,
                "outer_offset_test": 20,
                "train_groups": 2,
                "test_groups": 1,
                "buffer": 0,
            },
        ),
        _toml_table(
            "simulation.detector",
            {"photons_per_pattern": 1e9},
        ),
        _toml_table("references", {"id": reference["id"]}, array_of_tables=True),
        _toml_table("references.runner", reference["runner"]),
        _toml_table("references.bridge", reference["bridge"]),
    ]
    return "\n".join(parts)


def _write_spec(tmp_path: Path, identity: dict[str, Any], **kwargs: Any) -> Path:
    spec = tmp_path / "reference_spec.toml"
    spec.write_text(_spec_text(identity, **kwargs), encoding="utf-8")
    return spec


def _hybrid_reference(identity: dict[str, Any], **spec_changes: Any) -> dict[str, Any]:
    reference = {
        "id": HYBRID_ID,
        "runner": _runner_table(),
        "bridge": _bridge_table(identity),
        "expected_differences": {},
    }
    reference.update(spec_changes)
    return reference


# ---------------------------------------------------------------------------
# Dictionary-loader entry-point stubs (pattern from test_torch_ablation_runtime)
# ---------------------------------------------------------------------------


class _Harness:
    """Stub the exact dictionary-loader Torch entry points and record calls."""

    CHECKPOINT_BYTES = b"reference-checkpoint-payload"

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        identity: dict[str, Any],
        *,
        metrics: dict[str, tuple[float, float]] | None = None,
        generic_canvas_delta: float = 0.0,
        historical_canvas_shape: tuple[int, int] = GT_SHAPE,
        restored_overrides: dict[str, Any] | None = None,
    ) -> None:
        from scripts.studies import grid_lines_torch_runner as runner_mod
        from ptycho import evaluation, params

        self.calls: list[tuple[Any, ...]] = []
        self.identity = identity
        self.metrics = metrics or {"mae": (0.05, 0.10), "ssim": (0.90, 0.97)}
        self.model = object()
        restored = {
            "architecture": "hybrid_resnet",
            "generator_output_mode": "real_imag",
            "hybrid_encoder_conv_hidden_scale": 2.0,
            "training_patch_weighting": "central_mask",
            "physics_forward_mode": "amplitude",
            "amplitude_physics_gain": 16.0,
            "torch_loss_mode": "mae",
            "seed": 3,
            "epochs": 5,
        }
        restored.update(restored_overrides or {})
        self.selected_model = SimpleNamespace(
            model_config=SimpleNamespace(
                architecture=restored["architecture"],
                generator_output_mode=restored["generator_output_mode"],
                hybrid_encoder_conv_hidden_scale=restored[
                    "hybrid_encoder_conv_hidden_scale"
                ],
                training_patch_weighting=restored["training_patch_weighting"],
                physics_forward_mode=restored["physics_forward_mode"],
                amplitude_physics_gain=restored["amplitude_physics_gain"],
            ),
            training_config=SimpleNamespace(
                epochs=restored["epochs"],
                torch_loss_mode=str(restored["torch_loss_mode"]).upper(),
            ),
            data_config=SimpleNamespace(subsample_seed=restored["seed"]),
            hparams={"generator_output": restored["generator_output_mode"]},
        )
        rng = np.random.default_rng(11)
        self.patches = rng.normal(size=(3, N_SMALL, N_SMALL, 1, 2)).astype(np.float32)
        self.historical_canvas = (
            rng.normal(size=historical_canvas_shape)
            + 1j * rng.normal(size=historical_canvas_shape)
        ).astype(np.complex64)
        self.generic_canvas = self.historical_canvas + np.complex64(
            generic_canvas_delta
        )
        harness = self
        monkeypatch.setattr(
            params,
            "cfg",
            {"offset": 98, "poison": "must-remain-untouched"},
        )

        def load_cached_dataset_with_metadata(npz_path: Path) -> tuple[dict, dict]:
            harness.calls.append(("load_dictionary", Path(npz_path).name))
            with np.load(npz_path) as data:
                loaded = {key: data[key] for key in data.files}
            return loaded, {"additional_parameters": {}}

        def run_torch_training(
            cfg: Any,
            train_data: dict,
            test_data: dict,
            train_metadata: Any = None,
            test_metadata: Any = None,
        ) -> dict[str, Any]:
            harness.calls.append(
                (
                    "run_torch_training",
                    type(train_data).__name__,
                    cfg.architecture,
                    cfg.seed,
                    cfg.epochs,
                    cfg.torch_loss_mode,
                    cfg.training_patch_weighting,
                    cfg.physics_forward_mode,
                    cfg.hybrid_encoder_conv_hidden_scale,
                )
            )
            checkpoint = Path(cfg.output_dir) / "checkpoints" / "best-checkpoint.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(harness.CHECKPOINT_BYTES)
            return {"model": harness.model, "history": {"train_loss": [1.0]}}

        def run_torch_inference(
            model: Any, test_data: dict, cfg: Any, metadata: Any = None
        ) -> np.ndarray:
            harness.calls.append(
                (
                    "run_torch_inference",
                    model is harness.model,
                    model is harness.selected_model,
                )
            )
            return harness.patches

        def reassemble(
            pred: np.ndarray,
            ground_truth: np.ndarray,
            test_data: dict,
            metadata: Any,
            cfg: Any,
        ) -> tuple[np.ndarray, str, dict]:
            harness.calls.append(("reassemble", cfg.reassembly_mode))
            if cfg.reassembly_mode == "grid_lines":
                return harness.historical_canvas.copy(), "grid_lines", {}
            return harness.generic_canvas.copy(), "position", {}

        def eval_reconstruction(
            stitched_obj: np.ndarray, ground_truth_obj: np.ndarray, **kwargs: Any
        ) -> dict[str, Any]:
            harness.calls.append(
                (
                    "eval_reconstruction",
                    stitched_obj.shape,
                    ground_truth_obj.shape,
                    kwargs.get("label"),
                    kwargs.get("trim_offset"),
                )
            )
            return {key: tuple(value) for key, value in harness.metrics.items()}

        monkeypatch.setattr(
            runner_mod,
            "load_cached_dataset_with_metadata",
            load_cached_dataset_with_metadata,
        )
        monkeypatch.setattr(runner_mod, "run_torch_training", run_torch_training)
        monkeypatch.setattr(runner_mod, "run_torch_inference", run_torch_inference)
        monkeypatch.setattr(
            runner_mod, "_reassemble_predictions_for_metrics", reassemble
        )
        monkeypatch.setattr(
            evaluation,
            "eval_reconstruction_explicit",
            eval_reconstruction,
        )
        monkeypatch.setattr(
            "scripts.studies.ablation.runtime_reference_execution.restore_checkpoint_from_hparams",
            lambda checkpoint, **kwargs: harness.selected_model,
        )

    def names(self) -> list[str]:
        return [entry[0] for entry in self.calls]


def _run(
    tmp_path: Path,
    identity: dict[str, Any],
    spec_path: Path,
    *,
    reference: str | None = None,
) -> Any:
    request = runtime_reference.ReferenceQualificationRequest(
        spec=spec_path,
        reference=reference,
        train_npz=identity["train"],
        test_npz=identity["test"],
        output_root=tmp_path / "out",
        base_dir=tmp_path,
    )
    return runtime_reference.run_reference_qualification(request)


# ---------------------------------------------------------------------------
# Checked spec parsing and resolution
# ---------------------------------------------------------------------------


def test_checked_spec_parses_with_both_reference_arms() -> None:
    spec = runtime_reference.load_reference_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    assert {arm.id for arm in spec.arms} == {HYBRID_ID, CNN_ID}
    assert spec.recipe.N == 128
    assert spec.recipe.gridsize == 1
    assert spec.recipe.size == 392
    assert spec.recipe.offset == 4
    assert spec.recipe.outer_offset_train == 8
    assert spec.recipe.outer_offset_test == 20
    assert spec.recipe.nimgs_train == 2
    assert spec.recipe.nimgs_test == 1
    assert spec.recipe.nphotons == 1e9
    assert spec.recipe.probe_smoothing_sigma == 0.5
    assert spec.recipe.probe_scale_mode == "pad_extrapolate"
    assert spec.recipe.probe_archive_sha256 == RUN1084_ARCHIVE_SHA
    assert spec.recipe.raw_probe_array_sha256 == RUN1084_RAW_ARRAY_SHA
    assert spec.recipe.transformed_probe_sha256 == RUN1084_TRANSFORMED_SHA
    assert {arm.runner["amplitude_physics_gain"] for arm in spec.arms} == {16.0}
    assert {arm.bridge["amplitude_physics_gain"] for arm in spec.arms} == {16.0}
    assert {arm.bridge["schema_version"] for arm in spec.arms} == {
        INTEGRATION_BRIDGE_CONTRACT_SCHEMA_VERSION
    }
    fingerprint = spec.recipe.fingerprint_sha256
    assert len(fingerprint) == 64
    assert fingerprint == spec.recipe.fingerprint_sha256
    assert fingerprint == "6d5274b5375c77b6f3038928c6584215f5b81e64b0c4c817d67d656a0150cb7e"
    assert spec.recipe.schema_version == 1
    assert spec.recipe.simulation is not None


def test_schema_v2_reference_reads_top_level_simulation_recipe(tmp_path: Path) -> None:
    identity = _write_dataset_files(tmp_path)
    reference = _hybrid_reference(identity)
    path = tmp_path / "reference_v2.toml"
    path.write_text(_spec_text_v2(identity, reference), encoding="utf-8")

    spec = runtime_reference.load_reference_spec(path, base_dir=tmp_path)

    assert spec.recipe.schema_version == 2
    assert spec.recipe.simulation is not None
    assert spec.recipe.simulation.N == N_SMALL
    assert spec.recipe.simulation.probe.transform_pipeline == (
        f"pad_extrapolate:{N_SMALL}|smooth:0.5"
    )
    assert spec.recipe.simulation.seed == 3
    assert spec.recipe.fingerprint_sha256 != "6d5274b5375c77b6f3038928c6584215f5b81e64b0c4c817d67d656a0150cb7e"


def test_checked_hybrid_arm_locks_exact_fixture_floors() -> None:
    spec = runtime_reference.load_reference_spec(CHECKED_SPEC, base_dir=REPO_ROOT)
    arm = spec.arm(HYBRID_ID)

    assert arm.floor_source == "fixture"
    assert arm.floors_locked
    requirement = runtime_reference.arm_requirement(arm)
    assert isinstance(requirement, IntegrationBridgeRequirement)
    assert requirement.fixture_amp_ssim_min == 0.8508652644013688
    assert requirement.fixture_phase_ssim_min == 0.9468665959387648
    assert requirement.fixture_amp_mae_max == 0.09668590068817139
    assert requirement.fixture_phase_mae_max == 0.15318376669684494
    assert requirement.loader_kind == "dictionary"
    assert requirement.seed == 3
    assert requirement.epochs == 5
    assert requirement.hybrid_encoder_conv_hidden_scale == 2.0
    assert requirement.training_patch_weighting == "central_mask"
    assert requirement.physics_forward_mode == "amplitude"
    assert requirement.torch_loss_mode == "mae"
    assert requirement.amplitude_physics_gain == 16.0


def test_checked_hybrid_runner_operands_match_integration_command() -> None:
    spec = runtime_reference.load_reference_spec(CHECKED_SPEC, base_dir=REPO_ROOT)
    runner_operands = spec.arm(HYBRID_ID).runner

    assert runner_operands["architecture"] == "hybrid_resnet"
    assert runner_operands["N"] == 128
    assert runner_operands["gridsize"] == 1
    assert runner_operands["epochs"] == 5
    assert runner_operands["batch_size"] == 16
    assert runner_operands["infer_batch_size"] == 16
    assert runner_operands["learning_rate"] == 2e-4
    assert runner_operands["scheduler"] == "ReduceLROnPlateau"
    assert runner_operands["plateau_factor"] == 0.5
    assert runner_operands["plateau_patience"] == 2
    assert runner_operands["plateau_min_lr"] == 1e-4
    assert runner_operands["plateau_threshold"] == 0.0
    assert runner_operands["seed"] == 3
    assert runner_operands["optimizer"] == "adam"
    assert runner_operands["weight_decay"] == 0.0
    assert runner_operands["adam_beta1"] == 0.9
    assert runner_operands["adam_beta2"] == 0.999
    assert runner_operands["torch_loss_mode"] == "mae"
    assert runner_operands["amplitude_physics_gain"] == 16.0
    assert runner_operands["generator_output_mode"] == "real_imag"
    assert runner_operands["probe_source"] == "custom"
    assert runner_operands["fno_modes"] == 12
    assert runner_operands["fno_width"] == 32
    assert runner_operands["fno_blocks"] == 4
    assert runner_operands["fno_cnn_blocks"] == 2


def test_checked_cnn_arm_is_frozen_source_with_pinned_floors() -> None:
    spec = runtime_reference.load_reference_spec(CHECKED_SPEC, base_dir=REPO_ROOT)
    arm = spec.arm(CNN_ID)

    assert arm.floor_source == "frozen_reference_artifact"
    assert arm.frozen_reference_artifact is not None
    assert arm.bridge["ssim_floor_reference_artifact_sha256"] == CNN_FROZEN_ARTIFACT_SHA
    requirement = runtime_reference.arm_requirement(arm)
    assert isinstance(requirement, IntegrationBridgeRequirement)
    assert requirement.epochs == 20
    assert requirement.fixture_amp_mae_max == 0.09612537115812302
    assert requirement.fixture_phase_mae_max == 0.21309200960630928
    assert requirement.fixture_amp_ssim_min == 0.8496891595066123
    assert requirement.fixture_phase_ssim_min == 0.9000671199457723
    if arm.frozen_reference_artifact.is_file():
        floors = runtime_reference.load_locked_ssim_floors(arm)
        assert floors is not None
        assert floors.source_artifact_sha256 == CNN_FROZEN_ARTIFACT_SHA


def test_checked_spec_dry_run_stays_free_of_torch_and_tensorflow() -> None:
    code = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from scripts.studies.ablation.runtime_reference import (
    load_reference_spec,
    render_reference_dry_run,
)
spec = load_reference_spec({str(CHECKED_SPEC)!r}, base_dir={str(REPO_ROOT)!r})
plan = render_reference_dry_run(spec)
assert "{HYBRID_ID}" in plan
assert "{CNN_ID}" in plan
blocked = [
    name
    for name in sys.modules
    if name == "torch" or name.startswith("tensorflow") or name == "lightning"
]
assert not blocked, blocked
print("isolated")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("isolated")


def test_dry_run_request_needs_no_npz_and_reports_lock_state(tmp_path: Path) -> None:
    request = runtime_reference.ReferenceQualificationRequest(
        spec=CHECKED_SPEC, dry_run=True, base_dir=REPO_ROOT
    )
    outcome = runtime_reference.run_reference_qualification(request)

    assert outcome.plan is not None
    assert HYBRID_ID in outcome.plan
    assert "floors_locked=true" in outcome.plan
    assert "amplitude_physics_gain=16.0" in outcome.plan
    assert INTEGRATION_BRIDGE_CONTRACT_SCHEMA_VERSION in outcome.plan
    assert INTEGRATION_BRIDGE_EVIDENCE_SCHEMA_VERSION in outcome.plan
    assert outcome.results == ()


def test_cli_main_dry_run_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = runtime_reference.main(["--spec", str(CHECKED_SPEC), "--dry-run"])

    assert code == 0
    assert HYBRID_ID in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Spec validation failures (fail closed at load time)
# ---------------------------------------------------------------------------


def test_spec_rejects_unknown_kind(tmp_path: Path) -> None:
    identity = _write_dataset_files(tmp_path)
    spec = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)], kind="other_v1"
    )
    with pytest.raises(StudyRequestError, match="kind"):
        runtime_reference.load_reference_spec(spec, base_dir=tmp_path)


def test_spec_rejects_duplicate_reference_ids(tmp_path: Path) -> None:
    identity = _write_dataset_files(tmp_path)
    spec = _write_spec(
        tmp_path,
        identity,
        references=[_hybrid_reference(identity), _hybrid_reference(identity)],
    )
    with pytest.raises(StudyRequestError, match="duplicate"):
        runtime_reference.load_reference_spec(spec, base_dir=tmp_path)


def test_spec_rejects_bridge_runner_command_mismatch(tmp_path: Path) -> None:
    identity = _write_dataset_files(tmp_path)
    reference = _hybrid_reference(identity)
    reference["bridge"]["seed"] = 4
    spec = _write_spec(tmp_path, identity, references=[reference])
    with pytest.raises(StudyRequestError, match="seed"):
        runtime_reference.load_reference_spec(spec, base_dir=tmp_path)


@pytest.mark.parametrize("gain", [0.0, -1.0, float("inf"), float("nan")])
def test_spec_rejects_nonpositive_or_nonfinite_reference_gain(
    tmp_path: Path, gain: float
) -> None:
    identity = _write_dataset_files(tmp_path)
    reference = _hybrid_reference(identity)
    reference["runner"]["amplitude_physics_gain"] = gain
    reference["bridge"]["amplitude_physics_gain"] = gain
    spec = _write_spec(tmp_path, identity, references=[reference])

    with pytest.raises(StudyRequestError, match="amplitude_physics_gain"):
        runtime_reference.load_reference_spec(spec, base_dir=tmp_path)


def test_amplitude_reference_arm_requires_calibrated_gain_16(tmp_path: Path) -> None:
    identity = _write_dataset_files(tmp_path)
    reference = _hybrid_reference(identity)
    reference["runner"]["amplitude_physics_gain"] = 4.0
    reference["bridge"]["amplitude_physics_gain"] = 4.0
    spec = _write_spec(tmp_path, identity, references=[reference])

    with pytest.raises(StudyRequestError, match="exactly 16"):
        runtime_reference.load_reference_spec(spec, base_dir=tmp_path)


def test_spec_rejects_bridge_recipe_geometry_mismatch(tmp_path: Path) -> None:
    identity = _write_dataset_files(tmp_path)
    reference = _hybrid_reference(identity)
    reference["bridge"]["grid_lines_size"] = 400
    spec = _write_spec(tmp_path, identity, references=[reference])
    with pytest.raises(StudyRequestError, match="grid_lines_size"):
        runtime_reference.load_reference_spec(spec, base_dir=tmp_path)


def test_spec_rejects_fixture_arm_with_frozen_artifact(tmp_path: Path) -> None:
    identity = _write_dataset_files(tmp_path)
    reference = _hybrid_reference(identity, frozen_reference_artifact="floors.json")
    spec = _write_spec(tmp_path, identity, references=[reference])
    with pytest.raises(StudyRequestError, match="fixture"):
        runtime_reference.load_reference_spec(spec, base_dir=tmp_path)


def test_spec_rejects_runner_path_operands(tmp_path: Path) -> None:
    identity = _write_dataset_files(tmp_path)
    reference = _hybrid_reference(identity)
    reference["runner"]["train_npz"] = "sneaky.npz"
    spec = _write_spec(tmp_path, identity, references=[reference])
    with pytest.raises(StudyRequestError, match="train_npz"):
        runtime_reference.load_reference_spec(spec, base_dir=tmp_path)


def test_spec_rejects_unknown_expected_difference_field(tmp_path: Path) -> None:
    identity = _write_dataset_files(tmp_path)
    reference = _hybrid_reference(
        identity,
        expected_differences={
            "seed": {"classification": "harmless", "justification": "never allowed"}
        },
    )
    spec = _write_spec(tmp_path, identity, references=[reference])
    with pytest.raises(StudyRequestError, match="seed"):
        runtime_reference.load_reference_spec(spec, base_dir=tmp_path)


# ---------------------------------------------------------------------------
# Reference dataset identity (recipe-pinned, content-fingerprinted)
# ---------------------------------------------------------------------------


def test_reference_npz_pair_validates_and_fingerprints(tmp_path: Path) -> None:
    from scripts.studies.ablation.datasets import (
        parse_grid_lines_reference_recipe,
        validate_reference_npz_pair,
    )

    identity = _write_dataset_files(tmp_path)
    recipe = parse_grid_lines_reference_recipe(
        "grid_lines_reference_miniature", _recipe_table(identity), base_dir=tmp_path
    )
    materialized = validate_reference_npz_pair(
        recipe, identity["train"], identity["test"]
    )

    assert materialized.recipe_fingerprint_sha256 == recipe.fingerprint_sha256
    assert materialized.train_sha256 == _file_sha256(identity["train"])
    assert materialized.test_sha256 == _file_sha256(identity["test"])
    assert materialized.probe_sha256 == identity["transformed_probe_sha256"]
    assert materialized.n_train == 6
    assert materialized.n_test == 3


def test_reference_npz_pair_rejects_probe_identity_mismatch(tmp_path: Path) -> None:
    from scripts.studies.ablation.datasets import (
        parse_grid_lines_reference_recipe,
        validate_reference_npz_pair,
    )

    identity = _write_dataset_files(tmp_path)
    table = _recipe_table(identity)
    table["transformed_probe_sha256"] = "a" * 64
    recipe = parse_grid_lines_reference_recipe(
        "grid_lines_reference_miniature", table, base_dir=tmp_path
    )
    with pytest.raises(DatasetError, match="probe"):
        validate_reference_npz_pair(recipe, identity["train"], identity["test"])


def test_reference_npz_pair_rejects_wrong_detector_size(tmp_path: Path) -> None:
    from scripts.studies.ablation.datasets import (
        parse_grid_lines_reference_recipe,
        validate_reference_npz_pair,
    )

    identity = _write_dataset_files(tmp_path)
    table = _recipe_table(identity)
    table["N"] = 32
    recipe = parse_grid_lines_reference_recipe(
        "grid_lines_reference_miniature", table, base_dir=tmp_path
    )
    with pytest.raises(DatasetError):
        validate_reference_npz_pair(recipe, identity["train"], identity["test"])


def test_reference_npz_pair_rejects_tampered_probe_archive(tmp_path: Path) -> None:
    from scripts.studies.ablation.datasets import (
        parse_grid_lines_reference_recipe,
        validate_reference_npz_pair,
    )

    identity = _write_dataset_files(tmp_path)
    recipe = parse_grid_lines_reference_recipe(
        "grid_lines_reference_miniature", _recipe_table(identity), base_dir=tmp_path
    )
    identity["archive"].write_bytes(b"tampered")
    with pytest.raises(DatasetError, match="archive"):
        validate_reference_npz_pair(recipe, identity["train"], identity["test"])


# ---------------------------------------------------------------------------
# Dictionary-loader execution flow (stubbed Torch entry points)
# ---------------------------------------------------------------------------


def test_reference_run_drives_dictionary_flow_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    harness = _Harness(monkeypatch, identity)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    spec = runtime_reference.load_reference_spec(spec_path, base_dir=tmp_path)

    result = runtime_reference.execute_reference_run(
        spec,
        spec.arm(HYBRID_ID),
        train_npz=identity["train"],
        test_npz=identity["test"],
        work_dir=tmp_path / "work",
    )

    assert harness.names() == [
        "load_dictionary",
        "load_dictionary",
        "run_torch_training",
        "run_torch_inference",
        "reassemble",
        "reassemble",
        "eval_reconstruction",
    ]
    training_call = harness.calls[2]
    assert training_call[1] == "dict"
    assert training_call[2] == "hybrid_resnet"
    assert training_call[3] == 3
    assert training_call[4] == 5
    assert training_call[5] == "mae"
    assert training_call[6] == "central_mask"
    assert training_call[7] == "amplitude"
    assert training_call[8] == 2.0
    assert harness.calls[3] == ("run_torch_inference", False, True)
    assert [call[1] for call in harness.calls if call[0] == "reassemble"] == [
        "grid_lines",
        "position",
    ]
    assert harness.calls[-1][4] == spec.recipe.offset
    assert (
        result.checkpoint_sha256 == hashlib.sha256(harness.CHECKPOINT_BYTES).hexdigest()
    )
    expected_patches = harness.patches[..., 0] + 1j * harness.patches[..., 1]
    assert result.pre_stitch_patch_sha256 == canonical_array_sha256(expected_patches)
    assert result.historical_canvas_sha256 == canonical_array_sha256(
        harness.historical_canvas
    )
    assert result.generic_canvas_sha256 == result.historical_canvas_sha256
    assert result.canvases_equivalent
    assert result.masks_equivalent
    assert result.no_resize_asserted
    assert result.gauge_handling == runtime_reference.DECLARED_GAUGE_HANDLING
    assert result.fixture_amp_ssim == 0.90
    assert result.fixture_phase_ssim == 0.97
    assert result.fixture_amp_mae == 0.05
    assert result.fixture_phase_mae == 0.10
    assert result.command["amplitude_physics_gain"] == 16.0
    assert result.command["architecture"] == "hybrid_resnet"
    assert result.command["generator_output_mode"] == "real_imag"


def test_reference_run_rejects_restored_gain4_before_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    harness = _Harness(
        monkeypatch, identity, restored_overrides={"amplitude_physics_gain": 4.0}
    )
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    spec = runtime_reference.load_reference_spec(spec_path, base_dir=tmp_path)

    with pytest.raises(RuntimeExecutionError, match="amplitude_physics_gain"):
        runtime_reference.execute_reference_run(
            spec,
            spec.arm(HYBRID_ID),
            train_npz=identity["train"],
            test_npz=identity["test"],
            work_dir=tmp_path / "work",
        )

    assert "run_torch_inference" not in harness.names()


def test_restored_binding_command_reads_authoritative_checkpoint_sections() -> None:
    from scripts.studies.ablation import runtime_reference_execution

    model = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="hybrid_resnet",
            generator_output_mode="real_imag",
            hybrid_encoder_conv_hidden_scale=2.0,
            training_patch_weighting="central_mask",
            physics_forward_mode="amplitude",
            amplitude_physics_gain=16.0,
            loss_function="WRONG_MODEL_SECTION",
        ),
        training_config=SimpleNamespace(
            epochs=5,
            torch_loss_mode="mae",
            subsample_seed=999,
        ),
        data_config=SimpleNamespace(subsample_seed=3),
        hparams={"generator_output": "wrong_hparams_section"},
    )

    assert runtime_reference_execution._restored_binding_command(model) == {
        "architecture": "hybrid_resnet",
        "generator_output_mode": "real_imag",
        "hybrid_encoder_conv_hidden_scale": 2.0,
        "training_patch_weighting": "central_mask",
        "physics_forward_mode": "amplitude",
        "amplitude_physics_gain": 16.0,
        "torch_loss_mode": "mae",
        "seed": 3,
        "epochs": 5,
    }


@pytest.mark.parametrize(
    ("field", "restored"),
    [
        ("architecture", "cnn"),
        ("generator_output_mode", "amp_phase"),
        ("hybrid_encoder_conv_hidden_scale", 1.0),
        ("training_patch_weighting", "probe"),
        ("physics_forward_mode", "rectangular_scaled"),
        ("torch_loss_mode", "poisson"),
        ("seed", 4),
        ("epochs", 6),
    ],
)
def test_reference_run_rejects_restored_binding_mismatch_before_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    restored: Any,
) -> None:
    identity = _write_dataset_files(tmp_path)
    harness = _Harness(
        monkeypatch, identity, restored_overrides={field: restored}
    )
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    spec = runtime_reference.load_reference_spec(spec_path, base_dir=tmp_path)

    with pytest.raises(RuntimeExecutionError, match=field):
        runtime_reference.execute_reference_run(
            spec,
            spec.arm(HYBRID_ID),
            train_npz=identity["train"],
            test_npz=identity["test"],
            work_dir=tmp_path / "work",
        )

    assert "run_torch_inference" not in harness.names()


def test_passing_floors_produce_sealed_pass_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    harness = _Harness(monkeypatch, identity)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )

    outcome = _run(tmp_path, identity, spec_path)

    assert outcome.passed
    (result,) = outcome.results
    assert result.verdict is Verdict.PASS
    assert result.category == "claim_prerequisite"
    evidence_path = tmp_path / "out" / HYBRID_ID / "reference_evidence.json"
    assert evidence_path.is_file()
    evidence = IntegrationBridgeEvidence.from_sealed_artifact_bytes(
        evidence_path.read_bytes()
    )
    assert evidence.contract.bridge_id == HYBRID_ID
    assert evidence.seed == 3
    assert evidence.epochs == 5
    assert evidence.amplitude_physics_gain == 16.0
    assert evidence.selected_checkpoint.endswith("best-checkpoint.ckpt")
    assert evidence.train_npz_sha256 == _file_sha256(identity["train"])
    assert evidence.test_npz_sha256 == _file_sha256(identity["test"])
    with np.load(identity["test"]) as archive:
        sealed_ground_truth = archive["YY_ground_truth"].squeeze()
    assert evidence.ground_truth_sha256 == canonical_array_sha256(
        sealed_ground_truth
    )
    report_path = tmp_path / "out" / "reference_qualification_report.json"
    report = json.loads(report_path.read_text())
    assert report["study_id"] == "grid_lines_reference_miniature_study"
    assert report["recipe_fingerprint_sha256"] == outcome.recipe_fingerprint_sha256
    (arm_report,) = [
        entry for entry in report["references"] if entry["id"] == HYBRID_ID
    ]
    assert arm_report["verdict"] == "pass"
    assert arm_report["floors_locked"] is True
    assert arm_report["selected_checkpoint"].endswith("best-checkpoint.ckpt")
    assert arm_report["command"]["amplitude_physics_gain"] == 16.0
    assert (
        arm_report["evidence_sha256"]
        == hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    )
    visual_path = tmp_path / "out" / HYBRID_ID / "visual_bundle/visual_review.png"
    manifest_path = tmp_path / "out" / HYBRID_ID / "visual_bundle/visual_manifest.json"
    assert visual_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["visual_sha256"] == _file_sha256(visual_path)
    assert manifest["checkpoint_sha256"] == evidence.checkpoint_sha256
    assert manifest["train_npz_sha256"] == evidence.train_npz_sha256
    assert manifest["test_npz_sha256"] == evidence.test_npz_sha256
    assert manifest["amplitude_physics_gain"] == 16.0
    assert manifest["evaluator"] == runtime_reference.DECLARED_GAUGE_HANDLING
    assert manifest["panels"] == [
        "amplitude_truth",
        "amplitude_reconstruction",
        "amplitude_absolute_error",
        "phase_truth",
        "phase_reconstruction",
        "phase_absolute_error",
    ]
    limits = manifest["panel_limits"]
    assert limits["amplitude_truth"] == limits["amplitude_reconstruction"]
    assert limits["amplitude_truth"][0] == 0.0
    assert limits["phase_truth"] == limits["phase_reconstruction"] == [
        -np.pi,
        np.pi,
    ]
    assert limits["amplitude_absolute_error"][0] == 0.0
    assert limits["phase_absolute_error"][0] == 0.0
    assert manifest["normalization_policy"] == (
        "raw_canvas_shared_truth_reconstruction_limits_v1"
    )
    assert manifest["historical_canvas_sha256"] == canonical_array_sha256(
        harness.historical_canvas
    )
    with np.load(identity["test"]) as archive:
        ground_truth = archive["YY_ground_truth"].squeeze()
    assert manifest["ground_truth_sha256"] == canonical_array_sha256(ground_truth)
    assert manifest["historical_canvas_array"]["shape"] == list(
        harness.historical_canvas.shape
    )
    assert manifest["historical_canvas_array"]["dtype"] == str(
        harness.historical_canvas.dtype
    )
    assert manifest["ground_truth_array"]["shape"] == list(ground_truth.shape)
    assert manifest["ground_truth_array"]["dtype"] == str(ground_truth.dtype)


def test_visual_bundle_refuses_preexisting_destination_without_changing_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.studies.ablation import runtime_reference_visuals

    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    spec = runtime_reference.load_reference_spec(spec_path, base_dir=tmp_path)
    result = runtime_reference.execute_reference_run(
        spec,
        spec.arm(HYBRID_ID),
        train_npz=identity["train"],
        test_npz=identity["test"],
        work_dir=tmp_path / "work",
    )
    bundle = tmp_path / "visuals" / "visual_bundle"
    bundle.mkdir(parents=True)
    png = bundle / "visual_review.png"
    manifest = bundle / "visual_manifest.json"
    png.write_bytes(b"old-png")
    manifest.write_bytes(b"old-manifest")

    with pytest.raises(RuntimeExecutionError, match="visual_bundle"):
        runtime_reference_visuals.publish_reference_visual(
            result, architecture="hybrid_resnet", output_dir=tmp_path / "visuals"
        )

    assert png.read_bytes() == b"old-png"
    assert manifest.read_bytes() == b"old-manifest"


def test_visual_bundle_second_write_failure_rolls_back_and_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.studies.ablation import runtime_reference_visuals

    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    spec = runtime_reference.load_reference_spec(spec_path, base_dir=tmp_path)
    result = runtime_reference.execute_reference_run(
        spec,
        spec.arm(HYBRID_ID),
        train_npz=identity["train"],
        test_npz=identity["test"],
        work_dir=tmp_path / "work",
    )
    original_write = runtime_reference_visuals._write_visual_bundle_file

    def fail_manifest(path: Path, data: bytes) -> None:
        if path.name == "visual_manifest.json":
            raise OSError("injected second write interruption")
        original_write(path, data)

    monkeypatch.setattr(
        runtime_reference_visuals, "_write_visual_bundle_file", fail_manifest
    )
    destination = tmp_path / "visuals"
    with pytest.raises(RuntimeExecutionError, match="interruption"):
        runtime_reference_visuals.publish_reference_visual(
            result, architecture="hybrid_resnet", output_dir=destination
        )

    assert not (destination / "visual_bundle").exists()
    assert not tuple(destination.glob(".visual_bundle.*"))

    monkeypatch.setattr(
        runtime_reference_visuals, "_write_visual_bundle_file", original_write
    )
    runtime_reference_visuals.publish_reference_visual(
        result, architecture="hybrid_resnet", output_dir=destination
    )
    assert (destination / "visual_bundle/visual_review.png").is_file()
    assert (destination / "visual_bundle/visual_manifest.json").is_file()


def test_visual_bundle_rejects_reconstruction_changed_after_hashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.studies.ablation import runtime_reference_visuals

    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    spec = runtime_reference.load_reference_spec(spec_path, base_dir=tmp_path)
    result = runtime_reference.execute_reference_run(
        spec,
        spec.arm(HYBRID_ID),
        train_npz=identity["train"],
        test_npz=identity["test"],
        work_dir=tmp_path / "work",
    )
    altered = dataclasses.replace(
        result, historical_canvas=result.historical_canvas + np.complex64(1.0)
    )

    with pytest.raises(RuntimeExecutionError, match="historical_canvas_sha256"):
        runtime_reference_visuals.publish_reference_visual(
            altered, architecture="hybrid_resnet", output_dir=tmp_path / "visuals"
        )

    assert not (tmp_path / "visuals/visual_bundle").exists()


def test_visual_bundle_rejects_ground_truth_changed_after_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.studies.ablation import runtime_reference_visuals

    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    spec = runtime_reference.load_reference_spec(spec_path, base_dir=tmp_path)
    result = runtime_reference.execute_reference_run(
        spec,
        spec.arm(HYBRID_ID),
        train_npz=identity["train"],
        test_npz=identity["test"],
        work_dir=tmp_path / "work",
    )
    altered = dataclasses.replace(
        result, ground_truth=result.ground_truth + np.complex64(1.0)
    )

    with pytest.raises(RuntimeExecutionError, match="ground_truth_sha256"):
        runtime_reference_visuals.publish_reference_visual(
            altered, architecture="hybrid_resnet", output_dir=tmp_path / "visuals"
        )

    assert not (tmp_path / "visuals/visual_bundle").exists()


@pytest.mark.parametrize("nonempty", [False, True])
def test_visual_bundle_no_replace_preserves_competitor_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, nonempty: bool
) -> None:
    from scripts.studies.ablation import runtime_reference_visuals

    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    spec = runtime_reference.load_reference_spec(spec_path, base_dir=tmp_path)
    result = runtime_reference.execute_reference_run(
        spec,
        spec.arm(HYBRID_ID),
        train_npz=identity["train"],
        test_npz=identity["test"],
        work_dir=tmp_path / "work",
    )
    destination = tmp_path / "visuals"
    competitor = destination / "visual_bundle"

    def publish_competitor(_destination: Path) -> None:
        competitor.mkdir()
        if nonempty:
            (competitor / "competitor.txt").write_bytes(b"competitor-bytes")

    monkeypatch.setattr(
        runtime_reference_visuals,
        "_before_visual_bundle_publish",
        publish_competitor,
        raising=False,
    )
    with pytest.raises(RuntimeExecutionError, match="overwrite|published bundle"):
        runtime_reference_visuals.publish_reference_visual(
            result, architecture="hybrid_resnet", output_dir=destination
        )

    assert competitor.is_dir()
    if nonempty:
        assert (competitor / "competitor.txt").read_bytes() == b"competitor-bytes"
    else:
        assert not tuple(competitor.iterdir())
    assert not tuple(destination.glob(".visual_bundle.*"))

    shutil.rmtree(competitor)
    monkeypatch.setattr(
        runtime_reference_visuals,
        "_before_visual_bundle_publish",
        lambda _destination: None,
        raising=False,
    )
    runtime_reference_visuals.publish_reference_visual(
        result, architecture="hybrid_resnet", output_dir=destination
    )
    assert (competitor / "visual_review.png").is_file()
    assert (competitor / "visual_manifest.json").is_file()


def test_floor_candidate_derives_documented_tolerances_from_sealed_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    _run(tmp_path, identity, spec_path)
    evidence_path = tmp_path / "out" / HYBRID_ID / "reference_evidence.json"
    candidate_path = tmp_path / "candidate.json"

    candidate = runtime_reference.derive_floor_candidate(
        spec_path,
        HYBRID_ID,
        evidence_path,
        candidate_path,
        base_dir=tmp_path,
    )

    assert candidate_path.is_file()
    assert candidate["amp_ssim_min"] == pytest.approx(0.865)
    assert candidate["phase_ssim_min"] == pytest.approx(0.955)
    assert candidate["amp_mae_max"] == pytest.approx(0.065)
    assert candidate["phase_mae_max"] == pytest.approx(0.125)
    provenance = candidate["provenance"]
    assert provenance["source_evidence_sha256"] == _file_sha256(evidence_path)
    assert (
        provenance["checkpoint_sha256"]
        == hashlib.sha256(_Harness.CHECKPOINT_BYTES).hexdigest()
    )
    assert provenance["train_npz_sha256"] == _file_sha256(identity["train"])
    assert provenance["test_npz_sha256"] == _file_sha256(identity["test"])
    assert provenance["amplitude_physics_gain"] == 16.0
    assert provenance["method"] == "task27_reference_tolerance_policy_v1"
    assert provenance["source_spec_sha256"] == _file_sha256(spec_path)
    assert provenance["integration_bridge_requirement_schema"] == (
        "hybrid_resnet_integration_bridge_v4"
    )
    assert len(provenance["integration_bridge_requirement_sha256"]) == 64

    before = candidate_path.read_bytes()
    with pytest.raises(RuntimeExecutionError, match="overwrite"):
        runtime_reference.derive_floor_candidate(
            spec_path,
            HYBRID_ID,
            evidence_path,
            candidate_path,
            base_dir=tmp_path,
        )
    assert candidate_path.read_bytes() == before


def test_floor_candidate_provenance_distinguishes_mutated_checked_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    _run(tmp_path, identity, spec_path)
    evidence = tmp_path / "out" / HYBRID_ID / "reference_evidence.json"
    original = runtime_reference.derive_floor_candidate(
        spec_path,
        HYBRID_ID,
        evidence,
        tmp_path / "original_candidate.json",
        base_dir=tmp_path,
    )
    mutated_spec = tmp_path / "mutated_reference_spec.toml"
    mutated_spec.write_text(
        spec_path.read_text().replace(
            'id = "grid_lines_reference_miniature_study"',
            'id = "grid_lines_reference_miniature_study_mutated"',
        )
    )

    mutated = runtime_reference.derive_floor_candidate(
        mutated_spec,
        HYBRID_ID,
        evidence,
        tmp_path / "mutated_candidate.json",
        base_dir=tmp_path,
    )

    original_provenance = original["provenance"]
    mutated_provenance = mutated["provenance"]
    assert original_provenance["source_spec_sha256"] != mutated_provenance[
        "source_spec_sha256"
    ]
    assert original_provenance["integration_bridge_requirement_sha256"] == (
        mutated_provenance["integration_bridge_requirement_sha256"]
    )


def test_floor_candidate_rejects_gain4_execution_against_gain16_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    _run(tmp_path, identity, spec_path)
    source = tmp_path / "out" / HYBRID_ID / "reference_evidence.json"
    payload = json.loads(source.read_text())
    payload["amplitude_physics_gain"] = 4.0
    tampered = tmp_path / "gain4_evidence.json"
    tampered.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    with pytest.raises(RuntimeExecutionError, match="command_mismatch"):
        runtime_reference.derive_floor_candidate(
            spec_path,
            HYBRID_ID,
            tampered,
            tmp_path / "candidate.json",
            base_dir=tmp_path,
        )


@pytest.mark.parametrize(
    ("metrics", "reason"),
    [
        ({"mae": (0.05, 0.10), "ssim": (0.70, 0.97)}, "ssim_floor_failed"),
        ({"mae": (0.15, 0.10), "ssim": (0.90, 0.97)}, "mae_guard_failed"),
    ],
)
def test_floor_candidate_rejects_sealed_evidence_that_failed_qualification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metrics: dict[str, tuple[float, float]],
    reason: str,
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(
        monkeypatch,
        identity,
        metrics=metrics,
    )
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    outcome = _run(tmp_path, identity, spec_path)
    assert not outcome.passed
    evidence = tmp_path / "out" / HYBRID_ID / "reference_evidence.json"

    with pytest.raises(RuntimeExecutionError, match=reason):
        runtime_reference.derive_floor_candidate(
            spec_path,
            HYBRID_ID,
            evidence,
            tmp_path / "candidate.json",
            base_dir=tmp_path,
        )


def test_floor_candidate_cli_requires_spec_and_reference_and_adjudicates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts.studies.ablation import runtime_reference_floors

    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    _run(tmp_path, identity, spec_path)
    monkeypatch.chdir(tmp_path)
    code = runtime_reference_floors.main(
        [
            "--spec",
            str(spec_path),
            "--reference",
            HYBRID_ID,
            "--evidence",
            str(tmp_path / "out" / HYBRID_ID / "reference_evidence.json"),
            "--output",
            str(tmp_path / "cli_candidate.json"),
        ]
    )

    assert code == 0
    assert (tmp_path / "cli_candidate.json").is_file()


def test_reference_output_root_must_be_fresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    harness = _Harness(monkeypatch, identity)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    (tmp_path / "out").mkdir()

    with pytest.raises(StudyRequestError, match="fresh output root"):
        _run(tmp_path, identity, spec_path)

    assert harness.names() == []


def test_below_floor_ssim_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity, metrics={"mae": (0.05, 0.10), "ssim": (0.70, 0.97)})
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )

    outcome = _run(tmp_path, identity, spec_path)

    assert not outcome.passed
    (result,) = outcome.results
    assert result.verdict is Verdict.FAIL
    assert result.reason == "integration_bridge_ssim_floor_failed"


def test_mae_guard_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity, metrics={"mae": (0.15, 0.10), "ssim": (0.90, 0.97)})
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )

    outcome = _run(tmp_path, identity, spec_path)

    (result,) = outcome.results
    assert result.verdict is Verdict.FAIL
    assert result.reason == "integration_bridge_mae_guard_failed"


def test_unlocked_frozen_reference_fails_closed_without_executing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    harness = _Harness(monkeypatch, identity)
    reference = _hybrid_reference(identity, frozen_reference_artifact="cnn_floors.json")
    reference["id"] = CNN_ID
    reference["runner"]["architecture"] = "cnn"
    reference["bridge"].update(
        bridge_id=CNN_ID,
        architecture="cnn",
        ssim_floor_source="frozen_reference_artifact",
    )
    spec_path = _write_spec(tmp_path, identity, references=[reference])

    outcome = _run(tmp_path, identity, spec_path)

    assert not outcome.passed
    (result,) = outcome.results
    assert result.verdict is Verdict.FAIL
    assert result.reason == "integration_bridge_ssim_floors_unlocked"
    assert harness.names() == []
    report = json.loads(
        (tmp_path / "out" / "reference_qualification_report.json").read_text()
    )
    (arm_report,) = report["references"]
    assert arm_report["floors_locked"] is False
    assert arm_report["verdict"] == "fail"


def test_frozen_artifact_locks_floors_and_gates_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(
        monkeypatch,
        identity,
        metrics={"mae": (0.05, 0.10), "ssim": (0.87, 0.93)},
        restored_overrides={"architecture": "cnn"},
    )
    frozen = json.dumps(
        {"amp_ssim_min": 0.886, "phase_ssim_min": 0.928},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    artifact = tmp_path / "cnn_floors.json"
    artifact.write_bytes(frozen)
    reference = _hybrid_reference(identity, frozen_reference_artifact="cnn_floors.json")
    reference["id"] = CNN_ID
    reference["runner"]["architecture"] = "cnn"
    reference["bridge"].update(
        bridge_id=CNN_ID,
        architecture="cnn",
        ssim_floor_source="frozen_reference_artifact",
        ssim_floor_reference_artifact_sha256=hashlib.sha256(frozen).hexdigest(),
        # Approximate declarations only; the frozen artifact locks the finals.
        fixture_amp_ssim_min=0.80,
        fixture_phase_ssim_min=0.85,
    )
    spec_path = _write_spec(tmp_path, identity, references=[reference])
    spec = runtime_reference.load_reference_spec(spec_path, base_dir=tmp_path)
    assert spec.arm(CNN_ID).floors_locked

    # 0.87/0.93 clears the declared approximations but not the locked floors.
    outcome = _run(tmp_path, identity, spec_path)
    (result,) = outcome.results
    assert result.verdict is Verdict.FAIL
    assert result.reason == "integration_bridge_ssim_floor_failed"


def test_frozen_artifact_pass_when_locked_floors_met(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(
        monkeypatch,
        identity,
        metrics={"mae": (0.05, 0.10), "ssim": (0.89, 0.93)},
        restored_overrides={"architecture": "cnn"},
    )
    frozen = json.dumps(
        {"amp_ssim_min": 0.886, "phase_ssim_min": 0.928},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    (tmp_path / "cnn_floors.json").write_bytes(frozen)
    reference = _hybrid_reference(identity, frozen_reference_artifact="cnn_floors.json")
    reference["id"] = CNN_ID
    reference["runner"]["architecture"] = "cnn"
    reference["bridge"].update(
        bridge_id=CNN_ID,
        architecture="cnn",
        ssim_floor_source="frozen_reference_artifact",
        ssim_floor_reference_artifact_sha256=hashlib.sha256(frozen).hexdigest(),
        fixture_amp_ssim_min=0.80,
        fixture_phase_ssim_min=0.85,
    )
    spec_path = _write_spec(tmp_path, identity, references=[reference])

    outcome = _run(tmp_path, identity, spec_path)

    (result,) = outcome.results
    assert result.verdict is Verdict.PASS


def test_unclassified_canvas_difference_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity, generic_canvas_delta=0.5)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )

    outcome = _run(tmp_path, identity, spec_path)

    (result,) = outcome.results
    assert result.verdict is Verdict.FAIL
    assert result.reason == "integration_bridge_unclassified_difference"


def test_comparison_invalidating_difference_fails_despite_passing_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity, generic_canvas_delta=0.5)
    reference = _hybrid_reference(
        identity,
        expected_differences={
            "canvas_equivalence": {
                "classification": "comparison_invalidating",
                "justification": "generic stitch disagrees beyond tolerance",
            }
        },
    )
    spec_path = _write_spec(tmp_path, identity, references=[reference])

    outcome = _run(tmp_path, identity, spec_path)

    (result,) = outcome.results
    assert result.verdict is Verdict.FAIL
    assert result.reason == "integration_bridge_comparison_invalidating_difference"


def test_harmless_classified_difference_still_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity, generic_canvas_delta=0.5)
    reference = _hybrid_reference(
        identity,
        expected_differences={
            "canvas_equivalence": {
                "classification": "harmless",
                "justification": (
                    "historical grid stitch and generic position stitch share the "
                    "declared crop boundary; hash equality is diagnostic"
                ),
            }
        },
    )
    spec_path = _write_spec(tmp_path, identity, references=[reference])

    outcome = _run(tmp_path, identity, spec_path)

    (result,) = outcome.results
    assert result.verdict is Verdict.PASS
    evidence = IntegrationBridgeEvidence.from_sealed_artifact_bytes(
        (tmp_path / "out" / HYBRID_ID / "reference_evidence.json").read_bytes()
    )
    (difference,) = evidence.recorded_differences
    assert difference.field == "canvas_equivalence"
    assert difference.classification == "harmless"


def test_hidden_resize_is_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity, historical_canvas_shape=(20, 20))
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    spec = runtime_reference.load_reference_spec(spec_path, base_dir=tmp_path)

    with pytest.raises(RuntimeExecutionError, match="no_resize"):
        runtime_reference.execute_reference_run(
            spec,
            spec.arm(HYBRID_ID),
            train_npz=identity["train"],
            test_npz=identity["test"],
            work_dir=tmp_path / "work",
        )


def test_undeclared_gauge_handling_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    spec = runtime_reference.load_reference_spec(spec_path, base_dir=tmp_path)
    arm = spec.arm(HYBRID_ID)
    result = runtime_reference.execute_reference_run(
        spec,
        arm,
        train_npz=identity["train"],
        test_npz=identity["test"],
        work_dir=tmp_path / "work",
    )
    tampered = dataclasses.replace(result, gauge_handling="undeclared_fit")

    with pytest.raises(RuntimeExecutionError, match="gauge"):
        runtime_reference.assemble_reference_evidence(spec, arm, tampered)


def test_missing_checkpoint_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    harness = _Harness(monkeypatch, identity)
    from scripts.studies import grid_lines_torch_runner as runner_mod

    def train_without_checkpoint(cfg: Any, *args: Any, **kwargs: Any) -> dict:
        return {"model": harness.model, "history": {}}

    monkeypatch.setattr(runner_mod, "run_torch_training", train_without_checkpoint)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    spec = runtime_reference.load_reference_spec(spec_path, base_dir=tmp_path)

    with pytest.raises(RuntimeExecutionError, match="checkpoint"):
        runtime_reference.execute_reference_run(
            spec,
            spec.arm(HYBRID_ID),
            train_npz=identity["train"],
            test_npz=identity["test"],
            work_dir=tmp_path / "work",
        )


def test_existing_evidence_refuses_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity)
    spec_path = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    evidence_path = tmp_path / "out" / HYBRID_ID / "reference_evidence.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_bytes(b"{}")

    with pytest.raises(StudyRequestError, match="evidence"):
        _run(tmp_path, identity, spec_path)


def test_reference_filter_selects_one_arm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _Harness(monkeypatch, identity)
    hybrid = _hybrid_reference(identity)
    cnn = _hybrid_reference(identity, frozen_reference_artifact="cnn_floors.json")
    cnn["id"] = CNN_ID
    cnn["runner"]["architecture"] = "cnn"
    cnn["bridge"].update(
        bridge_id=CNN_ID,
        architecture="cnn",
        ssim_floor_source="frozen_reference_artifact",
    )
    spec_path = _write_spec(tmp_path, identity, references=[hybrid, cnn])

    outcome = _run(tmp_path, identity, spec_path, reference=HYBRID_ID)

    assert [result.id for result in outcome.results] == [HYBRID_ID]
    with pytest.raises(StudyRequestError, match="unknown reference"):
        _run(tmp_path, identity, spec_path, reference="nope")


# ---------------------------------------------------------------------------
# Facade re-exports
# ---------------------------------------------------------------------------


def test_runtime_facade_reexports_reference_api() -> None:
    assert runtime.run_reference_qualification is (
        runtime_reference.run_reference_qualification
    )
    assert runtime.load_reference_spec is runtime_reference.load_reference_spec
    assert runtime.execute_reference_run is runtime_reference.execute_reference_run
    assert "run_reference_qualification" in runtime.__all__
    assert "ReferenceQualificationRequest" in runtime.__all__


def test_datasets_facade_reexports_reference_recipe_api() -> None:
    from scripts.studies.ablation import dataset_reference, datasets

    assert datasets.parse_grid_lines_reference_recipe is (
        dataset_reference.parse_grid_lines_reference_recipe
    )
    assert datasets.validate_reference_npz_pair is (
        dataset_reference.validate_reference_npz_pair
    )
    assert "GridLinesReferenceRecipe" in datasets.__all__
    assert "validate_reference_npz_pair" in datasets.__all__
