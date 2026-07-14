"""Dictionary-loader execution and sealed evidence for reference qualification.

Drives the historical grid-lines dictionary path — the executable reference
condition — through the canonical train/checkpoint/reconstruct/evaluate flow:

1. ``grid_lines_torch_runner.load_cached_dataset_with_metadata`` (dictionary
   loader; the transformed probe passes through unchanged),
2. ``grid_lines_torch_runner.run_torch_training`` (dict containers ->
   ``_train_with_lightning``),
3. ``lightning_utils.find_best_checkpoint`` + SHA-256 (checkpoint identity),
4. ``grid_lines_torch_runner.run_torch_inference`` (pre-stitch patches),
5. ``grid_lines_torch_runner._reassemble_predictions_for_metrics`` twice:
   once in historical ``grid_lines`` mode and once in generic ``position``
   mode (dual-evaluator canvas/mask diagnostics),
6. ``ptycho.evaluation.eval_reconstruction`` with the declared gauge handling
   (fixture metric method) — never through the runner's resize branch.

Hidden resizing, undeclared gauge fitting, and truth-dependent alignment are
prohibited: shape mismatches fail closed before any resize could occur, and
evidence assembly refuses any gauge label other than the declared constant.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .dataset_reference import (
    MaterializedReferenceDataset,
    validate_reference_npz_pair,
)
from .dataset_provenance import canonical_array_sha256
from .runtime_checkpoint import restore_checkpoint_from_hparams
from .runtime_atomic import atomic_write_bytes_no_clobber
from .runtime_errors import RuntimeExecutionError, sha256_file
from .runtime_errors import stage as _stage
from .runtime_reference_spec import (
    BRIDGE_COMMAND_FIELDS,
    ReferenceArm,
    ReferenceQualificationSpec,
)
from .verdicts import (
    BRIDGE_DIFFERENCE_IDS,
    INTEGRATION_BRIDGE_EVIDENCE_SCHEMA_VERSION,
    IntegrationBridgeEvidence,
)

__all__ = [
    "DECLARED_GAUGE_HANDLING",
    "ReferenceRunResult",
    "assemble_reference_evidence",
    "execute_reference_run",
    "seal_reference_evidence",
]

#: The single declared gauge of the fixture metric method: ``ptycho.evaluation.
#: eval_reconstruction`` mean-normalizes predicted amplitude and plane-fits
#: phase before scoring. Any other gauge label is undeclared and fails closed.
DECLARED_GAUGE_HANDLING = "eval_reconstruction_mean_amplitude_plane_phase_v1"

#: NPZ metadata additional_parameters keys measurable at execution time,
#: mapped onto the bridge-contract fields they instantiate.
_METADATA_CONTRACT_FIELDS = {
    "size": "grid_lines_size",
    "offset": "grid_lines_offset",
    "outer_offset_train": "outer_offset_train",
    "outer_offset_test": "outer_offset_test",
    "nimgs_train": "nimgs_train",
    "nimgs_test": "nimgs_test",
    "probe_smoothing_sigma": "probe_smoothing_sigma",
}


@dataclass(frozen=True)
class ReferenceRunResult:
    """Typed evidence produced by one dictionary-loader reference run."""

    arm_id: str
    materialized: MaterializedReferenceDataset
    best_checkpoint: Path
    checkpoint_sha256: str
    pre_stitch_patch_sha256: str
    historical_canvas_sha256: str
    ground_truth_sha256: str
    generic_canvas_sha256: str
    historical_mask_sha256: str
    generic_mask_sha256: str
    canvases_equivalent: bool
    masks_equivalent: bool
    no_resize_asserted: bool
    gauge_handling: str
    fixture_amp_mae: float
    fixture_phase_mae: float
    fixture_amp_ssim: float
    fixture_phase_ssim: float
    historical_canvas: np.ndarray
    ground_truth: np.ndarray
    #: Verbatim command operands of the executed runner configuration.
    command: Mapping[str, Any]
    #: Contract fields measured during execution (override declared values).
    measured_contract: Mapping[str, Any]


def _complex_patches(runner_mod: Any, predictions: np.ndarray) -> np.ndarray:
    array = np.asarray(predictions)
    if array.ndim >= 2 and array.shape[-1] == 2 and not np.iscomplexobj(array):
        return np.asarray(runner_mod.to_complex_patches(array))
    return array


def _finite_mask(canvas: np.ndarray) -> np.ndarray:
    return np.isfinite(canvas.real) & np.isfinite(canvas.imag)


def _ground_truth(test_data: Mapping[str, Any]) -> np.ndarray:
    for key in ("YY_ground_truth", "YY_full", "objectGuess"):
        value = test_data.get(key)
        if value is not None:
            return np.asarray(value)
    raise RuntimeExecutionError(
        "ground_truth",
        "reference test data must provide YY_ground_truth, YY_full, or objectGuess",
    )


def _metric_pair(metrics: Mapping[str, Any], key: str) -> tuple[float, float]:
    pair = metrics.get(key)
    try:
        amp, phase = float(pair[0]), float(pair[1])
    except (TypeError, ValueError, IndexError) as error:
        raise RuntimeExecutionError(
            "metrics", f"eval metrics must carry a numeric {key!r} pair: {error}"
        ) from error
    if not np.isfinite(amp) or not np.isfinite(phase):
        raise RuntimeExecutionError("metrics", f"{key} metrics must be finite")
    return amp, phase


def _restored_binding_command(model: Any) -> dict[str, Any]:
    """Extract binding operands from configs reconstructed out of checkpoint hparams."""
    model_config = getattr(model, "model_config", None)
    training_config = getattr(model, "training_config", None)
    data_config = getattr(model, "data_config", None)
    if model_config is None or training_config is None or data_config is None:
        raise RuntimeExecutionError(
            "checkpoint_bindings",
            "restored checkpoint is missing model_config, training_config, or "
            "data_config",
        )

    def required(holder: Any, name: str) -> Any:
        if isinstance(holder, Mapping):
            value = holder.get(name)
        else:
            value = getattr(holder, name, None)
        if value is None:
            raise RuntimeExecutionError(
                "checkpoint_bindings",
                f"restored checkpoint does not persist binding operand {name!r}",
            )
        return value

    return {
        "architecture": str(required(model_config, "architecture")),
        "generator_output_mode": str(
            required(model_config, "generator_output_mode")
        ),
        "hybrid_encoder_conv_hidden_scale": float(
            required(model_config, "hybrid_encoder_conv_hidden_scale")
        ),
        "training_patch_weighting": str(
            required(model_config, "training_patch_weighting")
        ),
        "physics_forward_mode": str(required(model_config, "physics_forward_mode")),
        "amplitude_physics_gain": float(
            required(model_config, "amplitude_physics_gain")
        ),
        "torch_loss_mode": str(
            required(training_config, "torch_loss_mode")
        ).lower(),
        "seed": int(required(data_config, "subsample_seed")),
        "epochs": int(required(training_config, "epochs")),
    }


def _validate_restored_binding_command(
    arm: ReferenceArm, command: Mapping[str, Any]
) -> None:
    mismatches = [
        field
        for field in BRIDGE_COMMAND_FIELDS
        if command[field] != arm.bridge[field]
    ]
    if mismatches:
        details = ", ".join(
            f"{field}={command[field]!r} (declared {arm.bridge[field]!r})"
            for field in mismatches
        )
        raise RuntimeExecutionError(
            "checkpoint_bindings",
            f"restored checkpoint binding mismatch before inference: {details}",
        )


def _measured_contract(
    spec: ReferenceQualificationSpec,
    cfg: Any,
    materialized: MaterializedReferenceDataset,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Instantiate contract fields the execution can measure directly."""
    n = spec.recipe.N
    measured: dict[str, Any] = {
        "loader_kind": "dictionary",
        "transformed_probe_sha256": materialized.probe_sha256,
        "dictionary_effective_probe_sha256": materialized.probe_sha256,
        "probe_target_n": n,
        "grid_lines_n": n,
        "grid_lines_gridsize": int(cfg.gridsize),
        "generic_position_crop_border": int(cfg.position_crop_border),
        "generic_effective_patch_n": n - 2 * int(cfg.position_crop_border),
        "probe_mask": bool(cfg.probe_mask),
        "probe_mask_sigma": float(cfg.probe_mask_sigma),
        "probe_mask_diameter": (
            None if cfg.probe_mask_diameter is None else float(cfg.probe_mask_diameter)
        ),
    }
    if not cfg.probe_mask:
        measured["probe_mask_tensor_is_none"] = True
        measured["resolved_probe_mask_kind"] = "identity"
        measured["resolved_probe_mask_sha256"] = canonical_array_sha256(
            np.ones((n, n), dtype=np.float32)
        )
    parameters = (metadata or {}).get("additional_parameters")
    if isinstance(parameters, Mapping):
        for key, field in _METADATA_CONTRACT_FIELDS.items():
            if key in parameters and parameters[key] is not None:
                measured[field] = parameters[key]
    return measured


def execute_reference_run(
    spec: ReferenceQualificationSpec,
    arm: ReferenceArm,
    *,
    train_npz: Path,
    test_npz: Path,
    work_dir: Path,
) -> ReferenceRunResult:
    """Run one reference arm through the canonical dictionary-loader flow."""
    from scripts.studies import grid_lines_torch_runner as runner_mod

    with _stage("dataset_identity"):
        materialized = validate_reference_npz_pair(spec.recipe, train_npz, test_npz)

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    with _stage("dictionary_load"):
        train_data, train_metadata = runner_mod.load_cached_dataset_with_metadata(
            Path(train_npz)
        )
        test_data, test_metadata = runner_mod.load_cached_dataset_with_metadata(
            Path(test_npz)
        )

    cfg = runner_mod.TorchRunnerConfig(
        train_npz=Path(train_npz),
        test_npz=Path(test_npz),
        output_dir=work,
        **dict(arm.runner),
    )

    with _stage("training"):
        results = runner_mod.run_torch_training(
            cfg,
            train_data,
            test_data,
            train_metadata=train_metadata,
            test_metadata=test_metadata,
        )
    trained_model = results.get("model")
    if trained_model is None and isinstance(results.get("models"), dict):
        trained_model = results["models"].get("diffraction_to_obj")
    if trained_model is None:
        raise RuntimeExecutionError("training", "training produced no model handle")

    with _stage("checkpoint_identity"):
        from ptycho_torch import lightning_utils

        best_checkpoint = lightning_utils.find_best_checkpoint(work)
        if best_checkpoint is None:
            raise RuntimeExecutionError(
                "checkpoint_identity", f"no checkpoint found under {work}"
            )
        best_checkpoint = Path(best_checkpoint).resolve()
        checkpoint_sha256 = sha256_file(best_checkpoint)

    with _stage("checkpoint_restore"):
        selected_model = restore_checkpoint_from_hparams(
            best_checkpoint,
            expected_sha256=checkpoint_sha256,
            device="cpu",
        )
        restored_command = _restored_binding_command(selected_model)
        _validate_restored_binding_command(arm, restored_command)

    with _stage("inference"):
        predictions = runner_mod.run_torch_inference(
            selected_model, test_data, cfg, metadata=test_metadata
        )
        patches = _complex_patches(runner_mod, predictions)
        pre_stitch_patch_sha256 = canonical_array_sha256(patches)

    ground_truth = np.squeeze(_ground_truth(test_data))
    if ground_truth.ndim != 2:
        raise RuntimeExecutionError(
            "ground_truth",
            f"reference ground truth must squeeze to 2D; got {ground_truth.shape}",
        )

    with _stage("historical_stitch"):
        historical_canvas, _, _ = runner_mod._reassemble_predictions_for_metrics(
            patches,
            ground_truth,
            test_data,
            test_metadata,
            replace(cfg, reassembly_mode="grid_lines"),
        )
        historical_canvas = np.squeeze(np.asarray(historical_canvas))
    with _stage("generic_stitch"):
        generic_canvas, _, _ = runner_mod._reassemble_predictions_for_metrics(
            patches,
            ground_truth,
            test_data,
            test_metadata,
            replace(cfg, reassembly_mode="position"),
        )
        generic_canvas = np.squeeze(np.asarray(generic_canvas))

    # No-hidden-resize assertion: the fixture metric method must see the
    # stitched canvas at exactly the ground-truth frame. A mismatch here is a
    # hard failure, never an interpolation.
    if historical_canvas.shape != ground_truth.shape:
        raise RuntimeExecutionError(
            "no_resize",
            "historical canvas shape "
            f"{historical_canvas.shape} != ground truth {ground_truth.shape}; "
            "resizing is prohibited and would invalidate the SSIM comparison",
        )

    historical_canvas_sha256 = canonical_array_sha256(historical_canvas)
    generic_canvas_sha256 = canonical_array_sha256(generic_canvas)
    historical_mask_sha256 = canonical_array_sha256(_finite_mask(historical_canvas))
    generic_mask_sha256 = canonical_array_sha256(_finite_mask(generic_canvas))

    with _stage("metrics"):
        from ptycho import evaluation

        ground_truth_sha256 = canonical_array_sha256(ground_truth)
        metrics = evaluation.eval_reconstruction(
            historical_canvas[..., None],
            ground_truth[..., None],
            label=arm.id,
        )
        amp_mae, phase_mae = _metric_pair(metrics, "mae")
        amp_ssim, phase_ssim = _metric_pair(metrics, "ssim")

    return ReferenceRunResult(
        arm_id=arm.id,
        materialized=materialized,
        best_checkpoint=best_checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        pre_stitch_patch_sha256=pre_stitch_patch_sha256,
        historical_canvas_sha256=historical_canvas_sha256,
        ground_truth_sha256=ground_truth_sha256,
        generic_canvas_sha256=generic_canvas_sha256,
        historical_mask_sha256=historical_mask_sha256,
        generic_mask_sha256=generic_mask_sha256,
        canvases_equivalent=historical_canvas_sha256 == generic_canvas_sha256,
        masks_equivalent=historical_mask_sha256 == generic_mask_sha256,
        no_resize_asserted=True,
        gauge_handling=DECLARED_GAUGE_HANDLING,
        fixture_amp_mae=amp_mae,
        fixture_phase_mae=phase_mae,
        fixture_amp_ssim=amp_ssim,
        fixture_phase_ssim=phase_ssim,
        historical_canvas=historical_canvas,
        ground_truth=ground_truth,
        command=restored_command,
        measured_contract=_measured_contract(spec, cfg, materialized, test_metadata),
    )


def _normalized(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_normalized(item) for item in value)
    return value


def _observed_differences(
    declared: Mapping[str, Any],
    executed: Mapping[str, Any],
    result: ReferenceRunResult,
) -> set[str]:
    # BRIDGE_DIFFERENCE_IDS already excludes identity, command-operand, and
    # gate-floor fields: only classifiable contract fields can diff here.
    observed = {
        field
        for field in executed
        if field in BRIDGE_DIFFERENCE_IDS
        and _normalized(executed[field]) != _normalized(declared[field])
    }
    if not result.canvases_equivalent:
        observed.add("canvas_equivalence")
    if not result.masks_equivalent:
        observed.add("mask_equivalence")
    return observed


def assemble_reference_evidence(
    spec: ReferenceQualificationSpec,
    arm: ReferenceArm,
    result: ReferenceRunResult,
) -> dict[str, Any]:
    """Build the v3 execution-evidence payload from one reference run.

    The executed contract starts from the declared bridge mapping and is
    overridden by every field the run measured; observed differences receive
    the spec's predeclared classification. An observed difference without a
    predeclared classification is deliberately left unrecorded so that
    ``evaluate_integration_bridge`` fails closed on it.
    """
    if result.arm_id != arm.id:
        raise RuntimeExecutionError(
            "evidence", f"result arm {result.arm_id!r} does not match {arm.id!r}"
        )
    if result.gauge_handling != DECLARED_GAUGE_HANDLING:
        raise RuntimeExecutionError(
            "gauge_handling",
            f"gauge handling {result.gauge_handling!r} is undeclared; only "
            f"{DECLARED_GAUGE_HANDLING!r} is permitted",
        )
    if not result.no_resize_asserted:
        raise RuntimeExecutionError(
            "no_resize", "reference evidence requires the no-resize assertion"
        )
    executed_contract = dict(arm.bridge)
    executed_contract.update(dict(result.measured_contract))
    executed_contract.update(dict(result.command))
    command_mismatches = sorted(
        field
        for field in BRIDGE_COMMAND_FIELDS
        if _normalized(result.command[field]) != _normalized(arm.bridge[field])
    )
    if command_mismatches:
        raise RuntimeExecutionError(
            "command_operands",
            "executed command operands diverge from the declared reference "
            f"condition: {command_mismatches}",
        )
    observed = _observed_differences(arm.bridge, executed_contract, result)
    recorded = [
        {
            "field": field,
            "classification": arm.expected_differences[field].classification,
            "justification": arm.expected_differences[field].justification,
        }
        for field in sorted(observed)
        if field in arm.expected_differences
    ]
    return {
        "schema_version": INTEGRATION_BRIDGE_EVIDENCE_SCHEMA_VERSION,
        "contract": executed_contract,
        "checkpoint_sha256": result.checkpoint_sha256,
        "selected_checkpoint": str(result.best_checkpoint),
        "train_npz_sha256": result.materialized.train_sha256,
        "test_npz_sha256": result.materialized.test_sha256,
        "pre_stitch_patch_sha256": result.pre_stitch_patch_sha256,
        "historical_canvas_sha256": result.historical_canvas_sha256,
        "ground_truth_sha256": result.ground_truth_sha256,
        "generic_canvas_sha256": result.generic_canvas_sha256,
        "historical_mask_sha256": result.historical_mask_sha256,
        "generic_mask_sha256": result.generic_mask_sha256,
        "canvases_equivalent": result.canvases_equivalent,
        "masks_equivalent": result.masks_equivalent,
        "no_resize_asserted": result.no_resize_asserted,
        "gauge_handling": result.gauge_handling,
        "recorded_differences": recorded,
        "fixture_amp_mae": result.fixture_amp_mae,
        "fixture_phase_mae": result.fixture_phase_mae,
        "fixture_amp_ssim": result.fixture_amp_ssim,
        "fixture_phase_ssim": result.fixture_phase_ssim,
        **{field: result.command[field] for field in BRIDGE_COMMAND_FIELDS},
    }


def seal_reference_evidence(
    payload: Mapping[str, Any], path: Path
) -> IntegrationBridgeEvidence:
    """Write canonical JSON bytes and reparse them as sealed evidence.

    The seal hash is recomputed from the written bytes by the consumer-side
    parser, never declared by this producer.
    """
    evidence_path = Path(path)
    if evidence_path.exists():
        raise RuntimeExecutionError(
            "evidence_seal",
            f"refusing to overwrite existing sealed evidence at {evidence_path}",
        )
    data = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    atomic_write_bytes_no_clobber(evidence_path, data, stage="evidence_seal")
    written = evidence_path.read_bytes()
    if hashlib.sha256(written).hexdigest() != hashlib.sha256(data).hexdigest():
        raise RuntimeExecutionError(
            "evidence_seal", f"sealed evidence readback mismatch at {evidence_path}"
        )
    return IntegrationBridgeEvidence.from_sealed_artifact_bytes(written)
