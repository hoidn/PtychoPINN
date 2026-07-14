"""Per-rung execution and sealed evidence for the bridge ladder (Task 21).

Each executing rung drives the same canonical grid-lines flow the reference
qualification used (train, checkpoint identity, patch inference, dual-stitch
diagnostics, fixture metric method), parameterized by the rung's resolved
ladder configuration:

- ``loader`` dispatches ingestion: ``dictionary`` uses
  ``grid_lines_torch_runner.load_cached_dataset_with_metadata`` +
  ``run_torch_training``; ``mmap`` trains through the REAL generic loader
  (``ptycho_torch.dataloader.PtychoDataset`` into the canonical
  ``_train_with_lightning`` entry — :mod:`runtime_ladder_mmap`), and its
  held-out ingestion is either the mechanical gs=1 schema adapter or the
  loader-grouped adapter for C > 1.
- ``gated_evaluator`` selects which stitched canvas the retained-SSIM gate
  scores (both canvases/masks are always hashed as diagnostics), and
  ``varpro_scaling`` routes the gated canvas through the wired VarPro seam.

The evidence seams are wired (Task 21b): VarPro, count consistency,
normalization reuse, and scan accounting live in
:mod:`runtime_ladder_seams`. Missing evidence still fails closed. Hidden
resizing and undeclared gauge handling remain hard failures exactly as in
the reference execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .dataset_reference import (
    LadderDatasetRecipe,
    MaterializedReferenceDataset,
    validate_ladder_npz_pair,
)
from .dataset_provenance import canonical_array_sha256
from .runtime_errors import RuntimeExecutionError, sha256_file
from .runtime_errors import stage as _stage
from .runtime_ladder_config import (
    LADDER_DIFFERENCE_IDS,
    RUNNER_PASSTHROUGH_FIELDS,
)
from .runtime_ladder_mmap import (
    load_grouped_generic_test_dict,
    recipe_stitch_metadata,
    train_via_generic_loader,
)
from .runtime_ladder_seams import (
    apply_varpro_to_canvas,
    build_inference_normalization_record,
    collect_scan_accounting,
    compute_count_consistency,
    observe_inference_input_scales,
    resolve_count_scale_operands,
    resolve_normalization_reuse,
    trivial_grouping_record,
)
from .runtime_reference_execution import DECLARED_GAUGE_HANDLING

__all__ = [
    "LADDER_DIFFERENCE_IDS",
    "LadderRunResult",
    "apply_varpro_to_canvas",
    "build_runner_config",
    "collect_scan_accounting",
    "compute_count_consistency",
    "build_inference_normalization_record",
    "execute_ladder_rung",
    "load_generic_test_dict",
    "observe_inference_input_scales",
    "resolve_count_scale_operands",
    "resolve_normalization_reuse",
    "train_via_generic_loader",
]

@dataclass(frozen=True)
class LadderRunResult:
    """Typed evidence produced by one executed ladder rung."""

    rung_id: str
    materialized: MaterializedReferenceDataset
    best_checkpoint: Path
    checkpoint_sha256: str
    pre_stitch_patch_sha256: str
    historical_canvas_sha256: str
    generic_canvas_sha256: str
    historical_mask_sha256: str
    generic_mask_sha256: str
    canvases_equivalent: bool
    masks_equivalent: bool
    no_resize_asserted: bool
    gauge_handling: str
    gated_evaluator: str
    amp_mae: float
    phase_mae: float
    amp_ssim: float
    phase_ssim: float
    effective_probe_sha256: str
    effective_probe_matches_recipe: bool
    inference_reuses_training_normalization: bool | None
    training_normalization_sha256: str | None
    inference_normalization_sha256: str | None
    varpro_applied: bool
    varpro_s1: float | None
    varpro_s2: float | None
    scan_accounting: Mapping[str, Any] | None
    canvas_coverage_fraction: float
    count_consistency: Mapping[str, Any] | None
    physics_scaling_constant: float | None
    resolved_config: Mapping[str, Any]


def build_runner_config(
    config: Mapping[str, Any], *, train_npz: Path, test_npz: Path, output_dir: Path
) -> Any:
    """Map the resolved ladder config onto TorchRunnerConfig operands."""
    from scripts.studies import grid_lines_torch_runner as runner_mod

    operands = {field: config[field] for field in RUNNER_PASSTHROUGH_FIELDS}
    return runner_mod.TorchRunnerConfig(
        train_npz=Path(train_npz),
        test_npz=Path(test_npz),
        output_dir=Path(output_dir),
        **operands,
    )

def load_generic_test_dict(
    test_npz: Path, recipe: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mechanically re-express a generic-schema NPZ as the runner test dict.

    Pure array reshaping — measurement values pass through bit-unchanged, so
    this adapter can never hide a normalization difference. The grid-lines
    stitch metadata is synthesized from the pinned recipe geometry.
    """
    with np.load(Path(test_npz), allow_pickle=False) as archive:
        diff3d = np.asarray(archive["diff3d"], dtype=np.float32)
        xcoords = np.asarray(archive["xcoords"], dtype=np.float32)
        ycoords = np.asarray(archive["ycoords"], dtype=np.float32)
        probe = np.asarray(archive["probeGuess"])
        truth = (
            np.asarray(archive["objectGuess"]) if "objectGuess" in archive.files
            else None
        )
    if diff3d.ndim == 3:
        diff3d = diff3d[..., np.newaxis]
    count = diff3d.shape[0]
    # Dictionary-path contract at gridsize=1: coords_nominal carries the
    # (zero) RELATIVE offsets; the global positions live in coords_offsets
    # ((y, x) at axis 2), which the position stitch requires.
    offsets = np.zeros((count, 1, 2, 1), dtype=np.float32)
    offsets[:, 0, 0, 0] = ycoords
    offsets[:, 0, 1, 0] = xcoords
    data: dict[str, Any] = {
        "diffraction": diff3d,
        "coords_nominal": np.zeros((count, 1, 2, 1), dtype=np.float32),
        "coords_offsets": offsets,
        "probeGuess": probe,
    }
    if truth is not None:
        data["YY_ground_truth"] = truth
    return data, recipe_stitch_metadata(recipe)


def _train_dictionary(
    runner_cfg: Any, config: Mapping[str, Any], train_npz: Path, test_dict: dict,
    test_metadata: Mapping[str, Any], work: Path
) -> tuple[Any, Path, Mapping[str, Any]]:
    from scripts.studies import grid_lines_torch_runner as runner_mod

    train_data, train_metadata = runner_mod.load_cached_dataset_with_metadata(
        Path(train_npz)
    )
    results = runner_mod.run_torch_training(
        runner_cfg,
        train_data,
        test_dict,
        train_metadata=train_metadata,
        test_metadata=dict(test_metadata),
    )
    model = results.get("model")
    if model is None and isinstance(results.get("models"), dict):
        model = results["models"].get("diffraction_to_obj")
    if model is None:
        raise RuntimeExecutionError("training", "training produced no model handle")
    return model, work, results


def _resolve_effective_probe(
    config: Mapping[str, Any],
    training_payload: Mapping[str, Any],
    materialized: MaterializedReferenceDataset,
) -> str:
    """Effective probe identity entering the model, fail-closed.

    A pass-through fallback is permitted ONLY for the dictionary loader,
    whose passthrough behavior is established Task 19 bridge evidence
    (``legacy_passthrough_config_inactive``). The mmap loader applies its own
    normalization policy, so mmap rungs must record the effective probe hash
    regardless of ``probe_normalize`` — anything else would seal fabricated
    pass-through evidence and hide a second variable inside the loader rung.
    """
    recorded = training_payload.get("effective_probe_sha256")
    if recorded is not None:
        return str(recorded)
    if config["loader"] == "dictionary":
        return materialized.probe_sha256
    raise RuntimeExecutionError(
        "effective_probe",
        "the generic mmap loader's effective probe hash was not recorded by "
        "the training path (extract_loader_evidence did not run) — "
        "pass-through may not be assumed for loader='mmap' regardless of "
        "probe_normalize",
    )


def _resolve_physics_scaling_constant(
    config: Mapping[str, Any], training_payload: Mapping[str, Any]
) -> float | None:
    """Resolved count scaling constant; mandatory when count_scale_mode=auto."""
    if config["count_scale_mode"] != "auto":
        return None
    value = training_payload.get("physics_scaling_constant")
    if value is None:
        raise RuntimeExecutionError(
            "count_scaling",
            "count_scale_mode=auto requires the training path to record the "
            "resolved physics scaling constant (train_via_generic_loader "
            "derives it); refusing to seal without it",
        )
    return float(value)


def _ground_truth(test_data: Mapping[str, Any]) -> np.ndarray:
    for key in ("YY_ground_truth", "YY_full", "objectGuess"):
        value = test_data.get(key)
        if value is None:
            continue
        truth = np.squeeze(np.asarray(value))
        if truth.ndim != 2:
            raise RuntimeExecutionError(
                "ground_truth",
                f"ladder ground truth must squeeze to 2D; got {truth.shape}",
            )
        return truth
    raise RuntimeExecutionError(
        "ground_truth", "ladder test data must provide ground truth"
    )


def _finite_mask(canvas: np.ndarray) -> np.ndarray:
    return np.isfinite(canvas.real) & np.isfinite(canvas.imag)


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


def execute_ladder_rung(
    spec: Any,
    rung: Any,
    *,
    train_npz: Path,
    test_npz: Path,
    work_dir: Path,
    seed: int | None = None,
) -> LadderRunResult:
    """Execute one rung through the canonical flow under its resolved config."""
    from dataclasses import replace

    from scripts.studies import grid_lines_torch_runner as runner_mod

    dataset: LadderDatasetRecipe = spec.dataset(rung.dataset)
    config = dict(rung.resolved_config)
    if seed is not None:
        config["seed"] = int(seed)

    with _stage("dataset_identity"):
        materialized = validate_ladder_npz_pair(dataset, train_npz, test_npz)

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    runner_cfg = build_runner_config(
        config, train_npz=Path(train_npz), test_npz=Path(test_npz), output_dir=work
    )

    grouping_record: Mapping[str, Any] | None = None
    with _stage("ingestion"):
        if config["loader"] == "dictionary":
            test_data, test_metadata = runner_mod.load_cached_dataset_with_metadata(
                Path(test_npz)
            )
        elif int(config["gridsize"]) == 1:
            test_data, test_metadata = load_generic_test_dict(
                Path(test_npz), dataset.recipe
            )
            grouping_record = trivial_grouping_record(
                int(np.asarray(test_data["diffraction"]).shape[0])
            )
        else:
            test_data, test_metadata, grouping_record = (
                load_grouped_generic_test_dict(
                    Path(test_npz), dataset.recipe, config, work / "test_ingest"
                )
            )

    with _stage("training"):
        if config["loader"] == "dictionary":
            model, checkpoint_root, payload = _train_dictionary(
                runner_cfg, config, Path(train_npz), test_data, test_metadata, work
            )
        else:
            model, checkpoint_root, payload = train_via_generic_loader(
                runner_cfg, config, dataset.recipe, Path(train_npz), Path(test_npz),
                work,
            )

    with _stage("checkpoint_identity"):
        from ptycho_torch import lightning_utils

        best_checkpoint = lightning_utils.find_best_checkpoint(checkpoint_root)
        if best_checkpoint is None:
            raise RuntimeExecutionError(
                "checkpoint_identity", f"no checkpoint found under {checkpoint_root}"
            )
        best_checkpoint = Path(best_checkpoint)
        checkpoint_sha256 = sha256_file(best_checkpoint)

    with _stage("inference"):
        with observe_inference_input_scales(model) as observed_scales:
            predictions = runner_mod.run_torch_inference(
                model, test_data, runner_cfg, metadata=test_metadata
            )
        patches = np.asarray(predictions)
        if patches.ndim >= 2 and patches.shape[-1] == 2 and not np.iscomplexobj(
            patches
        ):
            patches = np.asarray(runner_mod.to_complex_patches(patches))
        pre_stitch_patch_sha256 = canonical_array_sha256(patches)

    ground_truth = _ground_truth(test_data)

    canvases: dict[str, np.ndarray] = {}
    infos: dict[str, Mapping[str, Any]] = {}
    for evaluator, mode in (("historical", "grid_lines"), ("generic", "position")):
        with _stage(f"{evaluator}_stitch"):
            canvas, _, info = runner_mod._reassemble_predictions_for_metrics(
                patches,
                ground_truth,
                test_data,
                test_metadata,
                replace(runner_cfg, reassembly_mode=mode),
            )
            canvases[evaluator] = np.squeeze(np.asarray(canvas))
            infos[evaluator] = info if isinstance(info, Mapping) else {}

    gated = canvases[str(config["gated_evaluator"])]
    # No-hidden-resize assertion: the metric method sees the gated canvas at
    # exactly the ground-truth frame; a mismatch fails hard, never resizes.
    if gated.shape != ground_truth.shape:
        raise RuntimeExecutionError(
            "no_resize",
            f"gated canvas shape {gated.shape} != ground truth "
            f"{ground_truth.shape}; resizing is prohibited and would "
            "invalidate the retained-SSIM comparison",
        )

    # Physics gauge for the VarPro/count reductions: the loader's effective
    # probe when the training path recorded it, else the materialized probe.
    effective_probe_array = payload.get("effective_probe")
    if effective_probe_array is not None:
        physics_probe, probe_gauge = effective_probe_array, "loader_effective_probe"
    else:
        physics_probe, probe_gauge = test_data.get("probeGuess"), "materialized_probe"

    varpro_s1: float | None = None
    varpro_s2: float | None = None
    if config["varpro_scaling"]:
        with _stage("varpro"):
            gated, s1, s2 = apply_varpro_to_canvas(
                gated, {**test_data, "probeGuess": physics_probe}, runner_cfg, patches
            )
            varpro_s1, varpro_s2 = float(s1), float(s2)

    # Coverage of the evaluated canvas (plan checkbox 2, recorded every rung).
    canvas_coverage_fraction = float(np.mean(_finite_mask(gated)))

    with _stage("metrics"):
        from ptycho import evaluation

        metrics = evaluation.eval_reconstruction(
            gated[..., None], ground_truth[..., None], label=rung.id
        )
        amp_mae, phase_mae = _metric_pair(metrics, "mae")
        amp_ssim, phase_ssim = _metric_pair(metrics, "ssim")

    historical_sha = canonical_array_sha256(canvases["historical"])
    generic_sha = canonical_array_sha256(canvases["generic"])
    historical_mask_sha = canonical_array_sha256(_finite_mask(canvases["historical"]))
    generic_mask_sha = canonical_array_sha256(_finite_mask(canvases["generic"]))

    scan_accounting: Mapping[str, Any] | None = None
    if rung.requires_scan_accounting:
        with _stage("scan_accounting"):
            if grouping_record is None:
                grouping_record = trivial_grouping_record(
                    int(np.asarray(test_data["diffraction"]).shape[0])
                )
            scan_accounting = collect_scan_accounting(
                grouping_record, _finite_mask(gated).astype(np.float64)
            )

    reuse: bool | None = None
    train_norm_sha: str | None = None
    infer_norm_sha: str | None = None
    if rung.requires_normalization_evidence:
        with _stage("normalization_evidence"):
            inference_record = build_inference_normalization_record(
                observed_scales, test_data
            )
            reuse, train_norm_sha, infer_norm_sha = resolve_normalization_reuse(
                payload, inference_record
            )

    physics_scaling_constant = _resolve_physics_scaling_constant(config, payload)

    count_consistency: Mapping[str, Any] | None = None
    if rung.requires_count_error_evidence:
        with _stage("count_consistency"):
            s1_value, s2_value, scale_source = resolve_count_scale_operands(
                config,
                varpro_s1=varpro_s1,
                varpro_s2=varpro_s2,
                model=model,
                training_payload=payload,
            )
            count_consistency = dict(
                compute_count_consistency(
                    np.asarray(test_data["diffraction"]),
                    physics_probe,
                    patches,
                    s1=s1_value,
                    s2=s2_value,
                )
            )
            count_consistency["scale_source"] = scale_source
            count_consistency["probe_gauge"] = probe_gauge
            count_consistency["probe_scaling"] = payload.get("probe_scaling")
    effective_probe = _resolve_effective_probe(config, payload, materialized)

    return LadderRunResult(
        rung_id=rung.id,
        materialized=materialized,
        best_checkpoint=best_checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        pre_stitch_patch_sha256=pre_stitch_patch_sha256,
        historical_canvas_sha256=historical_sha,
        generic_canvas_sha256=generic_sha,
        historical_mask_sha256=historical_mask_sha,
        generic_mask_sha256=generic_mask_sha,
        canvases_equivalent=historical_sha == generic_sha,
        masks_equivalent=historical_mask_sha == generic_mask_sha,
        no_resize_asserted=True,
        gauge_handling=DECLARED_GAUGE_HANDLING,
        gated_evaluator=str(config["gated_evaluator"]),
        amp_mae=amp_mae,
        phase_mae=phase_mae,
        amp_ssim=amp_ssim,
        phase_ssim=phase_ssim,
        effective_probe_sha256=effective_probe,
        effective_probe_matches_recipe=(
            effective_probe == dataset.recipe.transformed_probe_sha256
        ),
        inference_reuses_training_normalization=reuse,
        training_normalization_sha256=train_norm_sha,
        inference_normalization_sha256=infer_norm_sha,
        varpro_applied=bool(config["varpro_scaling"]),
        varpro_s1=varpro_s1,
        varpro_s2=varpro_s2,
        scan_accounting=scan_accounting,
        canvas_coverage_fraction=canvas_coverage_fraction,
        count_consistency=count_consistency,
        physics_scaling_constant=physics_scaling_constant,
        resolved_config=config,
    )
