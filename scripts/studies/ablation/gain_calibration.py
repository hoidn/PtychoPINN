"""Amplitude-physics-gain calibration harness (plan Task 26, design §4).

Runs the EXACT rung0 reference recipe (dictionary flow, corrected emission,
seed 3, 5 epochs — resolved from the bridge-ladder spec's ``[baseline.config]``)
at explicit ``amplitude_physics_gain`` values and seals per-point evidence:
val-loss trajectory, amp/phase SSIM through the ladder's own stitch+metric
path, and decoder-output statistics (design §4 secondary instrumentation).

Predeclared decision criterion (design §4, encoded in
:func:`select_gain_rule` BEFORE any sweep run):

- Sweep (Rule A): ``gain in {1, 4, 16, 64}``.
- Quality plateau: swept gains whose amp SSIM is within
  ``PLATEAU_AMP_SSIM_TOLERANCE`` (0.02 — the Task 28 convergence-gate
  tolerance) of the sweep maximum AND >= ``AMP_SSIM_FLOOR`` (0.85, the halt
  criterion).
- Rule B (init-time self-calibration, TF ``intensity_scale`` convention):
  ``G_B = rms(observed amplitude) / rms(predicted amplitude at init, gain 1)``
  over the first ``RULE_B_SAMPLE_COUNT`` training samples in dataset order
  (deterministic, data-derived, batch-size-independent).
- Rule B is preferred iff ``G_B`` lands inside ``[min(plateau), max(plateau)]``;
  otherwise the best swept constant wins (amp SSIM, then phase SSIM, then the
  smaller gain). If no swept gain reaches the floor the calibration HALTS and
  the fix phase returns to design.

CLI (one GPU job per invocation, sequential)::

    python -m scripts.studies.ablation.gain_calibration point \
        --train-npz T --test-npz E --output-root R --gain 16 [--tag gain_16]
    python -m scripts.studies.ablation.gain_calibration init-stats \
        --train-npz T --test-npz E --output-root R
    python -m scripts.studies.ablation.gain_calibration summarize --output-root R

Contract: docs/superpowers/specs/2026-07-12-probe-rank-physics-contract-fix-design.md §4;
docs/specs/spec-ptycho-torch-probe-layout.md §3.3 (gain semantics).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .dataset_reference import validate_ladder_npz_pair
from .runtime_errors import RuntimeExecutionError, sha256_file
from .runtime_ladder_execution import (
    _finite_mask,
    _ground_truth,
    _metric_pair,
    _train_dictionary,
    build_runner_config,
)
from .runtime_ladder_spec import load_ladder_spec

__all__ = [
    "AMP_SSIM_FLOOR",
    "GAIN_SWEEP",
    "PLATEAU_AMP_SSIM_TOLERANCE",
    "BaselinePoint",
    "GainPointRequest",
    "decoder_output_statistics",
    "fixed_dictionary_batch",
    "init_scale_match_gain",
    "resolve_baseline_point",
    "rule_b_scan",
    "run_gain_point",
    "run_init_stats",
    "select_gain_rule",
    "summarize_sweep",
    "tensor_stats",
]

#: Predeclared Rule A sweep (design §4). The historical accident predicts 16.
GAIN_SWEEP = (1.0, 4.0, 16.0, 64.0)
#: Halt criterion (design §4): the chosen rule must reach this amp SSIM.
AMP_SSIM_FLOOR = 0.85
#: Plateau membership tolerance vs the sweep maximum (mirrors the Task 28
#: convergence-gate |d amp SSIM| <= 0.02).
PLATEAU_AMP_SSIM_TOLERANCE = 0.02
#: Rule B derivation: first-K training samples, dataset order, fixed chunks.
RULE_B_SAMPLE_COUNT = 256
RULE_B_CHUNK = 16
#: Decoder-statistics fixed-batch size (first samples of each container).
STATS_BATCH_SIZE = 16

_EVIDENCE_NAME = "gain_point.json"
_RULE_B_NAME = "rule_b_derivation.json"
_SUMMARY_NAME = "sweep_summary.json"
_DATASET_PROVENANCE_FIELDS = (
    "id",
    "recipe_fingerprint_sha256",
    "train_sha256",
    "test_sha256",
    "probe_sha256",
    "n_train",
    "n_test",
)


@dataclass(frozen=True)
class GainPointRequest:
    """One calibration run of the rung0 recipe at an explicit gain."""

    spec: Path
    train_npz: Path
    test_npz: Path
    output_root: Path
    gain: float
    tag: str
    epochs: int | None = None
    seed: int | None = None
    base_dir: Path | None = None


@dataclass(frozen=True)
class BaselinePoint:
    """rung0 baseline recipe resolved onto runner operands plus one gain."""

    spec: Any
    dataset: Any
    config: dict[str, Any]
    runner_cfg: Any


def resolve_baseline_point(
    spec_path: Path,
    *,
    train_npz: Path,
    test_npz: Path,
    work_dir: Path,
    gain: float,
    epochs: int | None = None,
    seed: int | None = None,
    base_dir: Path | None = None,
) -> BaselinePoint:
    """Resolve the ladder baseline (rung0) config onto a TorchRunnerConfig.

    The recipe comes verbatim from the spec's ``[baseline.config]`` (the
    dictionary flow the sealed reference used); the only deltas this harness
    may apply are the calibration gain, an optional epoch override (0 for the
    init-stats derivation run), and an optional seed override.
    """
    spec = load_ladder_spec(spec_path, base_dir=base_dir)
    config = dict(spec.baseline.config)
    if config.get("loader") != "dictionary":
        raise RuntimeExecutionError(
            "baseline_config",
            "gain calibration requires the rung0 dictionary-flow baseline; "
            f"spec baseline declares loader={config.get('loader')!r}",
        )
    if seed is not None:
        config["seed"] = int(seed)
    if epochs is not None:
        config["epochs"] = int(epochs)
    dataset = spec.dataset(spec.baseline.dataset)
    runner_cfg = build_runner_config(
        config, train_npz=Path(train_npz), test_npz=Path(test_npz),
        output_dir=Path(work_dir),
    )
    runner_cfg = replace(runner_cfg, amplitude_physics_gain=float(gain))
    return BaselinePoint(
        spec=spec, dataset=dataset, config=config, runner_cfg=runner_cfg
    )


def fixed_dictionary_batch(
    container: Mapping[str, Any], indices: Iterable[int], device: Any = None
):
    """Replicate the inline dataset's batch conventions on fixed indices.

    Mirrors ``components.PtychoLightningDataset.__getitem__`` at
    gridsize=1/C=1 under the legacy amplitude contract: channel-first images,
    ``(B, C, 1, 2)`` relative coords, unit rms/physics scaling constants and
    the documented ``(B, C, P, H, W)`` probe layout (PROBE-RANK-001 §3.1).
    """
    import torch

    idx = list(indices)
    x = torch.as_tensor(np.asarray(container["X"]))[idx]
    observed = torch.as_tensor(np.asarray(container["observed_images"]))[idx]
    coords = torch.as_tensor(np.asarray(container["coords_relative"]))[idx]
    images = x.permute(0, 3, 1, 2).contiguous().float()
    if observed.ndim == 4:
        observed = observed.permute(0, 3, 1, 2).contiguous()
    observed = observed.float()
    coords = coords.permute(0, 3, 1, 2).contiguous().float()
    probe_raw = torch.as_tensor(np.asarray(container["probe"]))
    if probe_raw.ndim != 2:
        raise RuntimeExecutionError(
            "fixed_batch", f"expected a shared (H, W) probe; got {tuple(probe_raw.shape)}"
        )
    batch = images.shape[0]
    probe = (
        probe_raw.to(torch.complex64)
        .unsqueeze(0).unsqueeze(0).unsqueeze(0)
        .expand(batch, 1, 1, -1, -1)
        .contiguous()
    )
    ones = torch.ones(batch, 1, 1, 1)
    fields = {
        "images": images,
        "observed_images": observed,
        "coords_relative": coords,
        "rms_scaling_constant": ones.clone(),
        "physics_scaling_constant": ones.clone(),
        "experiment_id": torch.zeros(batch, dtype=torch.long),
    }
    scale = ones.clone()
    if device is not None:
        fields = {k: v.to(device) for k, v in fields.items()}
        probe = probe.to(device)
        scale = scale.to(device)
    return fields, probe, scale


def tensor_stats(t: Any) -> dict[str, float]:
    """Distribution statistics for decoder-output instrumentation."""
    import torch

    flat = t.detach().reshape(-1).float().cpu()
    q = torch.quantile(flat, torch.tensor([0.01, 0.5, 0.99]))
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "rms": float(torch.sqrt((flat**2).mean())),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "p01": float(q[0]),
        "p50": float(q[1]),
        "p99": float(q[2]),
        # Absolute saturation bands: for a sigmoid amplitude head the top
        # band measures head saturation directly (Task 22 CNN question); for
        # the hybrid real/imag head they report absolute output levels.
        "frac_gt_0p5": float((flat > 0.5).float().mean()),
        "frac_gt_0p9": float((flat > 0.9).float().mean()),
        "frac_gt_0p99": float((flat > 0.99).float().mean()),
    }


def init_scale_match_gain(*, obs_sq_sum: float, pred_sq_sum: float) -> float:
    """Rule B gain: rms(observed amplitude) / rms(init predicted amplitude).

    The TF ``intensity_scale`` convention (``ptycho/diffsim.py:scale_nphotons``
    matches expected photon energy, i.e. an RMS-ratio scale constant),
    re-expressed on the torch amplitude forward at gain 1.
    """
    if not math.isfinite(obs_sq_sum) or not math.isfinite(pred_sq_sum):
        raise RuntimeExecutionError(
            "rule_b", "non-finite energy accumulators in Rule B derivation"
        )
    if pred_sq_sum <= 0.0 or obs_sq_sum <= 0.0:
        raise RuntimeExecutionError(
            "rule_b",
            "degenerate energies in Rule B derivation "
            f"(obs_sq_sum={obs_sq_sum!r}, pred_sq_sum={pred_sq_sum!r})",
        )
    return math.sqrt(obs_sq_sum / pred_sq_sum)


def _stats_model(model: Any):
    """Deterministic instrumentation copy (train-mode forwards mutate BN
    running stats; the copy keeps the sealed model handle pristine)."""
    return copy.deepcopy(model)


def _amplitude_forward(model: Any, batch, *, gain: float, seed: int):
    """Deterministic amplitude-mode training forward at an explicit gain.

    Follows the sealed step-parity convention (train mode, fixed torch seed,
    no_grad; see runtime_ladder_step_parity_cli.loss_with_gain): the gain is
    forced on the module's shared ModelConfig (read live by ForwardModel) and
    restored afterwards.
    """
    import torch

    fields, probe, scale = batch
    config = model.model_config
    original = getattr(config, "amplitude_physics_gain", 1.0)
    try:
        config.amplitude_physics_gain = float(gain)
        torch.manual_seed(int(seed))
        model.train()
        with torch.no_grad():
            pred, amp, phase = model(
                fields["images"],
                fields["coords_relative"],
                probe,
                fields["rms_scaling_constant"],
                fields["rms_scaling_constant"],
                fields["experiment_id"],
            )
            loss = float(model.compute_loss((fields, probe, scale)))
    finally:
        config.amplitude_physics_gain = original
    return pred, amp, phase, loss


def decoder_output_statistics(
    model: Any, container: Mapping[str, Any], *, label: str,
    trained_gain: float, seed: int, batch_size: int = STATS_BATCH_SIZE,
) -> dict[str, Any]:
    """Design §4 secondary instrumentation on a fixed first-``batch_size``
    batch: decoder object amplitude/phase distributions, the raw (gain-1)
    predicted-amplitude scale vs the measurement scale, and the training
    loss at the run's own gain."""
    import torch

    total = int(np.asarray(container["X"]).shape[0])
    idx = range(min(batch_size, total))
    stats_model = _stats_model(model)
    device = next(stats_model.parameters()).device
    batch = fixed_dictionary_batch(container, idx, device=device)
    pred_gain1, amp, phase, _ = _amplitude_forward(
        stats_model, batch, gain=1.0, seed=seed
    )
    _, _, _, loss_at_gain = _amplitude_forward(
        stats_model, batch, gain=trained_gain, seed=seed
    )
    observed = batch[0]["observed_images"]
    obs_rms = float(torch.sqrt((observed.detach() ** 2).mean()))
    pred_rms = float(torch.sqrt((pred_gain1.detach() ** 2).mean()))
    return {
        "label": label,
        "batch_indices": [int(idx.start), int(idx.stop)],
        "obj_amp": tensor_stats(amp),
        "obj_phase": tensor_stats(phase),
        "pred_amp_gain1": tensor_stats(pred_gain1),
        "observed_amp": tensor_stats(observed),
        "obs_rms": obs_rms,
        "pred_rms_gain1": pred_rms,
        "obs_over_pred_rms_gain1": (
            obs_rms / pred_rms if pred_rms > 0 else None
        ),
        "loss_at_trained_gain": loss_at_gain,
        "trained_gain": float(trained_gain),
    }


def rule_b_scan(
    model: Any, container: Mapping[str, Any], *, seed: int,
    sample_count: int = RULE_B_SAMPLE_COUNT, chunk: int = RULE_B_CHUNK,
) -> dict[str, Any]:
    """Rule B derivation on the INIT-state model (design §4).

    Deterministic and batch-size-independent: the first ``sample_count``
    training samples in dataset order, forwarded at gain 1 in fixed chunks
    (per-sample physics at gridsize=1, so chunking cannot change values).
    """
    import torch

    total = int(np.asarray(container["X"]).shape[0])
    count = min(sample_count, total)
    stats_model = _stats_model(model)
    device = next(stats_model.parameters()).device
    obs_sq = 0.0
    pred_sq = 0.0
    chunk_ratios: list[float] = []
    for start in range(0, count, chunk):
        idx = range(start, min(start + chunk, count))
        batch = fixed_dictionary_batch(container, idx, device=device)
        pred, _, _, _ = _amplitude_forward(stats_model, batch, gain=1.0, seed=seed)
        observed = batch[0]["observed_images"].detach()
        o = float((observed**2).sum())
        p = float((pred.detach() ** 2).sum())
        obs_sq += o
        pred_sq += p
        chunk_ratios.append(math.sqrt(o / p) if p > 0 else float("nan"))
    gain = init_scale_match_gain(obs_sq_sum=obs_sq, pred_sq_sum=pred_sq)
    return {
        "rule": "init_scale_match",
        "convention": (
            "G_B = rms(observed amplitude) / rms(predicted amplitude at "
            "init, gain 1); TF intensity_scale analogue (design §4 Rule B)"
        ),
        "sample_count": count,
        "chunk": chunk,
        "seed": int(seed),
        "gain": gain,
        "obs_sq_sum": obs_sq,
        "pred_sq_sum": pred_sq,
        "per_chunk_gain": chunk_ratios,
    }


def select_gain_rule(
    points: Sequence[Mapping[str, Any]],
    rule_b_gain: float,
    *,
    amp_floor: float = AMP_SSIM_FLOOR,
    tolerance: float = PLATEAU_AMP_SSIM_TOLERANCE,
) -> dict[str, Any]:
    """Predeclared design-§4 decision: plateau membership decides the rule."""
    pts = sorted((dict(p) for p in points), key=lambda p: float(p["gain"]))
    if not pts:
        raise RuntimeExecutionError("selection", "no sweep points provided")
    decision: dict[str, Any] = {
        "criterion": "design_2026-07-12_s4",
        "amp_ssim_floor": amp_floor,
        "plateau_tolerance": tolerance,
        "rule_b_gain": float(rule_b_gain),
        "sweep": [
            {
                "gain": float(p["gain"]),
                "amp_ssim": float(p["amp_ssim"]),
                "phase_ssim": float(p["phase_ssim"]),
            }
            for p in pts
        ],
    }
    best_amp = max(float(p["amp_ssim"]) for p in pts)
    plateau = [
        p
        for p in pts
        if float(p["amp_ssim"]) >= best_amp - tolerance
        and float(p["amp_ssim"]) >= amp_floor
    ]
    decision["plateau_gains"] = [float(p["gain"]) for p in plateau]
    if not plateau:
        decision.update(
            halt=True,
            preferred_rule=None,
            selected_gain=None,
            rule_b_in_plateau=False,
            requires_confirmation_run=False,
            reason=(
                "no swept gain reaches the amp-SSIM floor "
                f"{amp_floor}; design §4 halt criterion — fix phase returns "
                "to design"
            ),
        )
        return decision
    lo = min(float(p["gain"]) for p in plateau)
    hi = max(float(p["gain"]) for p in plateau)
    in_plateau = lo <= float(rule_b_gain) <= hi
    decision["halt"] = False
    decision["rule_b_in_plateau"] = in_plateau
    if in_plateau:
        at_swept_value = any(
            float(p["gain"]) == float(rule_b_gain) for p in plateau
        )
        decision.update(
            preferred_rule="rule_b_init_scale_match",
            selected_gain=float(rule_b_gain),
            requires_confirmation_run=not at_swept_value,
            reason=(
                "Rule B lands inside the sweep quality plateau "
                f"[{lo}, {hi}]; self-calibrating beats a magic constant "
                "(design §4)"
            ),
        )
    else:
        winner = max(
            plateau,
            key=lambda p: (
                float(p["amp_ssim"]),
                float(p["phase_ssim"]),
                -float(p["gain"]),
            ),
        )
        decision.update(
            preferred_rule="rule_a_fixed_constant",
            selected_gain=float(winner["gain"]),
            requires_confirmation_run=False,
            reason=(
                f"Rule B gain {rule_b_gain} falls outside the plateau "
                f"[{lo}, {hi}]; the swept constant wins (design §4)"
            ),
        )
    return decision


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _copy_metrics_csv(work: Path, point_dir: Path) -> str | None:
    candidates = sorted(work.glob("lightning_logs/version_*/metrics.csv"))
    if not candidates:
        return None
    target = point_dir / "metrics.csv"
    shutil.copyfile(candidates[-1], target)
    return target.name


def _evaluate_point(
    model,
    runner_cfg,
    config,
    test_data,
    test_metadata,
    rung_label,
    *,
    trim_offset,
):
    """Ladder-identical evaluation: inference, dual stitch, gated metrics."""
    from dataclasses import replace as dc_replace

    from scripts.studies import grid_lines_torch_runner as runner_mod

    predictions = runner_mod.run_torch_inference(
        model, test_data, runner_cfg, metadata=test_metadata
    )
    patches = np.asarray(predictions)
    if patches.ndim >= 2 and patches.shape[-1] == 2 and not np.iscomplexobj(patches):
        patches = np.asarray(runner_mod.to_complex_patches(patches))
    ground_truth = _ground_truth(test_data)
    canvases: dict[str, np.ndarray] = {}
    for evaluator, mode in (("historical", "grid_lines"), ("generic", "position")):
        canvas, _, _ = runner_mod._reassemble_predictions_for_metrics(
            patches,
            ground_truth,
            test_data,
            test_metadata,
            dc_replace(runner_cfg, reassembly_mode=mode),
        )
        canvases[evaluator] = np.squeeze(np.asarray(canvas))
    gated = canvases[str(config["gated_evaluator"])]
    if gated.shape != ground_truth.shape:
        raise RuntimeExecutionError(
            "no_resize",
            f"gated canvas shape {gated.shape} != ground truth "
            f"{ground_truth.shape}; resizing is prohibited",
        )
    from ptycho import evaluation

    metrics = evaluation.eval_reconstruction_explicit(
        gated[..., None],
        ground_truth[..., None],
        label=rung_label,
        trim_offset=trim_offset,
    )
    amp_mae, phase_mae = _metric_pair(metrics, "mae")
    amp_ssim, phase_ssim = _metric_pair(metrics, "ssim")
    return {
        "gated_evaluator": str(config["gated_evaluator"]),
        "amp_mae": amp_mae,
        "phase_mae": phase_mae,
        "amp_ssim": amp_ssim,
        "phase_ssim": phase_ssim,
        "canvas_coverage_fraction": float(np.mean(_finite_mask(gated))),
    }


def _load_selected_checkpoint(checkpoint_path: Path) -> Any:
    """Restore the selected state from its persisted Lightning hparams."""
    from ptycho_torch.model import PtychoPINN_Lightning

    try:
        model = PtychoPINN_Lightning.load_from_checkpoint(
            str(checkpoint_path), map_location="cpu", strict=True
        )
    except (TypeError, RuntimeError) as exc:
        raise RuntimeExecutionError(
            "checkpoint_load",
            f"selected checkpoint {checkpoint_path} could not be restored from "
            f"its persisted hyperparameters: {exc}",
        ) from exc
    model.eval()
    return model


def _seal_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _seal_atomic_create(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish immutable evidence without replacing an existing file."""
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise RuntimeExecutionError(
                "gain_point",
                f"evidence already sealed at {path}; refusing to overwrite",
            ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _run_point_common(request: GainPointRequest, *, init_stats: bool) -> dict[str, Any]:
    from scripts.studies import grid_lines_torch_runner as runner_mod

    output_root = Path(request.output_root)
    point_dir = output_root / request.tag
    evidence_path = point_dir / (_RULE_B_NAME if init_stats else _EVIDENCE_NAME)
    if evidence_path.exists():
        raise RuntimeExecutionError(
            "gain_point",
            f"evidence already sealed at {evidence_path}; refusing to "
            "overwrite (re-run under a fresh tag/output root)",
        )
    work = point_dir / "work"
    point = resolve_baseline_point(
        Path(request.spec),
        train_npz=Path(request.train_npz),
        test_npz=Path(request.test_npz),
        work_dir=work,
        gain=float(request.gain),
        epochs=0 if init_stats else request.epochs,
        seed=request.seed,
        base_dir=request.base_dir,
    )
    config, runner_cfg = point.config, point.runner_cfg
    if init_stats:
        # No checkpoint/early-stop callbacks and no logger for the 0-epoch
        # derivation run; the model handle at init is the whole product.
        runner_cfg = replace(
            runner_cfg, enable_checkpointing=False, logger_backend=None
        )
    materialized = validate_ladder_npz_pair(
        point.dataset, Path(request.train_npz), Path(request.test_npz)
    )
    point_dir.mkdir(parents=True, exist_ok=True)
    test_data, test_metadata = runner_mod.load_cached_dataset_with_metadata(
        Path(request.test_npz)
    )
    started = time.time()
    model, _, results = _train_dictionary(
        runner_cfg, config, Path(request.train_npz), test_data, test_metadata, work
    )
    duration = time.time() - started
    seed = int(config["seed"])
    history = results.get("history") or {}
    evidence: dict[str, Any] = {
        "schema_version": "gain_calibration_point_v1",
        "mode": "init_stats" if init_stats else "sweep_point",
        "tag": request.tag,
        "gain": float(request.gain),
        "seed": seed,
        "epochs": int(config["epochs"]) if not init_stats else 0,
        "recipe": "rung0_reference (ladder spec baseline.config, dictionary flow)",
        "git_commit": _git_commit(),
        "dataset": {
            "id": point.dataset.id,
            "recipe_fingerprint_sha256": materialized.recipe_fingerprint_sha256,
            "train_sha256": materialized.train_sha256,
            "test_sha256": materialized.test_sha256,
            "probe_sha256": materialized.probe_sha256,
            "n_train": materialized.n_train,
            "n_test": materialized.n_test,
        },
        "history": {
            "train_loss": list(history.get("train_loss") or []),
            # First val entry is Lightning's pre-fit sanity pass (2 batches).
            "val_loss": list(history.get("val_loss") or []),
        },
        "train_seconds": duration,
    }
    train_container = results.get("train_container") or {}
    test_container = results.get("test_container") or {}
    if init_stats:
        evidence["rule_b"] = rule_b_scan(model, train_container, seed=seed)
        evidence["decoder_stats"] = [
            decoder_output_statistics(
                model, train_container, label="train_first16",
                trained_gain=1.0, seed=seed,
            ),
            decoder_output_statistics(
                model, test_container, label="val_first16",
                trained_gain=1.0, seed=seed,
            ),
        ]
    else:
        from ptycho_torch import lightning_utils

        selected_checkpoint = lightning_utils.find_best_checkpoint(work)
        if selected_checkpoint is None:
            raise RuntimeExecutionError(
                "checkpoint_identity", f"no checkpoint found under {work}"
            )
        selected_checkpoint = Path(selected_checkpoint)
        evidence["selected_checkpoint"] = str(selected_checkpoint)
        evidence["checkpoint_sha256"] = sha256_file(selected_checkpoint)
        selected_model = _load_selected_checkpoint(selected_checkpoint)
        evidence["metrics"] = _evaluate_point(
            selected_model,
            runner_cfg,
            config,
            test_data,
            test_metadata,
            request.tag,
            trim_offset=int(point.dataset.recipe.offset),
        )
        evidence["decoder_stats"] = [
            decoder_output_statistics(
                selected_model, train_container, label="train_first16",
                trained_gain=float(request.gain), seed=seed,
            ),
            decoder_output_statistics(
                selected_model, test_container, label="val_first16",
                trained_gain=float(request.gain), seed=seed,
            ),
        ]
        evidence["metrics_csv"] = _copy_metrics_csv(work, point_dir)
    _seal_atomic_create(evidence_path, evidence)
    return evidence


def run_gain_point(request: GainPointRequest) -> dict[str, Any]:
    """Train + evaluate one sweep point of the predeclared gain sweep."""
    return _run_point_common(request, init_stats=False)


def run_init_stats(request: GainPointRequest) -> dict[str, Any]:
    """0-epoch derivation run: Rule B gain + init decoder statistics."""
    return _run_point_common(request, init_stats=True)


def _require_complete_finite_sweep(points: Sequence[Mapping[str, Any]]) -> None:
    numeric_fields = (
        "gain",
        "amp_ssim",
        "phase_ssim",
        "amp_mae",
        "phase_mae",
        "final_val_loss",
    )
    for point in points:
        for field in numeric_fields:
            try:
                finite = math.isfinite(float(point[field]))
            except (KeyError, TypeError, ValueError):
                finite = False
            if not finite:
                raise RuntimeExecutionError(
                    "summary",
                    f"sweep point {point.get('tag')!r} field {field!r} "
                    "must be finite",
                )
    gains = [float(point["gain"]) for point in points]
    if sorted(gains) != sorted(GAIN_SWEEP):
        raise RuntimeExecutionError(
            "summary",
            "predeclared Rule-A gain set must be present exactly once: "
            f"expected {list(GAIN_SWEEP)}, got {gains}",
        )


def _evidence_provenance(
    payload: Mapping[str, Any], *, expected_epochs: int, label: str
) -> dict[str, Any]:
    try:
        raw_dataset = payload["dataset"]
        recipe = payload["recipe"]
        git_commit = payload["git_commit"]
        seed = payload["seed"]
        dataset = {
            field: raw_dataset[field]
            for field in _DATASET_PROVENANCE_FIELDS
        }
        provenance = {
            "seed": seed,
            "epochs": int(payload["epochs"]),
            "recipe": recipe,
            "git_commit": git_commit,
            "dataset": dataset,
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeExecutionError(
            "summary", f"{label} provenance is incomplete: {exc}"
        ) from exc
    if provenance["epochs"] != expected_epochs:
        raise RuntimeExecutionError(
            "summary",
            f"{label} provenance requires epochs={expected_epochs}; "
            f"got {provenance['epochs']}",
        )
    if type(seed) is not int or seed != 3:
        raise RuntimeExecutionError(
            "summary",
            f"{label} provenance requires the prescribed reference seed 3; "
            f"got {provenance['seed']}",
        )
    if not isinstance(recipe, str) or not recipe:
        raise RuntimeExecutionError(
            "summary", f"{label} provenance recipe must be a nonempty string"
        )
    if not isinstance(git_commit, str) or not git_commit:
        raise RuntimeExecutionError(
            "summary", f"{label} provenance git_commit must be a nonempty string"
        )
    for field in _DATASET_PROVENANCE_FIELDS[:5]:
        value = dataset[field]
        if not isinstance(value, str) or not value:
            raise RuntimeExecutionError(
                "summary",
                f"{label} provenance dataset field {field!r} must be a "
                "nonempty string",
            )
    for field in _DATASET_PROVENANCE_FIELDS[5:]:
        value = dataset[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeExecutionError(
                "summary",
                f"{label} provenance dataset field {field!r} must be positive",
            )
    return provenance


def _require_homogeneous_provenance(
    point_payloads: Sequence[Mapping[str, Any]],
    rule_b_payload: Mapping[str, Any],
) -> dict[str, Any]:
    provenances = [
        _evidence_provenance(payload, expected_epochs=5, label="Rule A point")
        for payload in point_payloads
    ]
    common = provenances[0]
    if any(provenance != common for provenance in provenances[1:]):
        raise RuntimeExecutionError(
            "summary", "Rule A sweep points have mixed provenance"
        )

    rule_b = _evidence_provenance(
        rule_b_payload, expected_epochs=0, label="Rule B derivation"
    )
    comparable_rule_b = dict(rule_b)
    comparable_rule_b["epochs"] = 5
    if comparable_rule_b != common:
        raise RuntimeExecutionError(
            "summary", "Rule B derivation provenance does not match Rule A sweep"
        )
    try:
        derived_seed = rule_b_payload["rule_b"]["seed"]
    except (KeyError, TypeError) as exc:
        raise RuntimeExecutionError(
            "summary", f"Rule B derivation provenance is incomplete: {exc}"
        ) from exc
    if type(derived_seed) is not int or derived_seed != 3:
        raise RuntimeExecutionError(
            "summary", "Rule B derivation provenance seed is inconsistent"
        )
    return common


def _finalize_confirmation(
    decision: dict[str, Any],
    confirmations: Sequence[Mapping[str, Any]],
    confirmation_payloads: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> None:
    required = bool(decision["requires_confirmation_run"])
    decision["confirmation_required"] = required
    if not required:
        if confirmations:
            raise RuntimeExecutionError(
                "summary", "unexpected Rule B confirmation evidence"
            )
        decision.update(complete=True, confirmation_complete=True)
        return

    if not confirmations:
        decision.update(complete=False, confirmation_complete=False)
        return
    if len(confirmations) != 1:
        raise RuntimeExecutionError(
            "summary", "exactly one Rule B confirmation is permitted"
        )

    confirmation = confirmations[0]
    payload = confirmation_payloads[0]
    for field in (
        "gain",
        "amp_ssim",
        "phase_ssim",
        "amp_mae",
        "phase_mae",
        "final_val_loss",
    ):
        try:
            finite = math.isfinite(float(confirmation[field]))
        except (KeyError, TypeError, ValueError):
            finite = False
        if not finite:
            raise RuntimeExecutionError(
                "summary", f"Rule B confirmation field {field!r} must be finite"
            )
    if float(confirmation["gain"]) != float(decision["selected_gain"]):
        raise RuntimeExecutionError(
            "summary", "Rule B confirmation does not match the selected gain"
        )
    confirmation_provenance = _evidence_provenance(
        payload, expected_epochs=5, label="Rule B confirmation"
    )
    if confirmation_provenance != provenance:
        raise RuntimeExecutionError(
            "summary", "Rule B confirmation provenance does not match the sweep"
        )
    if float(confirmation["amp_ssim"]) < AMP_SSIM_FLOOR:
        raise RuntimeExecutionError(
            "summary",
            f"Rule B confirmation does not reach the amp SSIM floor "
            f"{AMP_SSIM_FLOOR}",
        )
    decision.update(
        complete=True,
        confirmation_complete=True,
        requires_confirmation_run=False,
        confirmation_result=dict(confirmation),
    )


def _checkpoint_identity(payload: Mapping[str, Any]) -> dict[str, str]:
    path_value = payload.get("selected_checkpoint")
    digest = payload.get("checkpoint_sha256")
    if not isinstance(path_value, str) or not path_value:
        raise RuntimeExecutionError(
            "summary", "sweep/confirmation point checkpoint path is required"
        )
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeExecutionError(
            "summary", "sweep/confirmation point checkpoint SHA-256 is invalid"
        )
    checkpoint_path = Path(path_value)
    if not checkpoint_path.exists():
        raise RuntimeExecutionError(
            "summary", f"checkpoint {checkpoint_path} does not exist"
        )
    if not checkpoint_path.is_file():
        raise RuntimeExecutionError(
            "summary", f"checkpoint path {checkpoint_path} is not a file"
        )
    actual = sha256_file(checkpoint_path)
    if actual != digest:
        raise RuntimeExecutionError(
            "summary",
            f"checkpoint {checkpoint_path} SHA-256 {actual} does not match "
            f"recorded {digest}",
        )
    return {"selected_checkpoint": path_value, "checkpoint_sha256": digest}


def summarize_sweep(output_root: Path) -> dict[str, Any]:
    """Collate sealed points + Rule B derivation into the selection verdict.

    Derived artifact (rewritten on each invocation, like the ladder report).
    """
    output_root = Path(output_root)
    summary_path = output_root / _SUMMARY_NAME
    summary_path.unlink(missing_ok=True)
    points = []
    point_payloads = []
    confirmations = []
    confirmation_payloads = []
    for path in sorted(output_root.glob(f"*/{_EVIDENCE_NAME}")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_identity = _checkpoint_identity(payload)
        entry = {
            "tag": payload["tag"],
            "gain": float(payload["gain"]),
            "amp_ssim": float(payload["metrics"]["amp_ssim"]),
            "phase_ssim": float(payload["metrics"]["phase_ssim"]),
            "amp_mae": float(payload["metrics"]["amp_mae"]),
            "phase_mae": float(payload["metrics"]["phase_mae"]),
            "final_val_loss": (payload["history"]["val_loss"] or [None])[-1],
            "evidence": str(path.relative_to(output_root)),
            **checkpoint_identity,
        }
        if payload["tag"].startswith("ruleb_confirm"):
            confirmations.append(entry)
            confirmation_payloads.append(payload)
        else:
            points.append(entry)
            point_payloads.append(payload)
    rule_b_path = output_root / "init_stats" / _RULE_B_NAME
    if not rule_b_path.exists():
        raise RuntimeExecutionError(
            "summary", f"missing Rule B derivation at {rule_b_path}"
        )
    rule_b = json.loads(rule_b_path.read_text(encoding="utf-8"))
    if not points:
        raise RuntimeExecutionError("summary", "no sealed sweep points found")
    _require_complete_finite_sweep(points)
    provenance = _require_homogeneous_provenance(point_payloads, rule_b)
    rule_b_gain = float(rule_b["rule_b"]["gain"])
    if not math.isfinite(rule_b_gain):
        raise RuntimeExecutionError("summary", "Rule B gain must be finite")
    decision = select_gain_rule(points, rule_b_gain)
    _finalize_confirmation(
        decision, confirmations, confirmation_payloads, provenance
    )
    summary = {
        "schema_version": "gain_calibration_summary_v1",
        "points": points,
        "rule_b": rule_b["rule_b"],
        "provenance": provenance,
        "confirmation_runs": confirmations,
        "decision": decision,
    }
    _seal_atomic(summary_path, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gain_calibration",
        description="Amplitude-physics-gain calibration (plan Task 26, design §4).",
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("point", "init-stats"):
        p = sub.add_parser(name)
        p.add_argument(
            "--spec", type=Path,
            default=Path("scripts/studies/specs/grid_lines_bridge_ladder.toml"),
        )
        p.add_argument("--train-npz", type=Path, required=True)
        p.add_argument("--test-npz", type=Path, required=True)
        p.add_argument("--output-root", type=Path, required=True)
        p.add_argument("--seed", type=int, default=None)
        if name == "point":
            p.add_argument("--gain", type=float, required=True)
            p.add_argument("--tag", default=None)
    s = sub.add_parser("summarize")
    s.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode == "summarize":
        summary = summarize_sweep(args.output_root)
        decision = summary["decision"]
        for point in summary["points"]:
            print(
                f"gain={point['gain']:<6g} amp_ssim={point['amp_ssim']:.4f} "
                f"phase_ssim={point['phase_ssim']:.4f} "
                f"final_val_loss={point['final_val_loss']}"
            )
        print(f"rule_b_gain={summary['rule_b']['gain']:.6g}")
        print(
            f"decision: rule={decision['preferred_rule']} "
            f"selected_gain={decision['selected_gain']} halt={decision['halt']} "
            f"complete={decision['complete']}"
        )
        if not decision["complete"]:
            return 4
        return 3 if decision["halt"] else 0
    if args.mode == "init-stats":
        request = GainPointRequest(
            spec=args.spec, train_npz=args.train_npz, test_npz=args.test_npz,
            output_root=args.output_root, gain=1.0, tag="init_stats",
            seed=args.seed,
        )
        evidence = run_init_stats(request)
        print(f"rule_b_gain={evidence['rule_b']['gain']:.6g}")
        return 0
    tag = args.tag or f"gain_{args.gain:g}".replace(".", "p")
    request = GainPointRequest(
        spec=args.spec, train_npz=args.train_npz, test_npz=args.test_npz,
        output_root=args.output_root, gain=args.gain, tag=tag, seed=args.seed,
    )
    evidence = run_gain_point(request)
    metrics = evidence["metrics"]
    print(
        f"gain={evidence['gain']:g} amp_ssim={metrics['amp_ssim']:.4f} "
        f"phase_ssim={metrics['phase_ssim']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
