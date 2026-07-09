"""VarPro / probe-weighting ablation harness (Task 1.5).

Trains via ``ptycho_torch.train_lightning_only::main()`` -- NOT
``ptycho_torch.train``'s ``cli_main`` -- per the decision recorded in
``docs/plans/2026-07-01-varpro-ablation-phase1-findings.md`` ("Path
equivalence") and investigated in ``.superpowers/sdd/task-1.4b-investigation.md``:
the canonical ``train.py`` path has a semantic scaling gap (missing
``rms_scaling_constant``/probe-normalization computation) that main's real
training runs never exercised, while ``train_lightning_only.py::main()`` is
the entry point main's own CLAUDE.md documents and a live DDP hotfix
(commit ``93ca0fc0``) proves was actually run in production.

For each arm this harness:
  1. Builds ``ptycho_torch.config_params`` dataclasses directly (ablation
     knobs are plain ``ModelConfig`` fields -- no CLI-flag threading needed).
  2. Calls ``train_lightning_only.main(existing_config=...)`` as a thin,
     in-process driver (no subprocess, no modification to that file).
  3. Loads the resulting Lightning checkpoint once via
     ``lightning_utils.load_checkpoint_with_configs`` and stages the held-out
     test NPZ into its own memory-mapped ``PtychoDataset`` (mirroring
     ``stage_train_dir``'s approach for the training NPZ).
  4. For each inference variant, calls
     ``ptycho_torch.reassembly.reconstruct_image_barycentric`` in-process
     with a fresh ``InferenceConfig(patch_weighting=..., varpro_scaling=...)``
     -- the pathway that actually honors both knobs on this branch (see "Fix
     wave 1" in the Task 1.5 report for why the previous CLI subprocess did
     not).
  5. Computes phase-aligned metrics on the final canvas plus degeneracy
     diagnostics on the pre-VarPro-rescale canvas (see "Fix wave 3"), writes
     per-variant artifacts, and merges shared ``summary.json``/
     ``invocation.json`` at the output root.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import numpy as np

import ablation_diagnostics
import ablation_figures
from ablation_figures import save_error_panel, save_recon_panel

if TYPE_CHECKING:
    from ptycho_torch.dataloader import PtychoDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
# Repo has no functioning editable install (its .pth finder points at a stale,
# nonexistent path) -- ptycho_torch only imports when the repo root is on
# sys.path. That happens implicitly under `python -m` or when cwd == repo
# root, but not for `python scripts/studies/varpro_probe_ablation_runner.py`
# (sys.path[0] is this file's own directory), which is how this script is
# documented to be invoked. Insert explicitly so it works either way.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Arm + variant tables (pure data -- no torch import needed to read these)
# ---------------------------------------------------------------------------

VARIANT_TABLE: Dict[str, Dict[str, Any]] = {
    "uniform_novarpro": {"patch_weighting": "uniform", "varpro_scaling": False},
    "uniform_varpro":   {"patch_weighting": "uniform", "varpro_scaling": True},
    "probe_novarpro":   {"patch_weighting": "probe",   "varpro_scaling": False},
    "probe_varpro":     {"patch_weighting": "probe",   "varpro_scaling": True},
}

_GS1_VARIANTS = ["uniform_novarpro", "uniform_varpro", "probe_novarpro", "probe_varpro"]
_GS2_VARIANTS = ["probe_varpro", "uniform_novarpro"]

ARM_TABLE: Dict[str, Dict[str, Any]] = {
    # physics_forward_mode="rectangular_scaled" AND cnn_output_mode="real_imag"
    # are pinned on every base arm below for main parity (finding
    # PORT-ABLATION-FWD-001): origin/main has neither knob -- the
    # rectangular-scaled forward is its only training forward, and its PT model
    # unconditionally builds real/imag decoder branches (recombined via
    # CombineComplexRectangular; there is no amp_phase head on main's PT side).
    # fno-stable's ModelConfig defaults are 'amplitude'
    # (ptycho_torch/config_params.py:158) and 'amp_phase'
    # (ptycho_torch/config_params.py:96); without pinning both here, every
    # "replication" arm would silently train a different physics forward and
    # head parametrization than main. The two pins are coupled on fno-stable:
    # the model fail-fasts on rectangular_scaled unless the resolved head
    # output is real_imag (ptycho_torch/model.py:1863-1870).
    "gs1_frozen": {
        "N": 64, "gridsize": 1, "training_patch_weighting": "probe",
        "rect_s1s2_trainable": False, "nphotons": 1e9,
        "physics_forward_mode": "rectangular_scaled",
        "cnn_output_mode": "real_imag",
        "variants": list(_GS1_VARIANTS),
    },
    "gs1_trainable": {
        "N": 64, "gridsize": 1, "training_patch_weighting": "probe",
        "rect_s1s2_trainable": True, "nphotons": 1e9,
        "physics_forward_mode": "rectangular_scaled",
        "cnn_output_mode": "real_imag",
        "variants": list(_GS1_VARIANTS),
    },
    "gs2_neither": {
        "N": 64, "gridsize": 2, "training_patch_weighting": "uniform",
        "rect_s1s2_trainable": False, "nphotons": 1e9,
        "physics_forward_mode": "rectangular_scaled",
        "cnn_output_mode": "real_imag",
        "variants": list(_GS2_VARIANTS),
    },
    "gs2_probe_frozen": {
        "N": 64, "gridsize": 2, "training_patch_weighting": "probe",
        "rect_s1s2_trainable": False, "nphotons": 1e9,
        "physics_forward_mode": "rectangular_scaled",
        "cnn_output_mode": "real_imag",
        "variants": list(_GS2_VARIANTS),
    },
    "gs2_probe_trainable": {
        "N": 64, "gridsize": 2, "training_patch_weighting": "probe",
        "rect_s1s2_trainable": True, "nphotons": 1e9,
        "physics_forward_mode": "rectangular_scaled",
        "cnn_output_mode": "real_imag",
        "variants": list(_GS2_VARIANTS),
    },
    "gs2_neither_n128": {
        "N": 128, "gridsize": 2, "training_patch_weighting": "uniform",
        "rect_s1s2_trainable": False, "nphotons": 1e9,
        "physics_forward_mode": "rectangular_scaled",
        "cnn_output_mode": "real_imag",
        "variants": list(_GS2_VARIANTS),
    },
    "gs2_probe_trainable_n128": {
        "N": 128, "gridsize": 2, "training_patch_weighting": "probe",
        "rect_s1s2_trainable": True, "nphotons": 1e9,
        "physics_forward_mode": "rectangular_scaled",
        "cnn_output_mode": "real_imag",
        "variants": list(_GS2_VARIANTS),
    },
    "repr_ampphase": {
        "N": 64, "gridsize": 1, "training_patch_weighting": "probe",
        "rect_s1s2_trainable": False, "nphotons": 1e9,
        "architecture": "cnn", "cnn_output_mode": "amp_phase",
        "variants": ["probe_varpro"],
    },
    "repr_realimag": {
        "N": 64, "gridsize": 1, "training_patch_weighting": "probe",
        "rect_s1s2_trainable": False, "nphotons": 1e9,
        "architecture": "cnn", "cnn_output_mode": "real_imag",
        "variants": ["probe_varpro"],
    },
}


def arm_names() -> List[str]:
    return list(ARM_TABLE.keys())


def resolve_arm(arm: str) -> Dict[str, Any]:
    if arm not in ARM_TABLE:
        raise ValueError(f"Unknown arm '{arm}'. Choices: {sorted(ARM_TABLE)}")
    return ARM_TABLE[arm]


def resolve_variants(arm: str) -> Dict[str, Dict[str, Any]]:
    cfg = resolve_arm(arm)
    return {name: VARIANT_TABLE[name] for name in cfg["variants"]}


def resolve_arm_with_overrides(
    arm: str,
    architecture: Optional[str] = None,
    cnn_output_mode: Optional[str] = None,
    N: Optional[int] = None,
    physics_forward_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve an arm's config dict, applying optional CLI overrides.

    Returns a copy of ``ARM_TABLE[arm]`` with ``architecture``/``cnn_output_mode``/
    ``N``/``physics_forward_mode`` replaced when the corresponding argument is
    not ``None``; the base entry in ``ARM_TABLE`` is left untouched.
    """
    arm_cfg = dict(resolve_arm(arm))
    if architecture is not None:
        arm_cfg["architecture"] = architecture
    if cnn_output_mode is not None:
        arm_cfg["cnn_output_mode"] = cnn_output_mode
    if N is not None:
        arm_cfg["N"] = N
    if physics_forward_mode is not None:
        arm_cfg["physics_forward_mode"] = physics_forward_mode
    return arm_cfg


def validate_n_matches_train_npz(effective_n: int, train_npz: Path) -> None:
    """Fail fast if ``effective_n`` mismatches ``train_npz``'s ``diff3d`` frame
    size, before any config/dataset construction.

    Without this check, an N mismatch surfaces only deep inside mmap probe
    allocation as a ``RuntimeError`` about tensor size mismatch (the 2026-07-07
    crash running the 128-frame ``lines_N128_train.npz`` against the arm
    table's hardcoded ``N: 64``). ``np.load`` on an ``NpzFile`` lazily
    decompresses only the accessed member, so reading just ``diff3d`` here is
    a cheap startup preflight.
    """
    with np.load(train_npz) as npz:
        frame_size = npz["diff3d"].shape[-1]
    if frame_size != effective_n:
        raise ValueError(
            f"Effective N ({effective_n}) does not match train npz "
            f"'{train_npz}' diff3d frame size ({frame_size}). "
            f"Pass --N {frame_size} to override."
        )


# ---------------------------------------------------------------------------
# Metric helpers (pure numpy, no torch dependency)
# ---------------------------------------------------------------------------

def _center_crop(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = arr.shape[-2], arr.shape[-1]
    th, tw = shape
    top = (h - th) // 2
    left = (w - tw) // 2
    return arr[..., top:top + th, left:left + tw]


def _overlap_crop(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    th = min(a.shape[-2], b.shape[-2])
    tw = min(a.shape[-1], b.shape[-1])
    return _center_crop(a, (th, tw)), _center_crop(b, (th, tw))


def align_global_phase(recon: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Phase-align ``recon`` to ``truth`` on their overlapping center crop.

    ``recon *= exp(-1j*angle(vdot(recon_crop, truth_crop)))`` per the Task 1.5
    brief, where ``vdot(a, b) := sum(a * conj(b))`` (the engineering/registration
    inner-product convention -- note this is the *opposite* argument-conjugation
    order from ``numpy.vdot(a, b) = sum(conj(a) * b)``; using numpy's own
    convention here would rotate by the correction angle's negative instead of
    cancelling it, confirmed by ``test_align_global_phase_removes_uniform_offset``).
    Returns ``(recon_aligned_crop, truth_crop)``, both cropped to the shared
    central overlap (recon/truth may differ in size).
    """
    recon_crop, truth_crop = _overlap_crop(np.asarray(recon), np.asarray(truth))
    inner = np.sum(recon_crop * np.conj(truth_crop))
    phase = np.angle(inner)
    aligned = recon_crop * np.exp(-1j * phase)
    return aligned, truth_crop


def compute_metrics(recon: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Phase-aligned complex/amplitude/phase MAE on the overlapping crop."""
    aligned, truth_crop = align_global_phase(recon, truth)
    complex_mae = float(np.mean(np.abs(aligned - truth_crop)))
    amp_mae = float(np.mean(np.abs(np.abs(aligned) - np.abs(truth_crop))))
    # Wrap phase differences to (-pi, pi] before averaging so a genuine
    # near-zero difference straddling +-pi doesn't read as ~2*pi.
    phase_diff = np.angle(np.exp(1j * (np.angle(aligned) - np.angle(truth_crop))))
    phase_mae = float(np.mean(np.abs(phase_diff)))
    return {"complex_mae": complex_mae, "amp_mae": amp_mae, "phase_mae": phase_mae}


def build_variant_metrics(
    recon: np.ndarray, truth: np.ndarray, prescale_canvas: np.ndarray, s1: float, s2: float,
    objframe_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble one variant's ``metrics.json``: MAE scores ``recon`` (final
    canvas) vs. ``truth``; degeneracy diagnostics score ``prescale_canvas``
    (pre-VarPro canvas) so a rail-saturated model isn't masked by the
    downstream rescale (Task 1.5 report, "Fix wave 3"); the existing
    (center-crop-basis) keys are kept intact for cross-run comparability.

    ``objframe_metrics`` (Task B4a), when provided, is merged in and
    ``diagnostics_basis`` is overwritten to name BOTH bases (e.g.
    ``"prescale_canvas+objframe_direct"``) so downstream readers can tell
    which numbers came from which readout -- see ``compute_objframe_metrics``.
    """
    metrics: Dict[str, Any] = dict(compute_metrics(recon, truth))
    metrics["s1"] = s1
    metrics["s2"] = s2
    metrics.update(ablation_diagnostics.canvas_rail_diagnostics(prescale_canvas))
    metrics["diagnostics_basis"] = "prescale_canvas"
    if objframe_metrics is not None:
        metrics.update(objframe_metrics)
        metrics["diagnostics_basis"] = "prescale_canvas+objframe_direct"
    return metrics


# ---------------------------------------------------------------------------
# Object-frame direct-placement metric (Task B4a): patches are placed at their
# own coords_global in the object's own pixel frame -- no scan-COM-anchored
# intermediate canvas, no center-crop of the truth -- following the
# oracle-verified pattern of recon_quality_gate.py:60-84 / diagnose_placement.py.
# Report .superpowers/sdd/ext/task-b4-report.md Sec 1c/4: the barycentric
# canvas's scan-COM anchor vs. this runner's object-center-cropped truth
# readout cost even ground-truth-perfect patches 0.33 corr / 0.28 amp MAE.
# ---------------------------------------------------------------------------

def _complex_lsq_gauge(recon: np.ndarray, truth: np.ndarray) -> complex:
    """Single complex least-squares scalar ``c`` minimizing ``||c*recon - truth||``
    (engineering convention: ``c = <recon, truth> / <recon, recon>``, matching
    ``diagnose_placement.py::gauge``). Operates on flat 1-D covered-pixel arrays."""
    denom = np.sum(np.abs(recon) ** 2)
    if denom < 1e-30:
        return 1.0 + 0.0j
    return complex(np.sum(np.conj(recon) * truth) / denom)


def _pearson_amp(a: np.ndarray, b: np.ndarray) -> float:
    """Scale-free Pearson correlation of ``|a|`` vs. ``|b|`` (real-valued)."""
    a = np.abs(a).ravel().astype(np.float64) - np.abs(a).mean()
    b = np.abs(b).ravel().astype(np.float64) - np.abs(b).mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom < 1e-30:
        return 0.0
    return float(np.sum(a * b) / denom)


def place_patches_objframe(
    patches: np.ndarray, coords_global: np.ndarray, obj_shape: Tuple[int, int], patch_size: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Place trimmed patches directly at their ``coords_global`` in the
    object's own ``obj_shape`` pixel frame, averaging by coverage count
    (no barycentric/probe weighting, no bilinear split -- integer coords per
    the oracle numbers in the B4 report).

    Args:
        patches: (n, patch_size, patch_size) complex, already trimmed to
            ``patch_size`` (the pre-reassembly ``middle_trim`` region).
        coords_global: (n, 2) with column 0 = x, column 1 = y (dataloader.py
            convention, verified by the B4 report Sec 4).
        obj_shape: (R, R) shape of the object frame to place patches into.
        patch_size: side length of each square patch.

    Returns:
        (canvas, coverage_mask, n_used, n_total): ``canvas`` is complex
        ``obj_shape``, zero outside ``coverage_mask``; ``n_used`` counts
        patches whose full footprint lies in-bounds (out-of-bounds patches
        are skipped, matching recon_quality_gate.py -- unlike the
        barycentric accumulator, this is only reachable on this synthetic
        oracle's/production's own object frame, so drops here are a
        scan-geometry fact, not a hygiene bug); ``n_total`` is ``len(patches)``.
    """
    R_y, R_x = obj_shape
    half = patch_size // 2
    n_total = patches.shape[0]
    canvas = np.zeros(obj_shape, dtype=np.complex128)
    weight = np.zeros(obj_shape, dtype=np.float64)

    cx = np.round(coords_global[:, 0]).astype(int)
    cy = np.round(coords_global[:, 1]).astype(int)

    n_used = 0
    for i in range(n_total):
        y0, x0 = cy[i] - half, cx[i] - half
        if y0 < 0 or x0 < 0 or y0 + patch_size > R_y or x0 + patch_size > R_x:
            continue
        canvas[y0:y0 + patch_size, x0:x0 + patch_size] += patches[i]
        weight[y0:y0 + patch_size, x0:x0 + patch_size] += 1.0
        n_used += 1

    coverage_mask = weight > 0
    canvas[coverage_mask] /= weight[coverage_mask]
    return canvas, coverage_mask, n_used, n_total


def compute_objframe_metrics(
    patches: np.ndarray, coords_global: np.ndarray, truth: np.ndarray, patch_size: int,
) -> Dict[str, Any]:
    """Object-frame direct-placement metrics (Task B4a deliverable 1).

    Places ``patches`` at ``coords_global`` directly in ``truth``'s own pixel
    frame, fits a single complex LSQ gauge (scale+phase) on covered pixels,
    and reports canvas-level and patch-level fidelity. Patch-level metrics
    are scale-free (per-patch Pearson) and report the per-patch LSQ scalar
    cluster (median |c|) so a gauged canvas MAE cannot mask a washed/
    low-passed reconstruction (B4 report concern 4: gauged MAE ~0.09-0.11
    persisted even though patch pearson stayed 0.5/0.3 vs. oracle 0.98).
    """
    canvas, coverage_mask, n_used, n_total = place_patches_objframe(
        patches, coords_global, truth.shape, patch_size,
    )
    if n_used == 0:
        raise ValueError(
            f"compute_objframe_metrics: 0/{n_total} patches placed in-bounds "
            "— check coords_global units/frame"
        )

    recon_cov = canvas[coverage_mask]
    truth_cov = truth[coverage_mask]
    gauge_c = _complex_lsq_gauge(recon_cov, truth_cov)
    gauged = gauge_c * recon_cov
    amp_mae_objframe_gauged = float(np.mean(np.abs(np.abs(gauged) - np.abs(truth_cov))))
    amp_pearson_objframe = _pearson_amp(recon_cov, truth_cov)

    half = patch_size // 2
    cx = np.round(coords_global[:, 0]).astype(int)
    cy = np.round(coords_global[:, 1]).astype(int)
    R_y, R_x = truth.shape
    patch_pearsons: List[float] = []
    patch_lsq_scalars: List[float] = []
    for i in range(patches.shape[0]):
        y0, x0 = cy[i] - half, cx[i] - half
        if y0 < 0 or x0 < 0 or y0 + patch_size > R_y or x0 + patch_size > R_x:
            continue
        truth_patch = truth[y0:y0 + patch_size, x0:x0 + patch_size]
        patch_pearsons.append(_pearson_amp(patches[i], truth_patch))
        patch_lsq_scalars.append(abs(_complex_lsq_gauge(patches[i].ravel(), truth_patch.ravel())))

    coverage_fraction = float(np.mean(coverage_mask))

    return {
        "amp_mae_objframe_gauged": amp_mae_objframe_gauged,
        "amp_pearson_objframe": amp_pearson_objframe,
        "patch_amp_pearson_mean": float(np.mean(patch_pearsons)) if patch_pearsons else 0.0,
        "patch_lsq_scalar_median": float(np.median(patch_lsq_scalars)) if patch_lsq_scalars else 0.0,
        "coverage_fraction": coverage_fraction,
        "n_patches_used": n_used,
        "n_patches_total": n_total,
    }


# ---------------------------------------------------------------------------
# Canvas / metrics writers (round-trippable)
# ---------------------------------------------------------------------------

def write_canvas_npz(path: Path, canvas: np.ndarray) -> None:
    np.savez(path, canvas=np.asarray(canvas).astype(np.complex64))


def read_canvas_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return data["canvas"]


def write_metrics_json(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))


def read_metrics_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


# Plotting helpers live in ablation_figures.py (F2); imported at module top.

# ---------------------------------------------------------------------------
# Training driver -- thin wrapper around train_lightning_only.py::main()
# ---------------------------------------------------------------------------

def build_configs(arm_cfg: Dict[str, Any], batch_size: int = 16, epochs: int = 50):
    """Construct the five ptycho_torch.config_params dataclasses for one arm.

    Ablation knobs (``training_patch_weighting``, ``rect_s1s2_trainable``) are
    plain ``ModelConfig`` fields -- see task-1.4b-investigation.md Q3: both
    construction paths (canonical CLI factory vs. this direct dataclass
    construction) produce byte-identical ``ModelConfig`` values for the same
    nominal knobs.
    """
    from ptycho_torch.config_params import (
        DataConfig, ModelConfig, TrainingConfig, InferenceConfig, DatagenConfig,
    )

    g = arm_cfg["gridsize"]
    channels = g * g
    data_config = DataConfig(
        N=arm_cfg["N"], grid_size=(g, g), C=channels, nphotons=arm_cfg["nphotons"],
    )
    # object_big derived from gridsize (matches both canonical config paths:
    # train.py:823 'object_big': args.gridsize > 1; utils.py:187-189 C_value > 1)
    # -- gs1 arms must NOT inherit ModelConfig's dataclass default (True).
    # architecture/cnn_output_mode/physics_forward_mode are only passed when the
    # arm specifies them, so Phase-1 arms (which omit all three) get a
    # byte-identical ModelConfig to before -- these knobs are training-time and
    # must reach ModelConfig directly (unlike patch_weighting/varpro_scaling,
    # which are inference-time and already routed via VARIANT_TABLE).
    extra = {}
    for key in ("architecture", "cnn_output_mode", "physics_forward_mode"):
        if key in arm_cfg:
            extra[key] = arm_cfg[key]
    model_config = ModelConfig(
        training_patch_weighting=arm_cfg["training_patch_weighting"],
        rect_s1s2_trainable=arm_cfg["rect_s1s2_trainable"],
        C_model=channels, C_forward=channels,
        object_big=(g > 1),
        **extra,
    )
    # torch_loss_mode='poisson': main's native loss, which requires count-convention
    # diffraction data (integer photon counts, like raw fly001.npz). The matrix
    # datasets are built in that convention by scripts/studies/make_count_datasets.py
    # (plan amendment #10). On DATA-001 normalized-amplitude data Poisson crashes on
    # integer support, and even MAE cannot train: the scale calibration assumes
    # counts, predictions start ~37x too small, and the decoder rails saturate
    # (first-matrix evidence at .artifacts/varpro_ablation/matrix).
    # strategy='auto': TrainingConfig.strategy defaults to 'ddp' on fno-stable
    # (varpro-ablation's default was 'auto'), so get_training_strategy() passes
    # the literal 'ddp' string through unchanged regardless of device count,
    # wrapping the model in DDP even on a single visible GPU. Under DDP's
    # strict unused-parameter check this in-process single-GPU harness then
    # crashes with "parameters that were not used in producing the loss" for
    # architecture branches this arm's config does not exercise. 'auto'
    # restores the harness's original single-process behavior; multi-GPU runs
    # still escalate to DDPStrategy via resolve_n_devices/get_training_strategy.
    training_config = TrainingConfig(
        batch_size=batch_size, epochs=epochs, torch_loss_mode='poisson', strategy='auto',
    )
    inference_config = InferenceConfig()
    datagen_config = DatagenConfig()
    return data_config, model_config, training_config, inference_config, datagen_config


def stage_train_dir(train_npz: Path, scratch_dir: Path) -> Path:
    """``train_lightning_only``'s data module globs every ``*.npz`` in a
    directory, pooling them -- so the training npz must be isolated (must
    not also contain the held-out test npz)."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    link = scratch_dir / train_npz.name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(Path(train_npz).resolve())
    return scratch_dir


def run_training(
    arm_cfg: Dict[str, Any], train_npz: Path, output_dir: Path,
    epochs: int, batch_size: int, run_name: str = "train",
) -> Tuple[Path, tuple]:
    """Invoke ``train_lightning_only.py::main()`` directly (in-process, not a
    CLI subprocess) with configs built from ``arm_cfg``. Per the Task 1.4b
    decision record, this is the entry point main's real training runs used
    (train.py's cli_main path has an unresolved scaling-semantics gap).
    train_lightning_only.py itself is not modified.
    """
    from ptycho_torch.train_lightning_only import main as train_main

    configs = build_configs(arm_cfg, batch_size=batch_size, epochs=epochs)
    train_dir = stage_train_dir(train_npz, output_dir / "train_data")
    run_dir = train_main(
        ptycho_dir=str(train_dir),
        existing_config=configs,
        output_dir=str(output_dir / "training_outputs"),
        run_name=run_name,
    )
    if run_dir is None:
        raise RuntimeError(
            "train_lightning_only.main() returned no run_dir "
            "(training likely failed on a non-rank-0 process or crashed silently)"
        )
    return Path(run_dir), configs


# ---------------------------------------------------------------------------
# Held-out test dataset -- mirrors stage_train_dir for the test NPZ
# ---------------------------------------------------------------------------

def build_test_dataset(test_npz: Path, model_config, data_config, training_config,
                        scratch_dir: Path) -> "PtychoDataset":
    """Stage the held-out test NPZ into its own scratch directory (mirroring
    ``stage_train_dir``) and construct a memory-mapped ``PtychoDataset`` for
    ``reconstruct_image_barycentric``, using the exact ``DataConfig``/
    ``ModelConfig``/``TrainingConfig`` the arm trained with (must match the
    checkpoint). ``remake_map=True`` forces a fresh memory map.
    """
    from ptycho_torch.dataloader import PtychoDataset

    test_dir = stage_train_dir(test_npz, scratch_dir / "test_data")
    return PtychoDataset(
        str(test_dir), model_config, data_config, training_config=training_config,
        data_dir=str(scratch_dir / "test_mmap"), remake_map=True,
    )


# ---------------------------------------------------------------------------
# Inference variants -- in-process reconstruct_image_barycentric
# ---------------------------------------------------------------------------

def run_inference_variant(
    model, dataset, training_config, data_config, model_config,
    patch_weighting: Literal["uniform", "probe"], varpro_scaling: bool,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Run one inference variant via ``reconstruct_image_barycentric`` -- the
    pathway that reads ``patch_weighting`` and gates VarPro canvas scaling on
    ``varpro_scaling`` (unlike the CLI subprocess this harness previously
    used; see module docstring / Task 1.5 report "Fix wave 1").

    Returns ``(scaled_canvas, prescale_canvas, s1, s2)``: ``prescale_canvas``
    is the pre-VarPro-rescale canvas (the 4th diagnostics-tuple element,
    ``reassembly.py:1287``) -- callers must use it, not ``scaled_canvas``,
    for degeneracy diagnostics ("Fix wave 3"; see ``build_variant_metrics``).
    ``training_config.device`` is read but never mutated (caller sets it
    once, outside the per-variant loop -- see ``run_arm``).
    """
    from ptycho_torch.config_params import InferenceConfig
    from ptycho_torch.reassembly import reconstruct_image_barycentric

    inference_config = InferenceConfig(patch_weighting=patch_weighting, varpro_scaling=varpro_scaling)

    # Declared return annotation only reflects return_diagnostics=False; the
    # True branch used here returns a 4-tuple (reassembly.py:1288 vs. 1013).
    canvas, _dataset_subset, stats, prescale_canvas = reconstruct_image_barycentric(  # type: ignore[reportAssignmentType]
        model, dataset, training_config, data_config, model_config, inference_config,
        gpu_ids=None, verbose=False, swap_detection='None', return_diagnostics=True,
    )
    # Explicit positions (not stats[-2]/stats[-1]): Task B4a appended a
    # canvas_anchor dict after s1/s2 (reassembly.py's diagnostics-tuple list
    # is now [inference_time, assembly_time, Psi_a, Psi_b, s1, s2, canvas_anchor]).
    s1, s2 = float(stats[4]), float(stats[5])
    recon = canvas.detach().cpu().numpy()
    prescale = prescale_canvas.detach().cpu().numpy()
    return recon, prescale, s1, s2


# Fixed internal chunk size for collect_objframe_patches's forward loop (below).
_OBJFRAME_FORWARD_CHUNK = 128


def collect_objframe_patches(
    model, dataset, middle_trim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run raw ``model.forward_predict`` over the full ``dataset``, trim each
    output to the central ``middle_trim`` region, and return it flattened
    alongside its ``coords_global`` -- the pre-reassembly patch/coordinate
    pair the object-frame metric (``compute_objframe_metrics``) places
    directly, bypassing the barycentric accumulator entirely (recon_quality_
    gate.py:46-63 / flux_sweep_eval.py::direct_placement_canvas pattern).
    Identical across an arm's inference variants: ``patch_weighting``/
    ``varpro_scaling`` only affect the barycentric accumulator and its
    downstream VarPro rescale, never ``forward_predict``'s raw output
    (``ptycho_torch/reassembly.py:1189`` takes no such argument) -- callers
    compute this once per arm, not once per variant. The forward runs in
    fixed-size chunks of ``_OBJFRAME_FORWARD_CHUNK`` groups to bound peak
    memory -- a single whole-dataset batch was OOM-killed three times at
    ~120 GB on a 9863-group test set.

    Returns:
        (patches, coords_global): ``patches`` is (n, middle_trim, middle_trim)
        complex; ``coords_global`` is (n, 2) with column 0 = x, column 1 = y
        (dataloader.py convention).
    """
    import torch
    from ptycho_torch.dataloader import Collate_Lightning

    n = len(dataset)
    collate = Collate_Lightning(False)
    patch_chunks = []
    coords_chunks = []
    for start in range(0, n, _OBJFRAME_FORWARD_CHUNK):
        end = min(start + _OBJFRAME_FORWARD_CHUNK, n)
        batch = collate(dataset[list(range(start, end))])
        tensor_dict, probe = batch[0], batch[1]
        with torch.no_grad():
            raw = model.forward_predict(
                tensor_dict["images"], tensor_dict["coords_relative"], probe,
                tensor_dict["rms_scaling_constant"],
            )
        patch_chunks.append(
            _center_crop(raw.numpy(), (middle_trim, middle_trim)).reshape(-1, middle_trim, middle_trim)
        )
        coords_chunks.append(tensor_dict["coords_global"].squeeze(2).numpy().reshape(-1, 2))
    patches = np.concatenate(patch_chunks, axis=0)
    coords_global = np.concatenate(coords_chunks, axis=0)
    return patches, coords_global


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_arm(
    arm: str, train_npz: Path, test_npz: Path, output_root: Path,
    smoke: bool = False, epochs: Optional[int] = None, batch_size: int = 16,
    device: str = "cpu", arm_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from ptycho_torch.lightning_utils import find_best_checkpoint, load_checkpoint_with_configs
    from ptycho_torch.model import PtychoPINN_Lightning

    if arm_cfg is None:
        arm_cfg = resolve_arm(arm)
    arm_dir = output_root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    resolved_epochs = 1 if smoke else (epochs or 50)
    run_dir, _configs = run_training(
        arm_cfg, train_npz, arm_dir, epochs=resolved_epochs, batch_size=batch_size,
    )

    ckpt_path = find_best_checkpoint(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}/checkpoints")
    # Load configs from the checkpoint itself (single source of truth for
    # what the model was actually trained with) rather than reusing the
    # in-memory configs object, which train_lightning_only.main() mutates.
    model, loaded_configs = load_checkpoint_with_configs(str(ckpt_path), PtychoPINN_Lightning, device="cpu")
    model.eval()
    data_config, model_config, training_config, _inference_config, _datagen_config = loaded_configs
    # Set device once, outside the per-variant loop, via a copy rather than
    # mutating the checkpoint-loaded training_config in place (M2).
    training_config = dataclasses.replace(training_config, device=device)

    with np.load(test_npz) as npz:
        truth = npz["objectGuess"]

    test_dataset = build_test_dataset(test_npz, model_config, data_config, training_config, arm_dir)

    # Task B4a: object-frame direct-placement patches/metrics are identical
    # across an arm's inference variants (collect_objframe_patches's
    # docstring) -- computed once here, merged into every variant's metrics.
    # middle_trim uses InferenceConfig's default (32), matching the fresh
    # InferenceConfig run_inference_variant builds per variant (it never
    # varies middle_trim -- only patch_weighting/varpro_scaling).
    from ptycho_torch.config_params import InferenceConfig

    middle_trim = InferenceConfig().middle_trim
    objframe_patches, objframe_coords = collect_objframe_patches(
        model, test_dataset, middle_trim,
    )
    objframe_metrics = compute_objframe_metrics(
        objframe_patches, objframe_coords, truth, middle_trim,
    )

    variant_summaries: Dict[str, Any] = {}
    aligned_variants: Dict[str, np.ndarray] = {}
    truth_crop: Optional[np.ndarray] = None
    for variant_name, variant_cfg in resolve_variants(arm).items():
        variant_dir = arm_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        recon, prescale_canvas, s1, s2 = run_inference_variant(
            model, test_dataset, training_config, data_config, model_config,
            variant_cfg["patch_weighting"], variant_cfg["varpro_scaling"],
        )
        write_canvas_npz(variant_dir / "canvas.npz", recon)

        # build_variant_metrics scores MAE on recon but diagnostics on
        # prescale_canvas (see its docstring, "Fix wave 3"); objframe_metrics
        # merges in the Task B4a object-frame direct-placement readout.
        metrics = build_variant_metrics(recon, truth, prescale_canvas, s1, s2, objframe_metrics)
        write_metrics_json(variant_dir / "metrics.json", metrics)

        aligned, truth_crop = align_global_phase(recon, truth)
        save_recon_panel(variant_dir / "recon_panel.png", aligned, truth_crop)
        save_error_panel(variant_dir / "error.png", aligned, truth_crop)
        aligned_variants[variant_name] = aligned

        variant_summaries[variant_name] = metrics

    # Every arm has >=1 variant (see ARM_TABLE); documents that invariant for
    # the type checker (truth_crop is otherwise Optional).
    assert truth_crop is not None, f"Arm '{arm}' has no variants to reconstruct"

    # F2: one combined comparison figure per arm (truth + every variant), in
    # addition to the per-variant panels above.
    ablation_figures.save_reconstruction_grid(arm_dir / "reconstruction_grid.png", truth_crop, aligned_variants)
    ablation_figures.save_error_grid(arm_dir / "error_grid.png", truth_crop, aligned_variants)

    # F3: fail loudly if this arm's on-disk artifacts are incomplete, contain
    # non-finite metrics, or probe/uniform canvases collapsed to the same
    # reconstruction -- makes the real matrix self-checking without GPUs.
    problems = ablation_diagnostics.validate_arm_outputs(arm_dir, list(variant_summaries))
    if problems:
        raise RuntimeError(
            f"Arm '{arm}' failed output validation ({len(problems)} problem(s)): " + "; ".join(problems)
        )

    summary = {"arm": arm, "arm_config": arm_cfg, "run_dir": str(run_dir), "variants": variant_summaries}
    (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _merge_json_dict(path: Path, key: str, value: Any) -> None:
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[key] = value
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))


def _write_invocation_record(
    output_root: Path, arm: str, train_npz: Path, test_npz: Path, smoke: bool,
    seed: Optional[int] = None,
) -> None:
    from ptycho_torch.train_lightning_only import _resolve_seed

    path = output_root / "invocation.json"
    effective_seed = seed if seed is not None else _resolve_seed()
    record = {
        "argv": sys.argv,
        "seed": effective_seed,
        "git_commit": _git_commit(),
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "datasets_provenance": str(Path(train_npz).resolve().parent / "provenance.json"),
        "smoke": smoke,
    }
    existing = json.loads(path.read_text()) if path.exists() else {"runs": {}}
    existing.setdefault("runs", {})[arm] = record
    path.write_text(json.dumps(existing, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=arm_names())
    parser.add_argument("--smoke", action="store_true", help="1 epoch (full held-out test set for inference)")
    parser.add_argument("--train-npz", required=True, type=Path)
    parser.add_argument("--test-npz", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    # Training-time overrides: route through ModelConfig via build_configs, same
    # as the arm-table keys they override. Default None means "no override" so
    # Phase-1 invocations (which never pass these flags) are unaffected.
    parser.add_argument("--architecture", default=None, help="Override the resolved arm's architecture")
    parser.add_argument(
        "--cnn-output-mode", default=None, choices=["amp_phase", "real_imag"],
        help="Override the resolved arm's cnn_output_mode",
    )
    parser.add_argument(
        "--N", type=int, default=None,
        help="Override the resolved arm's N (diffraction/patch size), e.g. 128",
    )
    parser.add_argument(
        "--physics-forward-mode", default=None, choices=["amplitude", "rectangular_scaled"],
        help="Override the resolved arm's physics_forward_mode",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Training seed. Sets PTYCHO_TORCH_SEED before training; "
             "defaults to 42 when omitted (see train_lightning_only._resolve_seed)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        os.environ["PTYCHO_TORCH_SEED"] = str(args.seed)

    arm_cfg = resolve_arm_with_overrides(
        args.arm, architecture=args.architecture, cnn_output_mode=args.cnn_output_mode,
        N=args.N, physics_forward_mode=args.physics_forward_mode,
    )
    validate_n_matches_train_npz(arm_cfg["N"], args.train_npz)
    summary = run_arm(
        args.arm, args.train_npz, args.test_npz, args.output_root,
        smoke=args.smoke, epochs=args.epochs, batch_size=args.batch_size, device=args.device,
        arm_cfg=arm_cfg,
    )

    _write_invocation_record(
        args.output_root, args.arm, args.train_npz, args.test_npz, args.smoke, seed=args.seed,
    )
    _merge_json_dict(args.output_root / "summary.json", args.arm, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
