"""Aligned-arm inference-variant grid: patch-weighting x VarPro, cross-pipeline (Task 4).

Ports the session-scratch harnesses (``run_full_grid.py``,
``aligned_inference_variants.py``, ``xy_swap_experiment.py::build_bridge_npz``)
that produced ``.artifacts/varpro_ablation/ext_matrix_aligned/variants_summary.md``
into the repo, since ``/tmp`` scratch is volatile. Reconstructed from
``.superpowers/sdd/ext/followups-task8-step1-report.md`` (the corrected
re-baseline methodology) and ``.superpowers/sdd/ext/task-plateau-oracle-report.md``
(border-crop/norm_Y_I derivation) per ``docs/findings.md`` row
REASSEMBLY-BRIDGE-001.

For each arm (a ``grid_lines_torch_runner.py``-produced run under
``<root>/<arm>/runs/<run_subdir>/``) and each of the four
patch_weighting x varpro_scaling variants, this module:

1. Bridges the root's shared ``data/test.npz`` (grid-lines-workflow NPZ
   schema) into the schema ``ptycho_torch.dataloader.PtychoDataset`` expects
   (REASSEMBLY-BRIDGE-001 trap 1: ``coords_offsets``' two columns are
   (row, col), i.e. (iy, ix) -- see ``ptycho.workflows.grid_lines_workflow
   ._build_scan_positions`` (``coords_global[:, 0, 0, :] = iy``,
   ``coords_global[:, 0, 1, :] = ix``) -- while the dataloader wants
   (x=horizontal, y=vertical); naive in-order bridging transposes every scan
   position.
2. Rebuilds the arm's model from its recorded ``runs/.../config.json``
   ``torch_runner_config`` and loads ``runs/.../model.pt`` weights, reusing
   ``hybrid_checkpoint_inference.py``'s build/load pattern but with a
   *superset* overrides dict (every ``TorchRunnerConfig`` field, not the
   narrow hand-picked subset that silently drops ``hybrid_*``/
   ``physics_forward_mode``/``training_patch_weighting``/``cnn_output_mode``/
   ``rect_s1s2_trainable`` and produces a shape-mismatched, silently-garbage
   ``strict=False`` load -- followups-task8-step1-report.md bug #1).
3. Runs ``varpro_probe_ablation_runner.build_test_dataset``/
   ``run_inference_variant`` (the pathway that honors both
   ``patch_weighting`` and ``varpro_scaling``) to get the assembled canvas.
4. Crops the canvas's ~20px near-zero-weight edge ring (REASSEMBLY-BRIDGE-001
   trap 2; matches ``InferenceConfig.window``'s documented default,
   ``ptycho_torch/config_params.py:253``) before scoring.
5. Applies the dataset's ``norm_Y_I`` amplitude bridge (REASSEMBLY-BRIDGE-001
   trap 3) to ``*_novarpro`` variants ONLY -- ``*_varpro`` variants already
   carry their own (separately-diagnosed, non-converging) VarPro s1/s2 scale
   fit and must not be double-corrected. Applying it to *both* the scored
   canvas and the pre-VarPro diagnostics canvas is required (the omission is
   the "norm_Y_I omission bug" this module's reproduction gate exists to
   catch -- see followups-task8-step1-report.md bug #3).

Output contract per arm (identical to the 5-epoch grid):
``variants/<variant>/{metrics.json,canvas.npz}`` + arm ``variants_summary.json``,
plus a root-level ``variants_summary.md``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# Repo has no functioning editable install -- ptycho_torch/scripts.studies only
# import when the repo root is on sys.path. That happens implicitly under
# `python -m` or when cwd == repo root, but not for
# `python scripts/studies/aligned_ablation_variant_grid.py` (sys.path[0] is
# this file's own directory), which is how this script is documented to be
# invoked (mirrors varpro_probe_ablation_runner.py's own convention).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BORDER_TRIM_PX = 20  # matches InferenceConfig.window's documented default (ptycho_torch/config_params.py:253)
_OBJECT_SOURCE_KEYS = ("YY_ground_truth", "YY_full", "objectGuess")


# ---------------------------------------------------------------------------
# Pure-numpy bridge/scoring primitives (no torch import -- CPU-cheap, unit
# tested directly by tests/studies/test_aligned_ablation_variant_grid.py).
# ---------------------------------------------------------------------------

def bridge_coords(coords_offsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert grid-lines-workflow ``coords_offsets`` to dataloader-convention
    ``(xcoords, ycoords)`` 1-D float32 arrays.

    ``coords_offsets`` (shape ``(N, 1, 2, 1)``) stores column 0 = iy
    (row/vertical), column 1 = ix (col/horizontal) --
    ``grid_lines_workflow._build_scan_positions`` (``coords_global[:, 0, 0, :]
    = iy``, ``coords_global[:, 0, 1, :] = ix``). ``ptycho_torch/dataloader.py``
    wants ``xcoords`` = horizontal, ``ycoords`` = vertical, so the columns
    must be swapped, not passed through in column order
    (docs/findings.md REASSEMBLY-BRIDGE-001).
    """
    co = np.asarray(coords_offsets)
    if co.ndim == 4:
        co = co[:, 0, :, 0]
    if co.ndim != 2 or co.shape[1] != 2:
        raise ValueError(
            f"coords_offsets must be (N,1,2,1) or (N,2); got shape {np.asarray(coords_offsets).shape}"
        )
    iy = co[:, 0]
    ix = co[:, 1]
    return ix.astype(np.float32), iy.astype(np.float32)


def squeeze_singleton_dims(arr: np.ndarray) -> np.ndarray:
    """Drop trailing and leading size-1 axes beyond a 2-D core, e.g. the
    grid-lines NPZ's trailing singleton-channel axis on ``diffraction``/
    ``YY_full``/``YY_ground_truth`` (``(N,H,W,1)`` -> ``(N,H,W)``), which
    survives verbatim through ``npz_headers``/``_get_diffraction_stack`` and
    otherwise makes ``PtychoDataset``'s per-sample tensor 5-D."""
    arr = np.asarray(arr)
    while arr.ndim > 2 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def select_object_source(test_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Pick a real-valued-content object array for the bridge NPZ's
    ``objectGuess`` (``dataloader.py`` requires this key unconditionally;
    it is otherwise inert here -- ``reconstruct_image_barycentric`` never
    reads ``dataset.data_dict['objectGuess']``)."""
    for key in _OBJECT_SOURCE_KEYS:
        if key in test_data:
            return squeeze_singleton_dims(test_data[key])
    raise ValueError(
        f"test npz has none of {_OBJECT_SOURCE_KEYS} (needed to populate the "
        "bridge NPZ's 'objectGuess')"
    )


def load_truth(test_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Scoring ground truth: the grid-lines-workflow NPZ's
    ``YY_ground_truth`` (task-plateau-oracle-report.md's clipped, (270,270)
    object frame -- not ``YY_full``, the un-clipped frame)."""
    if "YY_ground_truth" not in test_data:
        raise ValueError("test npz missing 'YY_ground_truth' (scoring truth)")
    return squeeze_singleton_dims(test_data["YY_ground_truth"])


def build_bridge_arrays(test_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Pure-numpy bridge from the grid-lines-workflow NPZ schema to the
    ``ptycho_torch.dataloader.PtychoDataset`` schema (REASSEMBLY-BRIDGE-001)."""
    xcoords, ycoords = bridge_coords(test_data["coords_offsets"])
    diffraction = squeeze_singleton_dims(test_data["diffraction"]).astype(np.float32)
    if diffraction.ndim != 3:
        raise ValueError(f"Expected diffraction to squeeze to (N,H,W); got shape {diffraction.shape}")
    if xcoords.shape[0] != diffraction.shape[0]:
        raise ValueError(
            f"coords ({xcoords.shape[0]}) and diffraction ({diffraction.shape[0]}) sample counts disagree"
        )
    return {
        "xcoords": xcoords,
        "ycoords": ycoords,
        "diffraction": diffraction,
        "probeGuess": np.asarray(test_data["probeGuess"]).astype(np.complex64),
        "objectGuess": select_object_source(test_data).astype(np.complex64),
    }


def write_bridge_npz(test_npz_path: Path, out_path: Path) -> Path:
    """Read the root's grid-lines-workflow ``test.npz`` and write the bridged
    NPZ ``PtychoDataset`` can ingest. Reads only the specific keys needed
    (the source NPZ also carries a pickled ``_metadata`` object array that
    ``allow_pickle=False`` correctly refuses)."""
    with np.load(test_npz_path) as data:
        bridged = build_bridge_arrays(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **bridged)
    return out_path


def crop_border(canvas: np.ndarray, border_px: int) -> np.ndarray:
    """Crop ``border_px`` off each edge of the assembled canvas's last two
    axes -- the ~20px near-zero-weight ring ``VectorizedWeightedAccumulator``'s
    strict ``+1``-margin bounds check leaves (REASSEMBLY-BRIDGE-001 trap 2)."""
    if border_px <= 0:
        return canvas
    h, w = canvas.shape[-2], canvas.shape[-1]
    if h <= 2 * border_px or w <= 2 * border_px:
        raise ValueError(f"border_px={border_px} too large for canvas shape {canvas.shape}")
    return canvas[..., border_px:h - border_px, border_px:w - border_px]


def apply_norm_y_i_bridge(
    canvas: np.ndarray, norm_y_i: float, varpro_scaling: bool,
) -> Tuple[np.ndarray, bool]:
    """Apply the dataset's ``norm_Y_I`` amplitude rescale to ``*_novarpro``
    variants ONLY (REASSEMBLY-BRIDGE-001 trap 3): ``*_varpro`` variants
    already carry their own VarPro s1/s2 scale fit and must not be
    double-corrected. Returns ``(bridged_canvas, applied)``."""
    if varpro_scaling:
        return canvas, False
    return canvas * norm_y_i, True


# ---------------------------------------------------------------------------
# Torch-dependent model build/load + orchestration (lazy imports throughout,
# so this module -- and the pure functions above -- import cleanly under
# pytest with no torch/GPU dependency).
# ---------------------------------------------------------------------------

def torch_runner_config_from_json(config_json_path: Path) -> Any:
    """Reconstruct the arm's ``TorchRunnerConfig`` from its recorded
    ``runs/.../config.json`` (written via ``asdict(cfg)`` --
    ``grid_lines_torch_runner.py:1697``), the single source of truth for what
    the checkpoint was actually trained with."""
    from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig

    data = json.loads(config_json_path.read_text())["torch_runner_config"]
    path_fields = {"train_npz", "test_npz", "output_dir", "artifact_root"}
    kwargs: Dict[str, Any] = {}
    for field in dataclasses.fields(TorchRunnerConfig):
        if field.name not in data:
            continue
        value = data[field.name]
        if field.name in path_fields and value is not None:
            value = Path(value)
        kwargs[field.name] = value
    return TorchRunnerConfig(**kwargs)


def build_model_overrides(cfg: Any) -> Dict[str, Any]:
    """Superset overrides dict: every ``TorchRunnerConfig`` field (passed
    through ``update_existing_config``'s ``hasattr``-gated setattr, so any
    field name matching a ``ModelConfig``/``DataConfig``/``TrainingConfig``
    attribute is honored) plus the fields ``create_training_payload`` requires
    but ``TorchRunnerConfig`` does not carry.

    Fixes the bug ``hybrid_checkpoint_inference.py._build_model_for_config``'s
    narrow, hand-picked overrides dict has: it omits every ``hybrid_*``
    encoder/bottleneck field, ``physics_forward_mode``,
    ``training_patch_weighting``, ``cnn_output_mode``, ``rect_s1s2_trainable``,
    silently producing a shape-mismatched ``state_dict`` that
    ``load_state_dict(strict=False)`` "loads" anyway with every mismatched
    layer skipped (followups-task8-step1-report.md bug #1).
    """
    overrides = dataclasses.asdict(cfg)
    overrides.update(
        {
            "n_groups": 1,  # dummy: unused by model construction, only validated as present
            "epochs": 1,
            "object_big": False,  # gs1 arms only (matches hybrid_checkpoint_inference.py)
            "probe_big": False,
        }
    )
    return overrides


def build_model(cfg: Any, train_npz: Path, scratch_output_dir: Path) -> Tuple[Any, Any, Any, Any]:
    """Build a bare (non-Lightning) model for ``cfg``, mirroring
    ``hybrid_checkpoint_inference.py._build_model_for_config`` with the
    superset overrides fix. Returns ``(model, model_config, data_config,
    training_config)``."""
    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.config_factory import create_training_payload
    from ptycho_torch.config_params import InferenceConfig as PTInferenceConfig
    from ptycho_torch.generators.registry import resolve_generator

    payload = create_training_payload(
        train_data_file=train_npz,
        output_dir=scratch_output_dir,
        overrides=build_model_overrides(cfg),
        execution_config=PyTorchExecutionConfig(
            learning_rate=float(cfg.learning_rate),
            deterministic=True,
            logger_backend="none",
            enable_progress_bar=False,
            enable_checkpointing=False,
        ),
    )
    generator = resolve_generator(payload.tf_training_config)
    model = generator.build_model(
        {
            "model_config": payload.pt_model_config,
            "data_config": payload.pt_data_config,
            "training_config": payload.pt_training_config,
            "inference_config": PTInferenceConfig(),
        }
    )
    return model, payload.pt_model_config, payload.pt_data_config, payload.pt_training_config


def load_model_weights(model: Any, model_pt: Path) -> Any:
    """Load ``model.pt`` (a bare ``state_dict``, ``grid_lines_torch_runner.py
    :1734``) and fail fast on any missing/unexpected key -- a partial,
    shape-mismatched ``strict=False`` load otherwise "succeeds" while silently
    producing garbage predictions (the bug class this module's reproduction
    gate exists to catch)."""
    import torch

    state = torch.load(model_pt, map_location="cpu")
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"load_state_dict({model_pt}) had missing_keys={result.missing_keys!r} "
            f"unexpected_keys={result.unexpected_keys!r}; refusing a silently-mismatched load."
        )
    model.eval()
    return model


def resolve_run_subdir(root: Path, arm: str, run_subdir: Optional[str]) -> str:
    """Auto-detect the arm's ``runs/<run_subdir>/`` name (e.g. ``pinn_cnn`` for
    the cnn arm) from the single run directory, unless explicitly overridden."""
    if run_subdir is not None:
        return run_subdir
    runs_dir = root / arm / "runs"
    candidates = sorted(p.name for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.is_dir() else []
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot auto-detect run subdir under {runs_dir}: found {candidates!r} "
            "(expected exactly one). Pass --run-subdir explicitly."
        )
    return candidates[0]


def score_variant(
    recon: np.ndarray,
    prescale_canvas: np.ndarray,
    s1: float,
    s2: float,
    truth: np.ndarray,
    norm_y_i: float,
    *,
    varpro_scaling: bool,
    border_trim_px: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Border-crop + conditional norm_Y_I bridge, then score against
    ``truth``. Reuses ``varpro_probe_ablation_runner``'s metric helpers for
    byte-identical MAE/diagnostics conventions with the 5-epoch grid."""
    from scripts.studies import ablation_diagnostics
    from scripts.studies.varpro_probe_ablation_runner import (
        _pearson_amp,
        align_global_phase,
        compute_metrics,
    )

    recon_cropped = crop_border(recon, border_trim_px)
    prescale_cropped = crop_border(prescale_canvas, border_trim_px)

    recon_bridged, norm_applied = apply_norm_y_i_bridge(recon_cropped, norm_y_i, varpro_scaling)
    prescale_bridged, _ = apply_norm_y_i_bridge(prescale_cropped, norm_y_i, varpro_scaling)

    metrics: Dict[str, Any] = dict(compute_metrics(recon_bridged, truth))
    aligned, truth_crop = align_global_phase(recon_bridged, truth)
    metrics["amp_corr"] = _pearson_amp(aligned, truth_crop)
    metrics["s1"] = float(s1)
    metrics["s2"] = float(s2)
    metrics["border_trim_px"] = border_trim_px
    metrics["norm_Y_I_applied"] = norm_applied
    metrics.update(ablation_diagnostics.canvas_rail_diagnostics(prescale_bridged))
    metrics["diagnostics_basis"] = "prescale_canvas"
    return metrics, recon_bridged


def run_arm(
    arm: str,
    root: Path,
    output_root: Path,
    run_subdir: Optional[str],
    bridged_test_npz: Path,
    truth: np.ndarray,
    norm_y_i: float,
    device: str,
    border_trim_px: int,
) -> Dict[str, Dict[str, Any]]:
    """Load one arm's checkpoint, run all four inference variants, and write
    ``variants/<variant>/{metrics.json,canvas.npz}`` + ``variants_summary.json``."""
    from scripts.studies.varpro_probe_ablation_runner import (
        VARIANT_TABLE,
        build_test_dataset,
        run_inference_variant,
        write_canvas_npz,
        write_metrics_json,
    )

    subdir = resolve_run_subdir(root, arm, run_subdir)
    arm_run_dir = root / arm / "runs" / subdir
    config_json = arm_run_dir / "config.json"
    model_pt = arm_run_dir / "model.pt"
    if not config_json.exists():
        raise FileNotFoundError(f"Missing {config_json}")
    if not model_pt.exists():
        raise FileNotFoundError(f"Missing {model_pt}")

    cfg = torch_runner_config_from_json(config_json)
    train_npz = root / "data" / "train.npz"
    scratch_root = output_root / "_scratch" / arm

    model, model_config, data_config, training_config = build_model(cfg, train_npz, scratch_root / "payload")
    load_model_weights(model, model_pt)
    training_config = dataclasses.replace(training_config, device=device)

    dataset = build_test_dataset(bridged_test_npz, model_config, data_config, training_config, scratch_root)

    arm_dir = output_root / arm
    variant_summaries: Dict[str, Any] = {}
    for variant_name, variant_cfg in VARIANT_TABLE.items():
        variant_dir = arm_dir / "variants" / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        recon, prescale, s1, s2 = run_inference_variant(
            model, dataset, training_config, data_config, model_config,
            variant_cfg["patch_weighting"], variant_cfg["varpro_scaling"],
        )
        metrics, canvas = score_variant(
            recon, prescale, s1, s2, truth, norm_y_i,
            varpro_scaling=variant_cfg["varpro_scaling"], border_trim_px=border_trim_px,
        )
        write_canvas_npz(variant_dir / "canvas.npz", canvas)
        write_metrics_json(variant_dir / "metrics.json", metrics)
        variant_summaries[variant_name] = metrics

    (arm_dir / "variants_summary.json").write_text(json.dumps(variant_summaries, indent=2, sort_keys=True))
    return variant_summaries


# ---------------------------------------------------------------------------
# Root-level summary + CLI
# ---------------------------------------------------------------------------

METHODOLOGY_PARAGRAPH = (
    "Methodology note (see .superpowers/sdd/ext/task-plateau-oracle-report.md for full "
    "derivation): the assembled barycentric canvas has a ~20px near-zero-weight edge ring "
    "(VectorizedWeightedAccumulator's valid_mask discards outermost scan-position patches "
    "entirely), which a naive whole-canvas correlation/MAE comparison scores as gross error "
    "against real truth content there. This table crops that 20px border before scoring "
    "(matching InferenceConfig.window's own documented intent) and, for *_novarpro variants "
    "only (s1=s2=1 by construction), applies the dataset's stored norm_Y_I scale bridge "
    "(grid_lines_workflow's native evaluation path applies this multiply; "
    "reconstruct_image_barycentric has no equivalent). *_varpro variants are NOT "
    "norm_Y_I-bridged (VarPro's own s1/s2 fit is the scale-calibration mechanism there)."
)


def write_variants_summary_md(path: Path, rows: List[Tuple[str, str, Dict[str, Any]]]) -> None:
    lines = [
        "# Aligned ablation arms: inference-time patch-weighting x VarPro variant grid",
        "",
        METHODOLOGY_PARAGRAPH,
        "",
        "| arm | variant | amp_mae | phase_mae | complex_mae | amp_corr | s1 | s2 | norm_Y_I_applied |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, variant, m in rows:
        lines.append(
            f"| {arm} | {variant} | {m['amp_mae']:.4f} | {m['phase_mae']:.4f} | "
            f"{m['complex_mae']:.4f} | {m['amp_corr']:.4f} | {m['s1']:.4f} | {m['s2']:.4f} | "
            f"{m['norm_Y_I_applied']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Arm outputs root (e.g. .artifacts/varpro_ablation/ext_matrix_aligned_20ep)")
    parser.add_argument("--arms", required=True, nargs="+", help="Arm subdirectory names under --root")
    parser.add_argument("--run-subdir", default=None, help="Override auto-detected runs/<subdir> name for every arm")
    parser.add_argument("--output-root", type=Path, default=None, help="Where to write variants/ + variants_summary.* (default: --root)")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--border-trim-px", type=int, default=BORDER_TRIM_PX)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root: Path = args.root
    output_root: Path = args.output_root or root
    output_root.mkdir(parents=True, exist_ok=True)

    test_npz_path = root / "data" / "test.npz"
    with np.load(test_npz_path) as data:
        truth = load_truth(data)
        norm_y_i = float(data["norm_Y_I"])

    bridged_test_npz = write_bridge_npz(test_npz_path, output_root / "_scratch" / "bridged_test.npz")

    rows: List[Tuple[str, str, Dict[str, Any]]] = []
    for arm in args.arms:
        variant_summaries = run_arm(
            arm, root, output_root, args.run_subdir,
            bridged_test_npz, truth, norm_y_i, args.device, args.border_trim_px,
        )
        for variant_name, metrics in variant_summaries.items():
            rows.append((arm, variant_name, metrics))

    write_variants_summary_md(output_root / "variants_summary.md", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
