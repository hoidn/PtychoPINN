#!/usr/bin/env python
"""Installed command-line boundary for the generic synthetic workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ptycho.simulation import object_producers as _object_producers  # noqa: E402

_PROFILE = "synthetic-lines"
_STAGE_ORDER = ("simulate", "train", "reconstruct", "evaluate")


def _parse_stages(value: str) -> tuple[str, ...]:
    stages = tuple(item.strip() for item in value.split(","))
    if not stages or any(not stage for stage in stages):
        raise argparse.ArgumentTypeError(
            "--stages must contain nonempty comma-separated stage names"
        )
    unknown = [stage for stage in stages if stage not in _STAGE_ORDER]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown stage {unknown[0]!r}; choose from {', '.join(_STAGE_ORDER)}"
        )
    if len(set(stages)) != len(stages):
        raise argparse.ArgumentTypeError("--stages must not contain duplicates")
    indices = [_STAGE_ORDER.index(stage) for stage in stages]
    if indices != sorted(indices):
        raise argparse.ArgumentTypeError("--stages must follow workflow order")
    return stages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate flat acquisitions, train a PyTorch model, strictly reload, "
            "reconstruct, and evaluate."
        ),
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--config", type=Path, help="JSON, TOML, or YAML workflow file")
    parser.add_argument("--profile", help="Named coherent workflow profile")
    parser.add_argument(
        "--stages", type=_parse_stages, help="Comma-separated stage subsequence"
    )
    parser.add_argument("--output-root", type=Path)

    simulation = parser.add_argument_group("simulation")
    simulation.add_argument("--N", dest="N", type=int)
    simulation.add_argument("--gridsize", type=int)
    simulation.add_argument("--seed", type=int)
    simulation.add_argument("--train-patterns", type=int)
    simulation.add_argument("--test-patterns", type=int)
    simulation.add_argument("--train-objects", type=int)
    simulation.add_argument("--test-objects", type=int)
    simulation.add_argument("--shared-object", action=argparse.BooleanOptionalAction)
    simulation.add_argument(
        "--frame-order-recipe",
        choices=("object-major-v1", "coordinate-major-interleaved-v1"),
    )
    simulation.add_argument(
        "--object-kind", choices=_object_producers.registered_object_kinds()
    )
    simulation.add_argument("--object-size", type=int)
    simulation.add_argument("--scan-buffer", type=int)
    simulation.add_argument(
        "--scan-position-layout",
        choices=("uniform_random", "raster", "fixed_pitch_raster"),
        help=(
            "Scan position rule; 'raster' spans the canvas and "
            "'fixed_pitch_raster' uses outer_offset/2 pitch"
        ),
    )
    simulation.add_argument("--scan-offset", type=int)
    simulation.add_argument("--outer-offset-train", type=int)
    simulation.add_argument("--outer-offset-test", type=int)
    simulation.add_argument("--photons-per-pattern", type=float)
    simulation.add_argument("--beamstop-diameter", type=float)
    simulation.add_argument(
        "--scale-contract-version",
        choices=("legacy_v1", "ci_intensity_v2"),
        help=(
            "Measurement units contract; must pair with --measurement-domain "
            "and --physics-forward-mode"
        ),
    )
    simulation.add_argument(
        "--measurement-domain",
        choices=("normalized_amplitude", "count_intensity"),
        help="Domain of the simulated detector arrays",
    )
    simulation.add_argument("--probe-source", choices=("ideal", "custom"))
    simulation.add_argument("--probe-path", type=Path)
    simulation.add_argument("--probe-transform")
    simulation.add_argument("--simulation-probe-mask-diameter", type=float)
    simulation.add_argument("--ideal-probe-scale", type=float)
    simulation.add_argument("--simulation-probe-normalization-scale", type=float)
    simulation.add_argument(
        "--patch-amplitude-normalization",
        choices=("none", "mean_patch_max"),
        help=(
            "Split-wide object gauge; mean_patch_max divides truth and "
            "diffraction amplitude by the mean per-frame patch maximum"
        ),
    )

    model = parser.add_argument_group("model")
    model.add_argument("--architecture")
    model.add_argument(
        "--model-probe-mask",
        action=argparse.BooleanOptionalAction,
    )
    model.add_argument("--model-probe-mask-diameter", type=float)
    model.add_argument("--model-probe-mask-sigma", type=float)
    model.add_argument("--fno-modes", type=int)
    model.add_argument("--fno-width", type=int)
    model.add_argument("--fno-blocks", type=int)
    model.add_argument("--fno-cnn-blocks", type=int)
    model.add_argument("--generator-output-mode")
    model.add_argument(
        "--physics-forward-mode",
        choices=("amplitude", "rectangular_scaled"),
        help=(
            "Training forward model; 'rectangular_scaled' requires the "
            "count-intensity contract and Poisson loss"
        ),
    )
    model.add_argument(
        "--cnn-output-mode",
        choices=("amp_phase", "real_imag"),
        help="Complex-output parameterization; 'rectangular_scaled' needs real_imag",
    )
    model.add_argument(
        "--rect-s1s2-init",
        choices=("ones", "dose_closure"),
        help="Initialize rectangular scales at one or by dose closure.",
    )

    training = parser.add_argument_group("training")
    training.add_argument("--train-raw-selection", type=int)
    training.add_argument("--training-groups", type=int)
    training.add_argument("--validation-groups", type=int)
    training.add_argument("--neighbor-count", type=int)
    training.add_argument(
        "--sequential-sampling", action=argparse.BooleanOptionalAction
    )
    training.add_argument("--epochs", type=int)
    training.add_argument("--batch-size", type=int)
    training.add_argument("--torch-training-seed", type=int)
    training.add_argument(
        "--batch-order-recipe",
        choices=("torch-generator-v1", "torch-implicit-july2026-v1"),
        help="Versioned training-example order; the July recipe is single-device only",
    )
    training.add_argument("--optimizer", choices=("adam", "adamw", "sgd"))
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--momentum", type=float)
    training.add_argument("--weight-decay", type=float)
    training.add_argument("--adam-beta1", type=float)
    training.add_argument("--adam-beta2", type=float)
    training.add_argument("--scheduler")
    training.add_argument("--lr-warmup-epochs", type=int)
    training.add_argument("--lr-min-ratio", type=float)
    training.add_argument("--plateau-factor", type=float)
    training.add_argument("--plateau-patience", type=int)
    training.add_argument("--plateau-min-lr", type=float)
    training.add_argument("--plateau-threshold", type=float)
    training.add_argument("--accum-steps", type=int)
    training.add_argument("--gradient-clip-val", type=float)
    training.add_argument(
        "--gradient-clip-algorithm",
        choices=("norm", "value", "agc"),
    )
    training.add_argument(
        "--torch-loss-mode",
        choices=("mae", "poisson"),
        help=(
            "Primary training objective; also sets the coupled "
            "model.loss_function and training.nll identity"
        ),
    )

    inference = parser.add_argument_group("inference")
    inference.add_argument("--groups-per-center", type=int)
    inference.add_argument("--inference-batch-size", type=int)
    inference.add_argument(
        "--varpro",
        action=argparse.BooleanOptionalAction,
        help="Fit the reconstruction to the acquisition/count gauge",
    )
    inference.add_argument(
        "--patch-weighting",
        choices=("uniform", "probe"),
        help="Patch assembly weights; tiled reconstruction requires uniform",
    )
    inference.add_argument("--pad-eval", action=argparse.BooleanOptionalAction)
    inference.add_argument("--middle-trim", type=int)
    inference.add_argument("--window", type=int)
    inference.add_argument(
        "--reconstruction-method",
        choices=("barycentric", "tiled"),
        help="Coordinate-aware barycentric assembly or strict fixed-raster tiling",
    )
    inference.add_argument(
        "--metric-crop-border",
        type=int,
        help="Symmetric border removed only from the aligned metric mask",
    )

    execution = parser.add_argument_group("execution")
    execution.add_argument(
        "--accelerator", choices=("auto", "cpu", "gpu", "cuda", "mps")
    )
    execution.add_argument("--devices", type=int)
    execution.add_argument("--strategy")
    execution.add_argument("--precision", choices=("32-true", "16-mixed", "bf16-mixed"))
    execution.add_argument("--workers", type=int)
    execution.add_argument("--logger", choices=("csv", "tensorboard", "mlflow"))
    execution.add_argument("--deterministic", action=argparse.BooleanOptionalAction)
    execution.add_argument("--pin-memory", action=argparse.BooleanOptionalAction)
    execution.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction
    )
    execution.add_argument("--prefetch-factor", type=int)
    execution.add_argument("--progress-bar", action=argparse.BooleanOptionalAction)
    execution.add_argument("--checkpointing", action=argparse.BooleanOptionalAction)
    execution.add_argument("--checkpoint-save-top-k", type=int)
    execution.add_argument("--checkpoint-monitor")
    execution.add_argument("--checkpoint-mode", choices=("min", "max"))
    execution.add_argument("--early-stop-patience", type=int)
    return parser


def parse_arguments(
    argv: list[str] | tuple[str, ...] | None = None,
) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _load_structured_config(path: Path) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            payload = json.loads(text)
        elif suffix == ".toml":
            payload = tomllib.loads(text)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(text)
        else:
            raise ValueError(
                "--config must use JSON (.json), TOML (.toml), or YAML (.yaml/.yml)"
            )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
        yaml.YAMLError,
    ) as error:
        raise ValueError(f"cannot load workflow config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"workflow config {path} must contain an object/mapping")
    return payload


def _put(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = target
    for part in path[:-1]:
        current = current.setdefault(part, {})
    current[path[-1]] = value


_ARG_PATHS: dict[str, tuple[str, ...]] = {
    "N": ("simulation", "N"),
    "gridsize": ("simulation", "gridsize"),
    "seed": ("simulation", "seed"),
    "train_patterns": ("simulation", "train_patterns"),
    "test_patterns": ("simulation", "test_patterns"),
    "train_objects": ("simulation", "train_objects"),
    "test_objects": ("simulation", "test_objects"),
    "shared_object": ("simulation", "shared_object"),
    "frame_order_recipe": ("simulation", "frame_order_recipe"),
    "object_kind": ("simulation", "object", "kind"),
    "scan_buffer": ("simulation", "scan", "buffer"),
    "scan_position_layout": ("simulation", "scan", "position_layout"),
    "scan_offset": ("simulation", "scan", "offset"),
    "outer_offset_train": ("simulation", "scan", "outer_offset_train"),
    "outer_offset_test": ("simulation", "scan", "outer_offset_test"),
    "photons_per_pattern": ("simulation", "detector", "photons_per_pattern"),
    "scale_contract_version": ("simulation", "scale_contract_version"),
    "measurement_domain": ("simulation", "measurement_domain"),
    "beamstop_diameter": ("simulation", "detector", "beamstop_diameter"),
    "probe_source": ("simulation", "probe", "source"),
    "probe_path": ("simulation", "probe", "source_path"),
    "probe_transform": ("simulation", "probe", "transform_pipeline"),
    "simulation_probe_mask_diameter": ("simulation", "probe", "mask_diameter"),
    "ideal_probe_scale": ("simulation", "probe", "ideal_scale"),
    "simulation_probe_normalization_scale": (
        "simulation",
        "probe",
        "simulation_normalization_scale",
    ),
    "patch_amplitude_normalization": (
        "simulation",
        "object",
        "patch_amplitude_normalization",
    ),
    "architecture": ("model", "architecture"),
    "model_probe_mask": ("model", "probe_mask"),
    "model_probe_mask_diameter": ("model", "probe_mask_diameter"),
    "model_probe_mask_sigma": ("model", "probe_mask_sigma"),
    "fno_modes": ("model", "fno_modes"),
    "fno_width": ("model", "fno_width"),
    "fno_blocks": ("model", "fno_blocks"),
    "fno_cnn_blocks": ("model", "fno_cnn_blocks"),
    "generator_output_mode": ("model", "generator_output_mode"),
    "physics_forward_mode": ("model", "physics_forward_mode"),
    "cnn_output_mode": ("model", "cnn_output_mode"),
    "rect_s1s2_init": ("model", "rect_s1s2_init"),
    "train_raw_selection": ("training", "train_raw_selection"),
    "training_groups": ("training", "training_groups"),
    "validation_groups": ("training", "validation_groups"),
    "neighbor_count": ("training", "neighbor_count"),
    "sequential_sampling": ("training", "sequential_sampling"),
    "epochs": ("training", "epochs"),
    "batch_size": ("training", "batch_size"),
    "torch_training_seed": ("training", "torch_training_seed"),
    "batch_order_recipe": ("training", "batch_order_recipe"),
    "optimizer": ("training", "optimizer"),
    "learning_rate": ("training", "learning_rate"),
    "momentum": ("training", "momentum"),
    "weight_decay": ("training", "weight_decay"),
    "adam_beta1": ("training", "adam_beta1"),
    "adam_beta2": ("training", "adam_beta2"),
    "scheduler": ("training", "scheduler"),
    "lr_warmup_epochs": ("training", "lr_warmup_epochs"),
    "lr_min_ratio": ("training", "lr_min_ratio"),
    "plateau_factor": ("training", "plateau_factor"),
    "plateau_patience": ("training", "plateau_patience"),
    "plateau_min_lr": ("training", "plateau_min_lr"),
    "plateau_threshold": ("training", "plateau_threshold"),
    "accum_steps": ("training", "accum_steps"),
    "gradient_clip_val": ("training", "gradient_clip_val"),
    "gradient_clip_algorithm": ("training", "gradient_clip_algorithm"),
    "torch_loss_mode": ("training", "torch_loss_mode"),
    "groups_per_center": ("inference", "groups_per_center"),
    "inference_batch_size": ("inference", "batch_size"),
    "varpro": ("inference", "varpro_scaling"),
    "patch_weighting": ("inference", "patch_weighting"),
    "pad_eval": ("inference", "pad_eval"),
    "middle_trim": ("inference", "middle_trim"),
    "window": ("inference", "window"),
    "reconstruction_method": ("inference", "reconstruction_method"),
    "metric_crop_border": ("inference", "metric_crop_border"),
    "stages": ("workflow", "stages"),
    "output_root": ("workflow", "output_root"),
    "accelerator": ("workflow", "accelerator"),
    "devices": ("workflow", "devices"),
    "strategy": ("workflow", "strategy"),
    "precision": ("workflow", "precision"),
    "workers": ("workflow", "num_workers"),
    "logger": ("workflow", "logger_backend"),
    "deterministic": ("workflow", "deterministic"),
    "pin_memory": ("workflow", "pin_memory"),
    "persistent_workers": ("workflow", "persistent_workers"),
    "prefetch_factor": ("workflow", "prefetch_factor"),
    "progress_bar": ("workflow", "enable_progress_bar"),
    "checkpointing": ("workflow", "enable_checkpointing"),
    "checkpoint_save_top_k": ("workflow", "checkpoint_save_top_k"),
    "checkpoint_monitor": ("workflow", "checkpoint_monitor_metric"),
    "checkpoint_mode": ("workflow", "checkpoint_mode"),
    "early_stop_patience": ("workflow", "early_stop_patience"),
}


def _cli_values(args: argparse.Namespace) -> dict[str, Any]:
    values = vars(args)
    patch: dict[str, Any] = {}
    for name, path in _ARG_PATHS.items():
        if name in values:
            _put(patch, path, values[name])
    if "object_size" in values:
        size = values["object_size"]
        _put(patch, ("simulation", "object", "image_size"), (size, size))
    if "torch_loss_mode" in values:
        # The three loss fields are one identity (_validate_loss_identity);
        # expand here so the CLI cannot author a contradiction.
        loss_function, nll = {
            "mae": ("MAE", False),
            "poisson": ("Poisson", True),
        }[values["torch_loss_mode"]]
        _put(patch, ("model", "loss_function"), loss_function)
        _put(patch, ("training", "nll"), nll)
    return patch


def build_pipeline_request(
    args: argparse.Namespace,
    *,
    raw_argv: tuple[str, ...] = (),
):
    from ptycho.workflows.synthetic_pipeline import SyntheticPipelineRequest

    values = vars(args)
    file_values: dict[str, Any] = {}
    file_profile = None
    if "config" in values:
        file_values = _load_structured_config(values["config"])
        if "profile" in file_values:
            file_profile = file_values.pop("profile")
            if not isinstance(file_profile, str) or not file_profile:
                raise ValueError("workflow config profile must be a nonempty string")
    profile = values.get("profile", file_profile or _PROFILE)
    return SyntheticPipelineRequest(
        profile=profile,
        file_values=file_values,
        cli_values=_cli_values(args),
        raw_argv=raw_argv,
        script_path="ptycho_synthetic",
    )


def _run_pipeline(request):
    from ptycho.workflows.synthetic_pipeline import run_synthetic_pipeline

    return run_synthetic_pipeline(request)


def main(argv: list[str] | tuple[str, ...] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = parse_arguments(raw_argv)
    request = build_pipeline_request(args, raw_argv=raw_argv)
    _run_pipeline(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
