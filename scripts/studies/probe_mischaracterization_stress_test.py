"""Probe mischaracterization stress study for reviewer revisions.

This script is intentionally study-local: it reuses the grid-lines TensorFlow
workflow helpers but keeps perturbation and manifest policy out of shared APIs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from skimage.restoration import unwrap_phase

from scripts.studies.invocation_logging import (
    capture_runtime_provenance,
    write_invocation_artifacts,
)

SCRIPT_PATH = "scripts/studies/probe_mischaracterization_stress_test.py"
TABLE2_AMP_SSIM = 0.9044216561120993
TABLE2_AMP_PSNR = 68.8864772792175
SMOKE_NOOP_MAX_ABS_DELTA = 1e-8
SMOKE_NOOP_MEAN_ABS_DELTA = 1e-10
BASELINE_AMP_SSIM_TOL = 0.03
BASELINE_AMP_PSNR_TOL_DB = 1.5
MILD_AMP_SSIM_DROP_LIMIT = 0.10
MILD_AMP_PSNR_DROP_LIMIT_DB = 3.0
MILD_CONDITION_IDS = (
    "phase_curvature_scale_0p75",
    "amplitude_blur_sigma_px_0p5",
    "phase_noise_sigma_rad_0p1pi_seed11",
)
SMOKE_ONLY_SIZE = 96
SMOKE_ONLY_OUTER_OFFSET = 32
SMOKE_ONLY_NIMGS_TRAIN = 1
SMOKE_ONLY_NIMGS_TEST = 1
SMOKE_ONLY_NEPOCHS = 1
CANONICAL_BUNDLE_NPZ = "canonical_condition_inputs.npz"
CANONICAL_BUNDLE_MANIFEST = "canonical_condition_inputs_manifest.json"
NORMALIZATION_BUNDLE_FIELDS = (
    "norm_Y_I_train_container",
    "norm_Y_I_test_container",
    "norm_Y_I_test_stitch",
    "intensity_scale_model",
)
REQUIRED_BUNDLE_FIELDS = (
    "X_train",
    "X_test",
    "Y_I_train",
    "Y_I_test",
    "Y_phi_train",
    "Y_phi_test",
    "coords_nominal_train",
    "coords_nominal_test",
    "coords_true_train",
    "coords_true_test",
    "YY_full_train",
    "YY_full_test",
    "YY_ground_truth_test",
    "probe_true",
    *NORMALIZATION_BUNDLE_FIELDS,
)
OPTIONAL_BUNDLE_FIELDS = (
    "coords_offsets_train",
    "coords_offsets_test",
    "nn_indices_train",
    "nn_indices_test",
    "global_offsets_train",
    "global_offsets_test",
    "local_offsets_train",
    "local_offsets_test",
)
@dataclass(frozen=True)
class PerturbationCondition:
    condition_id: str
    perturbation_type: str
    value: float | None = None
    seed: int | None = None
    reviewer_facing: bool = True
    renormalize_energy: bool = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if hasattr(value, "history") and isinstance(value.history, dict):
        return value.history
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "history") and isinstance(value.history, dict):
        return _jsonable(value.history)
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default, sort_keys=True))
    return path


def _package_version(distribution_name: str, module_name: str | None = None) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        if module_name is None:
            return None
    except Exception:
        if module_name is None:
            return None
    if module_name is None:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _python_version_command() -> str | None:
    try:
        return subprocess.check_output(["python", "--version"], text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def capture_study_runtime_provenance() -> dict[str, Any]:
    provenance = capture_runtime_provenance()
    provenance.update(
        {
            "python_version": platform.python_version(),
            "python_version_command": _python_version_command(),
            "git_commit": _git_commit(),
            "package_versions": {
                "tensorflow": _package_version("tensorflow", "tensorflow"),
                "numpy": _package_version("numpy", "numpy"),
                "scikit-image": _package_version("scikit-image", "skimage"),
            },
        }
    )
    return provenance


def array_sha256(arr: np.ndarray) -> str:
    """Return a dtype- and shape-aware SHA256 for a NumPy array."""
    contiguous = np.ascontiguousarray(arr)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(b"|")
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def probe_energy(probe: np.ndarray) -> float:
    return float(np.sum(np.abs(probe) ** 2))


def renormalize_probe_energy(perturbed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    target = probe_energy(reference)
    current = probe_energy(perturbed)
    if current <= 0:
        raise ValueError("cannot renormalize a zero-energy probe")
    return (np.asarray(perturbed) * math.sqrt(target / current)).astype(np.complex64)


def _format_float_token(value: float) -> str:
    text = f"{value:g}".replace(".", "p").replace("-", "m")
    return text


def build_perturbation_grid() -> list[PerturbationCondition]:
    conditions = [
        PerturbationCondition("baseline", "baseline", value=None, seed=None),
    ]
    for scale, token in ((0.75, "0p75"), (0.50, "0p50"), (0.25, "0p25")):
        conditions.append(
            PerturbationCondition(
                condition_id=f"phase_curvature_scale_{token}",
                perturbation_type="phase_curvature_scale",
                value=scale,
            )
        )
    for sigma, token in ((0.5, "0p5"), (1.0, "1p0"), (2.0, "2p0")):
        conditions.append(
            PerturbationCondition(
                condition_id=f"amplitude_blur_sigma_px_{token}",
                perturbation_type="amplitude_blur_sigma_px",
                value=sigma,
            )
        )
    for multiple, seed in ((0.1, 11), (0.2, 17), (0.4, 23)):
        conditions.append(
            PerturbationCondition(
                condition_id=f"phase_noise_sigma_rad_{_format_float_token(multiple)}pi_seed{seed}",
                perturbation_type="phase_noise_sigma_rad",
                value=multiple * math.pi,
                seed=seed,
            )
        )
    return conditions


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        from skimage.filters import gaussian

        return gaussian(image, sigma=sigma, preserve_range=True)
    return gaussian_filter(image, sigma=sigma)


def apply_probe_perturbation(
    probe: np.ndarray,
    condition: PerturbationCondition,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(probe, dtype=np.complex64)
    metadata: dict[str, Any] = {
        "condition_id": condition.condition_id,
        "perturbation_type": condition.perturbation_type,
        "value": condition.value,
        "seed": condition.seed,
        "reviewer_facing": condition.reviewer_facing,
        "source_probe_sha256": array_sha256(source),
        "source_probe_energy": probe_energy(source),
    }

    if condition.perturbation_type == "baseline":
        perturbed = np.array(source, copy=True)
    elif condition.perturbation_type == "phase_curvature_scale":
        if condition.value is None:
            raise ValueError("phase_curvature_scale requires value")
        amplitude = np.abs(source)
        phase = unwrap_phase(np.angle(source))
        phase_median = float(np.median(phase))
        scaled_phase = phase_median + float(condition.value) * (phase - phase_median)
        perturbed = amplitude * np.exp(1j * scaled_phase)
        metadata["phase_median"] = phase_median
    elif condition.perturbation_type == "amplitude_blur_sigma_px":
        if condition.value is None:
            raise ValueError("amplitude_blur_sigma_px requires value")
        amplitude = _gaussian_blur(np.abs(source), float(condition.value))
        perturbed = amplitude * np.exp(1j * np.angle(source))
    elif condition.perturbation_type == "phase_noise_sigma_rad":
        if condition.value is None or condition.seed is None:
            raise ValueError("phase_noise_sigma_rad requires value and seed")
        rng = np.random.default_rng(int(condition.seed))
        noise = rng.normal(loc=0.0, scale=float(condition.value), size=source.shape)
        noise = noise - float(np.mean(noise))
        perturbed = np.abs(source) * np.exp(1j * (np.angle(source) + noise))
        metadata["noise_mean_after_centering"] = float(np.mean(noise))
        metadata["noise_std_after_centering"] = float(np.std(noise))
    else:
        raise ValueError(f"unsupported perturbation type: {condition.perturbation_type}")

    if condition.renormalize_energy:
        perturbed = renormalize_probe_energy(perturbed, source)
        normalization_policy = "renormalize_total_energy"
    else:
        perturbed = np.asarray(perturbed, dtype=np.complex64)
        normalization_policy = "none"

    metadata.update(
        {
            "normalization_policy": normalization_policy,
            "assumed_probe_sha256": array_sha256(perturbed),
            "assumed_probe_energy": probe_energy(perturbed),
        }
    )
    return perturbed.astype(np.complex64), metadata


def save_probe_visuals(output_dir: Path, condition_id: str, probe: np.ndarray) -> dict[str, str]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    amp_path = output_dir / f"{condition_id}_amp.png"
    phase_path = output_dir / f"{condition_id}_phase.png"
    plt.imsave(amp_path, np.abs(probe), cmap="viridis")
    plt.imsave(phase_path, np.angle(probe), cmap="twilight", vmin=-math.pi, vmax=math.pi)
    return {"amp": str(amp_path), "phase": str(phase_path)}


def build_condition_manifest_entry(
    *,
    condition: PerturbationCondition,
    source_probe: np.ndarray,
    true_probe: np.ndarray,
    assumed_probe: np.ndarray,
    perturbation_metadata: dict[str, Any],
    preflight: dict[str, Any] | None = None,
    canonical_data_checksums: dict[str, Any] | None = None,
    assumed_probe_checksums: dict[str, Any] | None = None,
    condition_probe_policy: str | None = None,
) -> dict[str, Any]:
    if preflight is not None:
        canonical_data_checksums = preflight.get("canonical_data_checksums")
        assumed_probe_checksums = preflight.get("assumed_probe_checksums")
        condition_probe_policy = preflight.get("condition_probe_policy")

    entry = {
        "condition_id": condition.condition_id,
        "reviewer_facing": bool(condition.reviewer_facing),
        "source_probe_sha256": array_sha256(source_probe),
        "true_probe_sha256": array_sha256(true_probe),
        "assumed_probe_sha256": array_sha256(assumed_probe),
        "normalization_policy": perturbation_metadata.get("normalization_policy"),
        "perturbation": {
            "type": condition.perturbation_type,
            "value": condition.value,
            "seed": condition.seed,
            "renormalize_energy": condition.renormalize_energy,
        },
    }
    if canonical_data_checksums is not None:
        entry["canonical_data_checksums"] = canonical_data_checksums
    if assumed_probe_checksums is not None:
        entry["assumed_probe_checksums"] = assumed_probe_checksums
    if condition_probe_policy is not None:
        entry["condition_probe_policy"] = condition_probe_policy
    if preflight is not None:
        entry["preflight_assertions"] = {
            "passed": True,
            "condition_data_matches_canonical": True,
            "container_probes_match_assumed_probe": True,
        }
    return entry


def build_canonical_probe(
    probe_npz: Path,
    N: int,
    probe_scale_mode: str,
    probe_smoothing_sigma: float,
    probe_mask_diameter: int | None = None,
    probe_transform_pipeline: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    from ptycho.workflows.grid_lines_workflow import (
        apply_probe_mask,
        apply_probe_transform_pipeline,
        load_probe_guess,
        normalize_probe_transform_pipeline,
    )

    source_probe = load_probe_guess(Path(probe_npz))
    normalized_pipeline, normalized_steps = normalize_probe_transform_pipeline(
        target_N=int(N),
        probe_shape=source_probe.shape,
        probe_scale_mode=probe_scale_mode,
        probe_smoothing_sigma=float(probe_smoothing_sigma),
        probe_transform_pipeline=probe_transform_pipeline,
    )
    canonical_probe = apply_probe_transform_pipeline(source_probe, normalized_steps)
    canonical_probe = apply_probe_mask(canonical_probe, probe_mask_diameter)
    meta = {
        "probe_npz": str(probe_npz),
        "source_probe_shape": list(source_probe.shape),
        "source_probe_dtype": str(source_probe.dtype),
        "source_probe_sha256": array_sha256(source_probe),
        "canonical_probe_shape": list(canonical_probe.shape),
        "canonical_probe_dtype": str(canonical_probe.dtype),
        "canonical_probe_sha256": array_sha256(canonical_probe),
        "canonical_probe_energy": probe_energy(canonical_probe),
        "probe_transform_pipeline": normalized_pipeline,
        "probe_transform_steps": normalized_steps,
        "probe_mask_diameter": probe_mask_diameter,
    }
    return canonical_probe.astype(np.complex64), meta


def prepare_output_root(output_root: Path, force: bool = False) -> Path:
    output_root = Path(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        if any(output_root.iterdir()):
            if force:
                raise FileExistsError(
                    f"{output_root} already exists and is not empty; --force does not clobber runs"
                )
            raise FileExistsError(f"{output_root} already exists")
    else:
        output_root.mkdir(parents=True)
    return output_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=False, default=None)
    parser.add_argument("--probe-npz", type=Path, default=Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"))
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--gridsize", type=int, default=1)
    parser.add_argument("--nimgs-train", type=int, default=2)
    parser.add_argument("--nimgs-test", type=int, default=2)
    parser.add_argument("--nphotons", type=float, default=1e9)
    parser.add_argument("--nepochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nll-weight", type=float, default=0.0)
    parser.add_argument("--mae-weight", type=float, default=1.0)
    parser.add_argument("--realspace-weight", type=float, default=0.0)
    parser.add_argument(
        "--probe-scale-mode",
        choices=("pad_preserve", "pad_extrapolate", "interpolate", "pipeline"),
        default="pad_preserve",
    )
    parser.add_argument("--probe-smoothing-sigma", type=float, default=0.5)
    parser.add_argument("--probe-transform-pipeline", type=str, default=None)
    parser.add_argument("--probe-mask-diameter", type=int, default=None)
    parser.add_argument("--set-phi", dest="set_phi", action="store_true", default=False)
    parser.add_argument("--no-set-phi", dest="set_phi", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-size", type=int, default=SMOKE_ONLY_SIZE)
    parser.add_argument("--smoke-outer-offset", type=int, default=SMOKE_ONLY_OUTER_OFFSET)
    parser.add_argument("--smoke-nimgs-train", type=int, default=SMOKE_ONLY_NIMGS_TRAIN)
    parser.add_argument("--smoke-nimgs-test", type=int, default=SMOKE_ONLY_NIMGS_TEST)
    parser.add_argument("--smoke-nepochs", type=int, default=SMOKE_ONLY_NEPOCHS)
    parser.add_argument("--conditions", type=str, default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--adopt-rerun-baseline", action="store_true")
    parser.add_argument("--export-paper-assets", action="store_true")
    parser.add_argument("--paper-root", type=Path, default=Path("/home/ollie/Documents/ptychopinnpaper2"))
    parser.add_argument("--child-smoke-runner", action="store_true")
    parser.add_argument("--child-condition-runner", action="store_true")
    parser.add_argument("--child-request-json", type=Path, default=None)
    args = parser.parse_args(argv)
    child_mode_count = int(args.child_smoke_runner) + int(args.child_condition_runner)
    if child_mode_count > 1:
        parser.error("choose only one child runner mode")
    if child_mode_count:
        if args.child_request_json is None:
            parser.error("--child-request-json is required for child runner modes")
    elif args.output_root is None:
        parser.error("--output-root is required unless running a child mode")
    return args


def grid_config_from_args(args: argparse.Namespace):
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig

    return GridLinesConfig(
        N=args.N,
        gridsize=args.gridsize,
        output_dir=args.output_root,
        probe_npz=args.probe_npz,
        nimgs_train=args.nimgs_train,
        nimgs_test=args.nimgs_test,
        nphotons=args.nphotons,
        nepochs=args.nepochs,
        batch_size=args.batch_size,
        nll_weight=args.nll_weight,
        mae_weight=args.mae_weight,
        realspace_weight=args.realspace_weight,
        probe_smoothing_sigma=args.probe_smoothing_sigma,
        probe_mask_diameter=args.probe_mask_diameter,
        probe_source="custom",
        probe_scale_mode=args.probe_scale_mode,
        probe_transform_pipeline=args.probe_transform_pipeline,
        set_phi=args.set_phi,
    )


def _config_payload(cfg: Any) -> dict[str, Any]:
    return _jsonable(asdict(cfg))


def execution_config_from_args(args: argparse.Namespace):
    """Return the effective grid config plus run metadata.

    Reviewer-facing full runs keep the approved configuration. The standalone
    smoke-only command intentionally uses a reduced geometry and one epoch so it
    can validate the probe-consumption branch without pretending to produce
    publishable metrics.
    """
    approved_cfg = grid_config_from_args(args)
    metadata: dict[str, Any] = {
        "smoke_only_reduced_workload": False,
        "reviewer_facing_metrics": True,
        "approved_reviewer_config": _config_payload(approved_cfg),
    }
    if not args.smoke_only:
        metadata["effective_config"] = _config_payload(approved_cfg)
        return approved_cfg, metadata

    smoke_cfg = replace(
        approved_cfg,
        size=int(args.smoke_size),
        outer_offset_train=int(args.smoke_outer_offset),
        outer_offset_test=int(args.smoke_outer_offset),
        nimgs_train=int(args.smoke_nimgs_train),
        nimgs_test=int(args.smoke_nimgs_test),
        nepochs=int(args.smoke_nepochs),
    )
    metadata.update(
        {
            "smoke_only_reduced_workload": True,
            "reviewer_facing_metrics": False,
            "effective_config": _config_payload(smoke_cfg),
        }
    )
    return smoke_cfg, metadata


def discover_provenance(
    args: argparse.Namespace,
    canonical_probe_meta: dict[str, Any] | None = None,
    runtime_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probe_npz = Path(args.probe_npz)
    payload: dict[str, Any] = {
        "created_utc": _utc_now(),
        "probe_npz": str(probe_npz),
        "canonical_probe": canonical_probe_meta or {},
        "table2": {
            "paper_metrics_path": "/home/ollie/Documents/ptychopinnpaper2/data/sim_lines_4x_metrics.json",
            "gs1_custom_amp_ssim": TABLE2_AMP_SSIM,
            "gs1_custom_amp_psnr": TABLE2_AMP_PSNR,
        },
        "checkpoint_policy": "rerun_baseline_no_proven_checkpoint",
        "matched_checkpoint": None,
        "discovered_files": {},
    }
    if runtime_provenance is not None:
        payload["runtime_provenance"] = runtime_provenance
    if probe_npz.exists():
        with np.load(probe_npz) as data:
            payload["probe_npz_arrays"] = {
                key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
                for key in data.files
            }
            if "probeGuess" in data:
                payload["probe_guess_sha256"] = array_sha256(data["probeGuess"])

    gs1_root = Path(".artifacts/sim_lines_4x_metrics_2026-01-27/gs1_custom")
    payload["discovered_files"]["gs1_custom_root"] = (
        sorted(str(path) for path in gs1_root.rglob("*") if path.is_file())
        if gs1_root.exists()
        else []
    )
    checkpoint_like: list[str] = []
    for base in (Path(".artifacts"), Path("outputs")):
        if not base.exists():
            continue
        for pattern in ("*gs1_custom*/*manifest*.json", "*gs1_custom*/*config*.json", "*gs1_custom*/*.keras", "*gs1_custom*/*.h5", "*gs1_custom*/*.h5.zip"):
            checkpoint_like.extend(str(path) for path in base.glob(pattern))
    payload["discovered_files"]["checkpoint_like"] = sorted(set(checkpoint_like))
    return payload


def base_manifest(
    args: argparse.Namespace,
    conditions: list[PerturbationCondition],
    runtime_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "created_utc": _utc_now(),
        "script": SCRIPT_PATH,
        "config": _jsonable(vars(args)),
        "conditions_requested": [condition.condition_id for condition in conditions],
        "true_probe_policy": "canonical_true_probe_fixed",
        "measurement_arrays_fixed_across_conditions": True,
        "scope": {"trainable_probe_variants": False, "joint_probe_position_refinement": False},
        "stable_core_files_modified": False,
        "baseline_policy": "rerun_baseline_no_proven_checkpoint",
        "documents_read": [
            ".artifacts/revision_studies/probe_mischaracterization_second_20260413T011228Z/approved_design.md",
            ".artifacts/revision_studies/probe_mischaracterization_second_20260413T011228Z/implementation_plan.md",
            ".artifacts/revision_studies/probe_mischaracterization_second_20260413T011228Z/plan_review.json",
            ".artifacts/revision_studies/probe_mischaracterization_second_20260413T011228Z/design_review.json",
            "state/revision-study-probe-second-20260413T011228Z/revision_context.md",
            "docs/index.md",
            "docs/findings.md",
            "docs/DEVELOPER_GUIDE.md",
            "docs/TESTING_GUIDE.md",
            "docs/development/INVOCATION_LOGGING_GUIDE.md",
            "docs/COMMANDS_REFERENCE.md",
            "docs/INITIATIVE_WORKFLOW_GUIDE.md",
            "/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md",
        ],
        "plan_review_findings_addressed": [
            "NEW-PLAN-PROBE-SECOND-001",
            "NEW-PLAN-PROBE-SECOND-002",
        ],
        "condition_manifests": {},
        "child_runs": {},
        "pivots_or_stop_conditions": [],
    }
    if runtime_provenance is not None:
        manifest["runtime_provenance"] = runtime_provenance
    return manifest


def select_conditions(spec: str) -> list[PerturbationCondition]:
    conditions = build_perturbation_grid()
    if spec == "all":
        return conditions
    wanted = {item.strip() for item in spec.split(",") if item.strip()}
    by_id = {condition.condition_id: condition for condition in conditions}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise ValueError(f"unknown condition ids: {missing}")
    return [condition for condition in conditions if condition.condition_id in wanted]


def role_for_artifact(path: Path) -> str:
    name = path.name
    if name == "manifest.json":
        return "run_manifest"
    if name == "provenance_discovery.json":
        return "provenance_discovery"
    if name.startswith("invocation."):
        return "invocation"
    if name == "probe_consumption_smoke.json":
        return "probe_consumption_smoke"
    if name == CANONICAL_BUNDLE_NPZ:
        return "canonical_condition_bundle_npz"
    if name == CANONICAL_BUNDLE_MANIFEST:
        return "canonical_condition_bundle_manifest"
    if name == "child_request.json":
        return "child_request"
    if name == "child_invocation.json":
        return "child_invocation"
    if name in {"child_stdout.log", "child_stderr.log"}:
        return "child_log"
    if name in {"metrics.json", "metrics.csv"}:
        return "metrics"
    if name == "artifact_manifest.json":
        return "artifact_manifest"
    if name.endswith("_amp.png") or name.endswith("_phase.png"):
        return "probe_visual"
    if name.endswith(".png") or name.endswith(".pdf"):
        return "figure"
    if name.endswith(".npz"):
        return "array_artifact"
    return "artifact"


def write_artifact_manifest(output_root: Path) -> Path:
    manifest_path = output_root / "artifact_manifest.json"
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        item = {
            "path": str(path),
            "role": role_for_artifact(path),
            "condition_id": None,
            "created_utc": _utc_now(),
            "sha256": None if path.name == "artifact_manifest.json" else file_sha256(path),
        }
        parts = path.relative_to(output_root).parts
        if len(parts) >= 3 and parts[0] == "conditions":
            item["condition_id"] = parts[1]
        elif path.parent.name == "probes" and path.name not in {"true_probe.npz"}:
            item["condition_id"] = path.name.rsplit("_", 1)[0]
        artifacts.append(item)
    if not any(Path(item["path"]) == manifest_path for item in artifacts):
        artifacts.append(
            {
                "path": str(manifest_path),
                "role": role_for_artifact(manifest_path),
                "condition_id": None,
                "created_utc": _utc_now(),
                "sha256": None,
            }
        )
    artifacts.sort(key=lambda item: item["path"])
    return write_json(manifest_path, {"created_utc": _utc_now(), "artifacts": artifacts})


def _copy_optional_array(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return value


def clone_container_with_probe(container: Any, assumed_probe: np.ndarray):
    from ptycho.loader import PtychoDataContainer

    # Parent runs must not use this helper for TensorFlow training after the
    # canonical child bundle has been written; child runners rebuild from disk.
    return PtychoDataContainer(
        X=np.array(container._X_np, copy=True),
        Y_I=np.array(container._Y_I_np, copy=True),
        Y_phi=np.array(container._Y_phi_np, copy=True),
        norm_Y_I=container.norm_Y_I,
        YY_full=_copy_optional_array(container.YY_full),
        coords_nominal=np.array(container._coords_nominal_np, copy=True),
        coords_true=np.array(container._coords_true_np, copy=True),
        nn_indices=_copy_optional_array(container.nn_indices),
        global_offsets=_copy_optional_array(container.global_offsets),
        local_offsets=_copy_optional_array(container.local_offsets),
        probeGuess=np.asarray(assumed_probe, dtype=np.complex64),
    )


def _to_numpy_value(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _array_record(
    name: str,
    value: Any,
    source_split: str,
    required: bool,
    absent_reason: str | None = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    array = _to_numpy_value(value)
    if array is None:
        return (
            {
                "name": name,
                "dtype": None,
                "shape": None,
                "checksum": None,
                "source_split": source_split,
                "required": bool(required),
                "status": "absent",
                "present": False,
                "absent_reason": absent_reason or f"source field {name} is absent",
            },
            None,
        )
    copied = np.array(array, copy=True)
    return (
        {
            "name": name,
            "dtype": str(copied.dtype),
            "shape": list(copied.shape),
            "checksum": array_sha256(copied),
            "source_split": source_split,
            "required": bool(required),
            "status": "present",
            "present": True,
            "absent_reason": None,
        },
        copied,
    )


def _required_container_array(container: Any, attr: str, fallback: Any) -> Any:
    return getattr(container, attr, fallback)


def _optional_container_value(container: Any, attr: str) -> Any:
    return getattr(container, attr, None)


def _bundle_constructor_mapping() -> dict[str, dict[str, str]]:
    return {
        "train": {
            "X": "X_train",
            "Y_I": "Y_I_train",
            "Y_phi": "Y_phi_train",
            "norm_Y_I": "norm_Y_I_train_container",
            "YY_full": "YY_full_train",
            "coords_nominal": "coords_nominal_train",
            "coords_true": "coords_true_train",
            "nn_indices": "nn_indices_train",
            "global_offsets": "global_offsets_train",
            "local_offsets": "local_offsets_train",
            "probeGuess": "condition assumed_probe",
        },
        "test": {
            "X": "X_test",
            "Y_I": "Y_I_test",
            "Y_phi": "Y_phi_test",
            "norm_Y_I": "norm_Y_I_test_container",
            "YY_full": "YY_full_test",
            "coords_nominal": "coords_nominal_test",
            "coords_true": "coords_true_test",
            "nn_indices": "nn_indices_test",
            "global_offsets": "global_offsets_test",
            "local_offsets": "local_offsets_test",
            "probeGuess": "condition assumed_probe",
        },
    }


def _normalization_alias_metadata(fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_checksum: dict[str, list[str]] = {}
    for name in NORMALIZATION_BUNDLE_FIELDS:
        checksum = fields.get(name, {}).get("checksum")
        if checksum is not None:
            by_checksum.setdefault(str(checksum), []).append(name)
    return {
        "by_checksum": by_checksum,
        "aliased_field_groups": [names for names in by_checksum.values() if len(names) > 1],
        "all_fields_distinct_by_value": all(len(names) == 1 for names in by_checksum.values()),
    }


def build_canonical_condition_bundle_payload(
    sim: dict[str, Any],
    true_probe: np.ndarray,
    cfg: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_container = sim["train"]["container"]
    test_container = sim["test"]["container"]
    arrays: dict[str, np.ndarray] = {}
    fields: dict[str, dict[str, Any]] = {}

    def add_field(
        name: str,
        value: Any,
        source_split: str,
        *,
        required: bool,
        absent_reason: str | None = None,
    ) -> None:
        record, array = _array_record(name, value, source_split, required, absent_reason)
        fields[name] = record
        if array is not None:
            arrays[name] = array

    add_field("X_train", _required_container_array(train_container, "_X_np", sim["train"]["X"]), "train", required=True)
    add_field("X_test", _required_container_array(test_container, "_X_np", sim["test"]["X"]), "test", required=True)
    add_field("Y_I_train", _required_container_array(train_container, "_Y_I_np", sim["train"].get("Y_I")), "train", required=True)
    add_field("Y_I_test", _required_container_array(test_container, "_Y_I_np", sim["test"].get("Y_I")), "test", required=True)
    add_field("Y_phi_train", _required_container_array(train_container, "_Y_phi_np", sim["train"].get("Y_phi")), "train", required=True)
    add_field("Y_phi_test", _required_container_array(test_container, "_Y_phi_np", sim["test"].get("Y_phi")), "test", required=True)
    add_field(
        "coords_nominal_train",
        _required_container_array(train_container, "_coords_nominal_np", sim["train"]["coords_nominal"]),
        "train",
        required=True,
    )
    add_field(
        "coords_nominal_test",
        _required_container_array(test_container, "_coords_nominal_np", sim["test"]["coords_nominal"]),
        "test",
        required=True,
    )
    add_field(
        "coords_true_train",
        _required_container_array(train_container, "_coords_true_np", sim["train"]["coords_true"]),
        "train",
        required=True,
    )
    add_field(
        "coords_true_test",
        _required_container_array(test_container, "_coords_true_np", sim["test"]["coords_true"]),
        "test",
        required=True,
    )
    add_field(
        "YY_full_train",
        _optional_container_value(train_container, "YY_full"),
        "train",
        required=True,
        absent_reason="source container YY_full is absent",
    )
    add_field(
        "YY_full_test",
        _optional_container_value(test_container, "YY_full"),
        "test",
        required=True,
        absent_reason="source container YY_full is absent",
    )
    add_field("YY_ground_truth_test", sim["test"].get("YY_ground_truth"), "test", required=True)
    add_field("coords_offsets_train", sim["train"].get("coords_offsets"), "train", required=False, absent_reason="source sim train coords_offsets is absent")
    add_field("coords_offsets_test", sim["test"].get("coords_offsets"), "test", required=False, absent_reason="source sim test coords_offsets is absent")
    for attr in ("nn_indices", "global_offsets", "local_offsets"):
        add_field(
            f"{attr}_train",
            _optional_container_value(train_container, attr),
            "train",
            required=False,
            absent_reason=f"source container {attr} is absent",
        )
        add_field(
            f"{attr}_test",
            _optional_container_value(test_container, attr),
            "test",
            required=False,
            absent_reason=f"source container {attr} is absent",
        )
    add_field("probe_true", true_probe, "global", required=True)
    add_field("norm_Y_I_train_container", train_container.norm_Y_I, "train", required=True)
    add_field("norm_Y_I_test_container", test_container.norm_Y_I, "test", required=True)
    add_field("norm_Y_I_test_stitch", sim["test"].get("norm_Y_I"), "test", required=True)
    add_field("intensity_scale_model", sim.get("intensity_scale"), "global", required=True)

    manifest = {
        "created_utc": _utc_now(),
        "schema": "probe_mischaracterization_process_isolation_bundle_v1",
        "npz_filename": CANONICAL_BUNDLE_NPZ,
        "fields": fields,
        "required_fields": list(REQUIRED_BUNDLE_FIELDS),
        "optional_fields": list(OPTIONAL_BUNDLE_FIELDS),
        "normalization_fields": list(NORMALIZATION_BUNDLE_FIELDS),
        "normalization_aliases": _normalization_alias_metadata(fields),
        "constructor_mapping": _bundle_constructor_mapping(),
        "checksum_policy": {
            "fixed_true_measurement_arrays": "must_match_manifest_before_training",
            "per_condition_assumed_probe": "loaded_from_condition_npz_and_checksummed_before_training",
        },
        "cfg": _jsonable(asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else vars(cfg) if hasattr(cfg, "__dict__") else cfg),
    }
    return arrays, manifest


def write_canonical_condition_bundle(
    output_root: Path,
    sim: dict[str, Any],
    true_probe: np.ndarray,
    cfg: Any,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    arrays, manifest = build_canonical_condition_bundle_payload(sim, true_probe, cfg)
    npz_path = output_root / CANONICAL_BUNDLE_NPZ
    manifest_path = output_root / CANONICAL_BUNDLE_MANIFEST
    if "norm_Y_I" in arrays:
        raise ValueError("bare norm_Y_I is not allowed in canonical condition bundle")
    np.savez(npz_path, **arrays)
    write_json(manifest_path, manifest)
    return {"npz_path": npz_path, "manifest_path": manifest_path, "manifest": manifest}


def validate_canonical_condition_bundle(bundle_path: Path, manifest_path: Path) -> dict[str, Any]:
    bundle_path = Path(bundle_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    with np.load(bundle_path) as data:
        keys = set(data.files)
        if "norm_Y_I" in keys:
            raise ValueError("canonical condition bundle contains forbidden bare norm_Y_I field")
        fields = manifest.get("fields", {})
        if "norm_Y_I" in fields:
            raise ValueError("canonical condition manifest contains forbidden bare norm_Y_I field")
        for name in REQUIRED_BUNDLE_FIELDS:
            if name not in fields:
                raise ValueError(f"canonical condition manifest missing required field {name}")
            record = fields[name]
            if record.get("status") != "present" or name not in keys:
                raise ValueError(f"canonical condition bundle missing required field {name}")
        for name in NORMALIZATION_BUNDLE_FIELDS:
            if name not in fields or name not in keys:
                raise ValueError(f"canonical condition bundle missing normalization field {name}")
        for name, record in fields.items():
            if record.get("status") == "absent":
                if not record.get("absent_reason"):
                    raise ValueError(f"canonical condition manifest field {name} is absent without an absence reason")
                if name in keys:
                    raise ValueError(f"canonical condition bundle includes field {name} despite manifest absence")
                continue
            if name not in keys:
                raise ValueError(f"canonical condition bundle missing manifest-present field {name}")
            actual = array_sha256(data[name])
            if actual != record.get("checksum"):
                raise ValueError(f"checksum mismatch for canonical condition bundle field {name}")
        mapping = manifest.get("constructor_mapping")
        if not mapping or mapping.get("train", {}).get("probeGuess") != "condition assumed_probe":
            raise ValueError("canonical condition manifest missing constructor mapping")
    return manifest


def _bundle_array_or_none(data: Any, name: str) -> np.ndarray | None:
    return np.array(data[name], copy=True) if name in data.files else None


def _bundle_preflight_record(
    *,
    bundle_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    sim: dict[str, Any],
    train_container: Any,
    test_container: Any,
    assumed_probe: np.ndarray,
    assumed_probe_path: Path,
) -> dict[str, Any]:
    field_checksums = {
        name: record.get("checksum")
        for name, record in manifest["fields"].items()
        if record.get("status") == "present"
    }
    assumed_probe_checksums = build_assumed_probe_checksums(
        train_container=train_container,
        test_container=test_container,
        assumed_probe=assumed_probe,
        assumed_probe_path=assumed_probe_path,
    )
    expected_probe = assumed_probe_checksums["expected_assumed_probe"]
    mismatched_probe_keys = sorted(
        key for key, checksum in assumed_probe_checksums.items() if checksum is not None and checksum != expected_probe
    )
    if mismatched_probe_keys:
        raise AssertionError(f"assumed probe checksums mismatch: {mismatched_probe_keys}")
    canonical_data_checksums = build_canonical_data_checksums(sim)
    return {
        "condition_probe_policy": "assumed_probe_replaces_container_probe",
        "canonical_bundle_path": str(bundle_path),
        "canonical_bundle_manifest_path": str(manifest_path),
        "constructor_mapping": manifest["constructor_mapping"],
        "constructor_input_checksums": field_checksums,
        "true_measurement_checksums": {
            name: field_checksums[name]
            for name in ("X_train", "X_test", "Y_I_train", "Y_I_test", "Y_phi_train", "Y_phi_test")
            if name in field_checksums
        },
        "train_container_checksums": {
            "X_train": array_sha256(train_container._X_np),
            "Y_I_train": array_sha256(train_container._Y_I_np),
            "Y_phi_train": array_sha256(train_container._Y_phi_np),
            "coords_nominal_train": array_sha256(train_container._coords_nominal_np),
            "coords_true_train": array_sha256(train_container._coords_true_np),
        },
        "test_container_checksums": {
            "X_test": array_sha256(test_container._X_np),
            "Y_I_test": array_sha256(test_container._Y_I_np),
            "Y_phi_test": array_sha256(test_container._Y_phi_np),
            "coords_nominal_test": array_sha256(test_container._coords_nominal_np),
            "coords_true_test": array_sha256(test_container._coords_true_np),
        },
        "assumed_probe_checksums": assumed_probe_checksums,
        "normalization_field_checksums": {
            name: field_checksums[name] for name in NORMALIZATION_BUNDLE_FIELDS if name in field_checksums
        },
        "normalization_aliases": manifest.get("normalization_aliases", {}),
        "canonical_data_checksums": canonical_data_checksums,
        "condition_data_checksums": canonical_data_checksums,
    }


def load_condition_inputs_for_child(
    bundle_path: Path,
    manifest_path: Path,
    assumed_probe_path: Path,
) -> dict[str, Any]:
    from ptycho.loader import PtychoDataContainer

    bundle_path = Path(bundle_path)
    manifest_path = Path(manifest_path)
    assumed_probe_path = Path(assumed_probe_path)
    manifest = validate_canonical_condition_bundle(bundle_path, manifest_path)
    assumed_probe = np.asarray(_probe_from_npz(assumed_probe_path), dtype=np.complex64)
    with np.load(bundle_path) as data:
        train_container = PtychoDataContainer(
            X=np.array(data["X_train"], copy=True),
            Y_I=np.array(data["Y_I_train"], copy=True),
            Y_phi=np.array(data["Y_phi_train"], copy=True),
            norm_Y_I=np.array(data["norm_Y_I_train_container"], copy=True),
            YY_full=_bundle_array_or_none(data, "YY_full_train"),
            coords_nominal=np.array(data["coords_nominal_train"], copy=True),
            coords_true=np.array(data["coords_true_train"], copy=True),
            nn_indices=_bundle_array_or_none(data, "nn_indices_train"),
            global_offsets=_bundle_array_or_none(data, "global_offsets_train"),
            local_offsets=_bundle_array_or_none(data, "local_offsets_train"),
            probeGuess=assumed_probe,
        )
        test_container = PtychoDataContainer(
            X=np.array(data["X_test"], copy=True),
            Y_I=np.array(data["Y_I_test"], copy=True),
            Y_phi=np.array(data["Y_phi_test"], copy=True),
            norm_Y_I=np.array(data["norm_Y_I_test_container"], copy=True),
            YY_full=_bundle_array_or_none(data, "YY_full_test"),
            coords_nominal=np.array(data["coords_nominal_test"], copy=True),
            coords_true=np.array(data["coords_true_test"], copy=True),
            nn_indices=_bundle_array_or_none(data, "nn_indices_test"),
            global_offsets=_bundle_array_or_none(data, "global_offsets_test"),
            local_offsets=_bundle_array_or_none(data, "local_offsets_test"),
            probeGuess=assumed_probe,
        )
        sim = {
            "train": {
                "X": np.array(data["X_train"], copy=True),
                "Y_I": np.array(data["Y_I_train"], copy=True),
                "Y_phi": np.array(data["Y_phi_train"], copy=True),
                "coords_nominal": np.array(data["coords_nominal_train"], copy=True),
                "coords_true": np.array(data["coords_true_train"], copy=True),
                "coords_offsets": _bundle_array_or_none(data, "coords_offsets_train"),
                "YY_full": _bundle_array_or_none(data, "YY_full_train"),
                "container": train_container,
            },
            "test": {
                "X": np.array(data["X_test"], copy=True),
                "Y_I": np.array(data["Y_I_test"], copy=True),
                "Y_phi": np.array(data["Y_phi_test"], copy=True),
                "coords_nominal": np.array(data["coords_nominal_test"], copy=True),
                "coords_true": np.array(data["coords_true_test"], copy=True),
                "coords_offsets": _bundle_array_or_none(data, "coords_offsets_test"),
                "YY_full": _bundle_array_or_none(data, "YY_full_test"),
                "YY_ground_truth": np.array(data["YY_ground_truth_test"], copy=True),
                "norm_Y_I": np.array(data["norm_Y_I_test_stitch"], copy=True),
                "container": test_container,
            },
            "intensity_scale": np.array(data["intensity_scale_model"], copy=True),
        }
    preflight = _bundle_preflight_record(
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        manifest=manifest,
        sim=sim,
        train_container=train_container,
        test_container=test_container,
        assumed_probe=assumed_probe,
        assumed_probe_path=assumed_probe_path,
    )
    return {
        "train_container": train_container,
        "test_container": test_container,
        "sim": sim,
        "assumed_probe": assumed_probe,
        "preflight": preflight,
        "manifest": manifest,
    }


def _checksum_array_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    return array_sha256(np.asarray(value))


def _checksum_probe_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    return array_sha256(np.asarray(value, dtype=np.complex64))


def _split_identifier_payload(
    *,
    train_X: Any,
    test_X: Any,
    train_coords_nominal: Any,
    test_coords_nominal: Any,
) -> dict[str, Any]:
    train_X_arr = np.asarray(train_X)
    test_X_arr = np.asarray(test_X)
    train_coords_arr = np.asarray(train_coords_nominal)
    test_coords_arr = np.asarray(test_coords_nominal)
    return {
        "train_count": int(train_X_arr.shape[0]),
        "test_count": int(test_X_arr.shape[0]),
        "train_X_shape": list(train_X_arr.shape),
        "test_X_shape": list(test_X_arr.shape),
        "train_coords_nominal_shape": list(train_coords_arr.shape),
        "test_coords_nominal_shape": list(test_coords_arr.shape),
        "train_channel_count": int(train_X_arr.shape[-1]) if train_X_arr.ndim else None,
        "test_channel_count": int(test_X_arr.shape[-1]) if test_X_arr.ndim else None,
    }


def _data_checksum_payload(
    *,
    train_X: Any,
    test_X: Any,
    train_coords_nominal: Any,
    test_coords_nominal: Any,
    train_coords_true: Any,
    test_coords_true: Any,
    YY_ground_truth: Any,
    norm_Y_I: Any,
    train_coords_offsets: Any = None,
    test_coords_offsets: Any = None,
) -> dict[str, Any]:
    split_identifiers = _split_identifier_payload(
        train_X=train_X,
        test_X=test_X,
        train_coords_nominal=train_coords_nominal,
        test_coords_nominal=test_coords_nominal,
    )
    return {
        "train_X": _checksum_array_or_none(train_X),
        "test_X": _checksum_array_or_none(test_X),
        "train_coords_nominal": _checksum_array_or_none(train_coords_nominal),
        "test_coords_nominal": _checksum_array_or_none(test_coords_nominal),
        "train_coords_true": _checksum_array_or_none(train_coords_true),
        "test_coords_true": _checksum_array_or_none(test_coords_true),
        "train_coords_offsets": _checksum_array_or_none(train_coords_offsets),
        "test_coords_offsets": _checksum_array_or_none(test_coords_offsets),
        "YY_ground_truth": _checksum_array_or_none(YY_ground_truth),
        "norm_Y_I": _checksum_array_or_none(norm_Y_I),
        "split_identifiers": split_identifiers,
        "split_identifiers_sha256": payload_sha256(split_identifiers),
    }


def build_canonical_data_checksums(sim: dict[str, Any]) -> dict[str, Any]:
    train_container = sim["train"].get("container")
    test_container = sim["test"].get("container")
    return _data_checksum_payload(
        train_X=getattr(train_container, "_X_np", sim["train"]["X"]),
        test_X=getattr(test_container, "_X_np", sim["test"]["X"]),
        train_coords_nominal=getattr(train_container, "_coords_nominal_np", sim["train"]["coords_nominal"]),
        test_coords_nominal=getattr(test_container, "_coords_nominal_np", sim["test"]["coords_nominal"]),
        train_coords_true=getattr(train_container, "_coords_true_np", sim["train"]["coords_true"]),
        test_coords_true=getattr(test_container, "_coords_true_np", sim["test"]["coords_true"]),
        train_coords_offsets=sim["train"].get("coords_offsets"),
        test_coords_offsets=sim["test"].get("coords_offsets"),
        YY_ground_truth=sim["test"]["YY_ground_truth"],
        norm_Y_I=sim["test"]["norm_Y_I"],
    )


def build_condition_data_checksums(
    *,
    sim: dict[str, Any],
    train_container: Any,
    test_container: Any,
) -> dict[str, Any]:
    return _data_checksum_payload(
        train_X=train_container._X_np,
        test_X=test_container._X_np,
        train_coords_nominal=train_container._coords_nominal_np,
        test_coords_nominal=test_container._coords_nominal_np,
        train_coords_true=train_container._coords_true_np,
        test_coords_true=test_container._coords_true_np,
        train_coords_offsets=sim["train"].get("coords_offsets"),
        test_coords_offsets=sim["test"].get("coords_offsets"),
        YY_ground_truth=sim["test"]["YY_ground_truth"],
        norm_Y_I=sim["test"]["norm_Y_I"],
    )


def _probe_from_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "probe" in data:
            return np.asarray(data["probe"], dtype=np.complex64)
        if data.files:
            return np.asarray(data[data.files[0]], dtype=np.complex64)
    raise ValueError(f"{path} does not contain a probe array")


def build_assumed_probe_checksums(
    *,
    train_container: Any,
    test_container: Any,
    assumed_probe: np.ndarray,
    assumed_probe_path: Path | None = None,
) -> dict[str, str | None]:
    return {
        "expected_assumed_probe": _checksum_probe_or_none(assumed_probe),
        "train_data_probe": _checksum_probe_or_none(train_container._probe_np),
        "test_data_probe": _checksum_probe_or_none(test_container._probe_np),
        "persisted_condition_probe": (
            _checksum_probe_or_none(_probe_from_npz(Path(assumed_probe_path)))
            if assumed_probe_path is not None
            else None
        ),
    }


def assert_condition_preflight(
    *,
    sim: dict[str, Any],
    train_container: Any,
    test_container: Any,
    assumed_probe: np.ndarray,
    assumed_probe_path: Path | None,
    canonical_data_checksums: dict[str, Any],
) -> dict[str, Any]:
    condition_data_checksums = build_condition_data_checksums(
        sim=sim,
        train_container=train_container,
        test_container=test_container,
    )
    if condition_data_checksums != canonical_data_checksums:
        mismatched = sorted(
            key
            for key in set(condition_data_checksums) | set(canonical_data_checksums)
            if condition_data_checksums.get(key) != canonical_data_checksums.get(key)
        )
        raise AssertionError(f"canonical data checksums mismatch: {mismatched}")

    assumed_probe_checksums = build_assumed_probe_checksums(
        train_container=train_container,
        test_container=test_container,
        assumed_probe=assumed_probe,
        assumed_probe_path=assumed_probe_path,
    )
    expected_probe = assumed_probe_checksums["expected_assumed_probe"]
    mismatched_probe_keys = sorted(
        key
        for key, checksum in assumed_probe_checksums.items()
        if checksum is not None and checksum != expected_probe
    )
    if mismatched_probe_keys:
        raise AssertionError(f"assumed probe checksums mismatch: {mismatched_probe_keys}")

    return {
        "condition_probe_policy": "assumed_probe_replaces_container_probe",
        "canonical_data_checksums": canonical_data_checksums,
        "condition_data_checksums": condition_data_checksums,
        "assumed_probe_checksums": assumed_probe_checksums,
    }


def reset_tf_state(clear_model_cache: bool = True) -> None:
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass
    if clear_model_cache:
        try:
            from ptycho import model as ptycho_model

            if hasattr(ptycho_model, "_lazy_cache"):
                ptycho_model._lazy_cache.clear()
            if hasattr(ptycho_model, "_model_construction_done"):
                ptycho_model._model_construction_done = False
            model_namespace = vars(ptycho_model)
            for name in ("autoencoder", "diffraction_to_obj", "autoencoder_no_nll"):
                if name in model_namespace:
                    delattr(ptycho_model, name)
        except Exception:
            pass
    gc.collect()


def persist_true_measurements(output_root: Path, sim: dict[str, Any], true_probe: np.ndarray) -> dict[str, str]:
    probes_dir = output_root / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    true_probe_path = probes_dir / "true_probe.npz"
    np.savez(true_probe_path, probe=true_probe)
    measurement_path = output_root / "true_measurement_data.npz"
    train_container = sim["train"].get("container")
    test_container = sim["test"].get("container")
    np.savez(
        measurement_path,
        X_train=getattr(train_container, "_X_np", sim["train"]["X"]),
        X_test=getattr(test_container, "_X_np", sim["test"]["X"]),
        train_coords_nominal=getattr(train_container, "_coords_nominal_np", sim["train"]["coords_nominal"]),
        test_coords_nominal=getattr(test_container, "_coords_nominal_np", sim["test"]["coords_nominal"]),
        train_coords_true=getattr(train_container, "_coords_true_np", sim["train"]["coords_true"]),
        test_coords_true=getattr(test_container, "_coords_true_np", sim["test"]["coords_true"]),
        YY_ground_truth=sim["test"]["YY_ground_truth"],
        norm_Y_I=np.array(sim["test"]["norm_Y_I"]),
    )
    if sim["train"].get("coords_offsets") is not None or sim["test"].get("coords_offsets") is not None:
        with np.load(measurement_path) as existing:
            payload = {key: existing[key] for key in existing.files}
        if sim["train"].get("coords_offsets") is not None:
            payload["train_coords_offsets"] = sim["train"]["coords_offsets"]
        if sim["test"].get("coords_offsets") is not None:
            payload["test_coords_offsets"] = sim["test"]["coords_offsets"]
        np.savez(measurement_path, **payload)
    return {"true_probe": str(true_probe_path), "true_measurement_data": str(measurement_path)}


def _prediction_delta(first: np.ndarray | None, second: np.ndarray | None) -> dict[str, Any]:
    if first is None or second is None:
        return {"max_abs_delta": None, "mean_abs_delta": None, "available": False}
    delta = np.abs(first - second)
    return {
        "max_abs_delta": float(np.max(delta)),
        "mean_abs_delta": float(np.mean(delta)),
        "available": True,
    }


def _assign_model_probe_variable(model: Any, probe: np.ndarray) -> tuple[bool, np.ndarray | None, str | None]:
    try:
        import tensorflow as tf
    except Exception as exc:
        return False, None, f"tensorflow import failed: {exc}"

    for layer in getattr(model, "layers", []):
        if layer.__class__.__name__ != "ProbeIllumination" or not hasattr(layer, "w"):
            continue
        old_value = layer.w.numpy()
        new_value = np.asarray(probe, dtype=np.complex64)
        while new_value.ndim < old_value.ndim:
            new_value = new_value[None, ...]
        if new_value.shape != old_value.shape:
            if old_value.shape == (1, probe.shape[0], probe.shape[1], 1):
                new_value = np.asarray(probe, dtype=np.complex64)[None, ..., None]
        if new_value.shape != old_value.shape:
            return False, old_value, f"probe variable shape {old_value.shape} incompatible with {probe.shape}"
        layer.w.assign(tf.cast(new_value, tf.complex64))
        return True, old_value, layer.name
    return False, None, "ProbeIllumination layer not found"


def _restore_model_probe_variable(model: Any, old_value: np.ndarray | None) -> None:
    if old_value is None:
        return
    for layer in getattr(model, "layers", []):
        if layer.__class__.__name__ == "ProbeIllumination" and hasattr(layer, "w"):
            layer.w.assign(old_value)
            return


def _stitch_prediction(prediction: np.ndarray, sim: dict[str, Any]) -> np.ndarray:
    from ptycho.workflows.grid_lines_workflow import stitch_predictions

    norm_Y_I = sim["test"]["norm_Y_I"]
    amp = stitch_predictions(prediction, norm_Y_I, part="amp")
    phase = stitch_predictions(prediction, norm_Y_I, part="phase")
    return amp * np.exp(1j * phase)


def run_probe_consumption_smoke(
    model: Any,
    sim: dict[str, Any],
    true_probe: np.ndarray,
    large_perturbed_probe: np.ndarray,
    cfg: Any,
) -> dict[str, Any]:
    from ptycho.workflows.grid_lines_workflow import run_pinn_inference

    true_test = clone_container_with_probe(sim["test"]["container"], true_probe)
    perturbed_test = clone_container_with_probe(sim["test"]["container"], large_perturbed_probe)

    true_pred = run_pinn_inference(model, true_test._X_np, true_test._coords_nominal_np)
    container_pred = run_pinn_inference(model, perturbed_test._X_np, perturbed_test._coords_nominal_np)
    container_result = _prediction_delta(true_pred, container_pred)
    container_result["metric_delta"] = None

    direct_result: dict[str, Any] = {"attempted": True}
    assigned, old_value, assign_note = _assign_model_probe_variable(model, large_perturbed_probe)
    direct_result["assign_note"] = assign_note
    direct_result["assigned"] = bool(assigned)
    if assigned:
        try:
            direct_pred = run_pinn_inference(model, true_test._X_np, true_test._coords_nominal_np)
            direct_result.update(_prediction_delta(true_pred, direct_pred))
        finally:
            _restore_model_probe_variable(model, old_value)
    else:
        direct_result.update({"max_abs_delta": None, "mean_abs_delta": None, "available": False})

    thresholds = {
        "max_abs_delta_noop": SMOKE_NOOP_MAX_ABS_DELTA,
        "mean_abs_delta_noop": SMOKE_NOOP_MEAN_ABS_DELTA,
    }
    max_deltas = [
        result.get("max_abs_delta")
        for result in (container_result, direct_result)
        if result.get("max_abs_delta") is not None
    ]
    if any(delta > SMOKE_NOOP_MAX_ABS_DELTA for delta in max_deltas):
        decision = "frozen_model_assumed_probe"
    else:
        decision = "fixed_wrong_probe_training"
    return {
        "created_utc": _utc_now(),
        "container_replacement": container_result,
        "direct_probe_variable": direct_result,
        "decision": decision,
        "thresholds": thresholds,
    }


def _history_payload(history: Any) -> dict[str, Any]:
    if history is None:
        return {}
    return {
        "epoch": [int(epoch) for epoch in getattr(history, "epoch", [])],
        "history": getattr(history, "history", {}),
    }


def _metric_value(metrics: dict[str, Any], key: str, index: int) -> float | None:
    try:
        value = metrics[key][index]
    except Exception:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def flatten_metrics(condition: PerturbationCondition, metrics: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "condition_id": condition.condition_id,
        "perturbation_type": condition.perturbation_type,
        "value": condition.value,
        "seed": condition.seed,
        "amp_ssim": _metric_value(metrics, "ssim", 0),
        "amp_psnr": _metric_value(metrics, "psnr", 0),
        "amp_mse": _metric_value(metrics, "mse", 0),
        "amp_mae": _metric_value(metrics, "mae", 0),
        "phase_ssim": _metric_value(metrics, "ssim", 1),
        "phase_psnr": _metric_value(metrics, "psnr", 1),
        "phase_mse": _metric_value(metrics, "mse", 1),
        "phase_mae": _metric_value(metrics, "mae", 1),
        "reviewer_facing": condition.reviewer_facing,
        "status": status,
    }


def evaluate_baseline_comparability(
    condition_results: dict[str, dict[str, Any]],
    *,
    adopt_rerun_baseline: bool,
) -> dict[str, Any]:
    baseline = condition_results.get("baseline")
    payload: dict[str, Any] = {
        "table2_amp_ssim": TABLE2_AMP_SSIM,
        "table2_amp_psnr": TABLE2_AMP_PSNR,
        "amp_ssim_tolerance": BASELINE_AMP_SSIM_TOL,
        "amp_psnr_tolerance_db": BASELINE_AMP_PSNR_TOL_DB,
        "adopt_rerun_baseline_requested": bool(adopt_rerun_baseline),
    }
    if not baseline or baseline.get("status") != "ok":
        return {
            **payload,
            "status": "blocked_missing_successful_baseline",
            "baseline_policy": "no_numeric_stress_table",
            "claim_safe": False,
            "reason": "baseline condition missing or failed",
        }

    amp_ssim = baseline.get("amp_ssim")
    amp_psnr = baseline.get("amp_psnr")
    if amp_ssim is None or amp_psnr is None:
        return {
            **payload,
            "status": "blocked_missing_baseline_metrics",
            "baseline_policy": "no_numeric_stress_table",
            "claim_safe": False,
            "reason": "baseline condition did not record amp_ssim and amp_psnr",
        }

    amp_ssim = float(amp_ssim)
    amp_psnr = float(amp_psnr)
    ssim_delta = abs(amp_ssim - TABLE2_AMP_SSIM)
    psnr_delta = abs(amp_psnr - TABLE2_AMP_PSNR)
    ssim_within_tolerance = ssim_delta <= BASELINE_AMP_SSIM_TOL
    psnr_within_tolerance = psnr_delta <= BASELINE_AMP_PSNR_TOL_DB
    comparable = ssim_within_tolerance or psnr_within_tolerance
    payload.update(
        {
            "rerun_amp_ssim": amp_ssim,
            "rerun_amp_psnr": amp_psnr,
            "amp_ssim_abs_delta": ssim_delta,
            "amp_psnr_abs_delta": psnr_delta,
            "amp_ssim_within_tolerance": ssim_within_tolerance,
            "amp_psnr_within_tolerance": psnr_within_tolerance,
            "table2_comparable": comparable,
        }
    )
    if comparable:
        return {
            **payload,
            "status": "comparable_to_table2",
            "baseline_policy": "rerun_baseline_table2_comparable",
            "claim_safe": True,
        }
    if adopt_rerun_baseline:
        return {
            **payload,
            "status": "adopt_rerun_baseline",
            "baseline_policy": "adopt_rerun_baseline_no_old_numeric_comparison",
            "claim_safe": True,
        }
    return {
        **payload,
        "status": "pivot_no_numeric_stress_table",
        "baseline_policy": "rerun_baseline_not_comparable",
        "claim_safe": False,
        "reason": "rerun baseline is outside Table 2 comparability tolerances",
    }


def evaluate_infrastructure_failure_gate(
    condition_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    failures_by_reason: dict[str, list[str]] = {}
    for condition_id, result in condition_results.items():
        if condition_id == "baseline" or result.get("status") == "ok":
            continue
        reason = str(result.get("error") or result.get("status") or "unknown failure").splitlines()[0]
        failures_by_reason.setdefault(reason, []).append(condition_id)

    repeated_reason = None
    repeated_conditions: list[str] = []
    for reason, condition_ids in failures_by_reason.items():
        if len(condition_ids) > len(repeated_conditions):
            repeated_reason = reason
            repeated_conditions = condition_ids

    failed_condition_count = sum(len(condition_ids) for condition_ids in failures_by_reason.values())
    if repeated_reason is not None and len(repeated_conditions) > 2:
        return {
            "status": "stop_full_grid",
            "claim_safe": False,
            "failed_condition_count": failed_condition_count,
            "repeated_failure_reason": repeated_reason,
            "repeated_failure_conditions": repeated_conditions,
        }
    return {
        "status": "pass",
        "claim_safe": True,
        "failed_condition_count": failed_condition_count,
        "failures_by_reason": failures_by_reason,
    }


def evaluate_mild_perturbation_gate(
    condition_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "amp_ssim_drop_limit": MILD_AMP_SSIM_DROP_LIMIT,
        "amp_psnr_drop_limit_db": MILD_AMP_PSNR_DROP_LIMIT_DB,
        "required_mild_conditions": list(MILD_CONDITION_IDS),
    }
    baseline = condition_results.get("baseline")
    if not baseline or baseline.get("status") != "ok":
        return {
            **payload,
            "status": "blocked_missing_successful_baseline",
            "export_allowed": False,
            "claim_safe": False,
            "robustness_claim_safe": False,
            "reason": "baseline condition missing or failed",
        }
    baseline_ssim = baseline.get("amp_ssim")
    baseline_psnr = baseline.get("amp_psnr")
    if baseline_ssim is None or baseline_psnr is None:
        return {
            **payload,
            "status": "blocked_missing_baseline_metrics",
            "export_allowed": False,
            "claim_safe": False,
            "robustness_claim_safe": False,
            "reason": "baseline condition did not record amp_ssim and amp_psnr",
        }

    missing = [
        condition_id
        for condition_id in MILD_CONDITION_IDS
        if condition_id not in condition_results or condition_results[condition_id].get("status") != "ok"
    ]
    if missing:
        return {
            **payload,
            "status": "blocked_missing_mild_conditions",
            "export_allowed": False,
            "claim_safe": False,
            "robustness_claim_safe": False,
            "missing_conditions": missing,
        }

    baseline_ssim = float(baseline_ssim)
    baseline_psnr = float(baseline_psnr)
    mild_conditions: dict[str, dict[str, Any]] = {}
    sensitivity_required = False
    for condition_id in MILD_CONDITION_IDS:
        result = condition_results[condition_id]
        amp_ssim = result.get("amp_ssim")
        amp_psnr = result.get("amp_psnr")
        if amp_ssim is None or amp_psnr is None:
            return {
                **payload,
                "status": "blocked_missing_mild_metrics",
                "export_allowed": False,
                "claim_safe": False,
                "robustness_claim_safe": False,
                "condition_id": condition_id,
            }
        ssim_drop = baseline_ssim - float(amp_ssim)
        psnr_drop = baseline_psnr - float(amp_psnr)
        requires_sensitivity_language = (
            ssim_drop > MILD_AMP_SSIM_DROP_LIMIT
            or psnr_drop > MILD_AMP_PSNR_DROP_LIMIT_DB
        )
        sensitivity_required = sensitivity_required or requires_sensitivity_language
        mild_conditions[condition_id] = {
            "amp_ssim": float(amp_ssim),
            "amp_psnr": float(amp_psnr),
            "amp_ssim_drop": ssim_drop,
            "amp_psnr_drop_db": psnr_drop,
            "requires_sensitivity_language": requires_sensitivity_language,
        }

    if sensitivity_required:
        return {
            **payload,
            "status": "sensitivity_language_required",
            "export_allowed": True,
            "claim_safe": True,
            "robustness_claim_safe": False,
            "mild_conditions": mild_conditions,
            "claim_boundary": "do_not_claim_probe_error_robustness_without_sensitivity_language",
        }
    return {
        **payload,
        "status": "pass",
        "export_allowed": True,
        "claim_safe": True,
        "robustness_claim_safe": True,
        "mild_conditions": mild_conditions,
    }


def failed_condition_payload(
    condition: PerturbationCondition,
    exc: BaseException,
    *,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "failed",
        "condition_id": condition.condition_id,
        "perturbation_type": condition.perturbation_type,
        "value": condition.value,
        "seed": condition.seed,
        "reviewer_facing": condition.reviewer_facing,
        "error": repr(exc),
    }
    if preflight is not None:
        payload["preflight"] = preflight
        payload["canonical_data_checksums"] = preflight.get("canonical_data_checksums")
        payload["condition_data_checksums"] = preflight.get("condition_data_checksums")
        payload["assumed_probe_checksums"] = preflight.get("assumed_probe_checksums")
        payload["condition_probe_policy"] = preflight.get("condition_probe_policy")
    return payload


def run_condition(
    *,
    condition: PerturbationCondition,
    sim: dict[str, Any],
    true_probe: np.ndarray,
    assumed_probe: np.ndarray,
    branch_decision: str,
    cfg: Any,
    output_root: Path,
    baseline_model: Any | None = None,
    canonical_data_checksums: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from ptycho.evaluation import eval_reconstruction
    from ptycho.workflows.grid_lines_workflow import (
        configure_legacy_params,
        run_pinn_inference,
        train_pinn_model,
    )

    condition_dir = output_root / "conditions" / condition.condition_id
    recon_dir = condition_dir / "recon"
    condition_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    assumed_probe_path = condition_dir / "assumed_probe.npz"
    np.savez(assumed_probe_path, probe=assumed_probe)

    model = baseline_model
    history = None
    preflight: dict[str, Any] | None = None
    canonical_data_checksums = canonical_data_checksums or build_canonical_data_checksums(sim)
    try:
        if branch_decision == "fixed_wrong_probe_training":
            reset_tf_state(clear_model_cache=True)
            train_container = clone_container_with_probe(sim["train"]["container"], assumed_probe)
            test_container = clone_container_with_probe(sim["test"]["container"], assumed_probe)
            preflight = assert_condition_preflight(
                sim=sim,
                train_container=train_container,
                test_container=test_container,
                assumed_probe=assumed_probe,
                assumed_probe_path=assumed_probe_path,
                canonical_data_checksums=canonical_data_checksums,
            )
            configure_legacy_params(cfg, assumed_probe)
            model, history = train_pinn_model(train_container)
            prediction = run_pinn_inference(model, test_container._X_np, test_container._coords_nominal_np)
        elif branch_decision == "frozen_model_assumed_probe":
            if model is None:
                raise ValueError("frozen_model_assumed_probe requires a baseline model")
            reset_tf_state(clear_model_cache=False)
            train_container = clone_container_with_probe(sim["train"]["container"], assumed_probe)
            test_container = clone_container_with_probe(sim["test"]["container"], assumed_probe)
            preflight = assert_condition_preflight(
                sim=sim,
                train_container=train_container,
                test_container=test_container,
                assumed_probe=assumed_probe,
                assumed_probe_path=assumed_probe_path,
                canonical_data_checksums=canonical_data_checksums,
            )
            assigned, old_value, note = _assign_model_probe_variable(model, assumed_probe)
            if not assigned:
                raise RuntimeError(f"could not assign probe variable for frozen-model branch: {note}")
            try:
                prediction = run_pinn_inference(model, test_container._X_np, test_container._coords_nominal_np)
            finally:
                _restore_model_probe_variable(model, old_value)
        else:
            raise ValueError(f"unsupported branch decision: {branch_decision}")
    except Exception as exc:
        if preflight is None:
            raise
        error_payload = failed_condition_payload(condition, exc, preflight=preflight)
        write_json(condition_dir / "metrics.json", error_payload)
        if branch_decision == "fixed_wrong_probe_training":
            reset_tf_state(clear_model_cache=True)
        return error_payload

    if prediction is None:
        error_payload = failed_condition_payload(
            condition,
            RuntimeError("PINN inference returned None"),
            preflight=preflight,
        )
        write_json(condition_dir / "metrics.json", error_payload)
        return error_payload

    stitched = _stitch_prediction(prediction, sim)
    recon = np.squeeze(stitched)
    if recon.ndim > 2:
        recon = recon[0]
    np.savez(recon_dir / "recon.npz", YY_pred=recon.astype(np.complex64), amp=np.abs(recon), phase=np.angle(recon))
    if history is not None:
        write_json(condition_dir / "train_history.json", _history_payload(history))

    metrics = eval_reconstruction(stitched, sim["test"]["YY_ground_truth"], label=condition.condition_id)
    flat = flatten_metrics(condition, metrics, status="ok")
    condition_payload = {
        "status": "ok",
        "condition": asdict(condition),
        "preflight": preflight,
        "metrics": metrics,
        **flat,
    }
    write_json(condition_dir / "metrics.json", condition_payload)
    return condition_payload


def write_metrics_outputs(output_root: Path, condition_results: dict[str, dict[str, Any]]) -> tuple[Path, Path]:
    metrics_json = output_root / "metrics.json"
    write_json(metrics_json, {"created_utc": _utc_now(), "conditions": condition_results})
    metrics_csv = output_root / "metrics.csv"
    fieldnames = [
        "condition_id",
        "perturbation_type",
        "value",
        "seed",
        "amp_ssim",
        "amp_psnr",
        "amp_mse",
        "amp_mae",
        "phase_ssim",
        "phase_psnr",
        "phase_mse",
        "phase_mae",
        "reviewer_facing",
        "status",
    ]
    with metrics_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in condition_results.values():
            writer.writerow({key: result.get(key) for key in fieldnames})
    return metrics_json, metrics_csv


def write_stress_figure(output_root: Path, condition_results: dict[str, dict[str, Any]]) -> Path | None:
    import matplotlib.pyplot as plt

    ok_results = [result for result in condition_results.values() if result.get("status") == "ok"]
    if not ok_results:
        return None
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "probe_mischaracterization_stress.png"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    for metric_key, axis, ylabel in (
        ("amp_ssim", axes[0, 0], "Amplitude SSIM"),
        ("amp_psnr", axes[0, 1], "Amplitude PSNR (dB)"),
    ):
        baseline = next((item for item in ok_results if item["condition_id"] == "baseline"), None)
        if baseline and baseline.get(metric_key) is not None:
            axis.axhline(float(baseline[metric_key]), color="0.4", linestyle="--", label="baseline")
        for perturbation_type in sorted({item["perturbation_type"] for item in ok_results if item["perturbation_type"] != "baseline"}):
            group = [
                item
                for item in ok_results
                if item["perturbation_type"] == perturbation_type and item.get(metric_key) is not None
            ]
            if not group:
                continue
            group.sort(key=lambda item: float(item["value"]))
            axis.plot(
                [float(item["value"]) for item in group],
                [float(item[metric_key]) for item in group],
                marker="o",
                label=perturbation_type,
            )
        axis.set_xlabel("Perturbation value")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return fig_path


def train_baseline_for_smoke(sim: dict[str, Any], true_probe: np.ndarray, cfg: Any) -> tuple[Any, Any]:
    from ptycho.workflows.grid_lines_workflow import configure_legacy_params, train_pinn_model

    reset_tf_state(clear_model_cache=True)
    configure_legacy_params(cfg, true_probe)
    train_container = clone_container_with_probe(sim["train"]["container"], true_probe)
    return train_pinn_model(train_container)


def _load_child_request(request_path: Path) -> dict[str, Any]:
    return json.loads(Path(request_path).read_text())


def _grid_config_from_payload(payload: dict[str, Any]):
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig

    config_payload = dict(payload)
    for key in ("output_dir", "probe_npz"):
        if key in config_payload and config_payload[key] is not None:
            config_payload[key] = Path(config_payload[key])
    return GridLinesConfig(**config_payload)


def _condition_from_payload(payload: dict[str, Any]) -> PerturbationCondition:
    return PerturbationCondition(
        condition_id=payload["condition_id"],
        perturbation_type=payload["perturbation_type"],
        value=payload.get("value"),
        seed=payload.get("seed"),
        reviewer_facing=payload.get("reviewer_facing", True),
        renormalize_energy=payload.get("renormalize_energy", True),
    )


def write_child_invocation_artifact(
    request_path: Path,
    request: dict[str, Any],
    mode_flag: str,
) -> Path:
    invocation_path = Path(request["child_invocation_path"])
    command = [
        "python",
        SCRIPT_PATH,
        mode_flag,
        "--child-request-json",
        str(request_path),
    ]
    payload = {
        "script": SCRIPT_PATH,
        "argv": command[2:],
        "command": " ".join(command),
        "parsed_args": {
            mode_flag.lstrip("-").replace("-", "_"): True,
            "child_request_json": str(request_path),
        },
        "cwd": str(Path.cwd()),
        "timestamp_utc": _utc_now(),
        "pid": os.getpid(),
        "request_path": str(request_path),
        "request_sha256": file_sha256(Path(request_path)),
        "request": request,
        "runtime_provenance": capture_study_runtime_provenance(),
    }
    write_json(invocation_path, payload)
    return invocation_path


def build_smoke_child_request(
    *,
    output_root: Path,
    bundle_paths: dict[str, Any],
    true_probe_path: Path,
    assumed_probe: np.ndarray,
    smoke_condition: PerturbationCondition,
    perturbation_metadata: dict[str, Any],
    cfg: Any,
) -> dict[str, Any]:
    smoke_dir = Path(output_root) / "conditions" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    assumed_probe_path = smoke_dir / "assumed_probe.npz"
    np.savez(assumed_probe_path, probe=assumed_probe)
    request_path = smoke_dir / "child_request.json"
    request = {
        "request_type": "smoke",
        "condition_id": "smoke",
        "run_root": str(output_root),
        "bundle_path": str(bundle_paths["npz_path"]),
        "bundle_manifest_path": str(bundle_paths["manifest_path"]),
        "true_probe_path": str(true_probe_path),
        "true_probe_sha256": array_sha256(_probe_from_npz(true_probe_path)),
        "assumed_probe_path": str(assumed_probe_path),
        "assumed_probe_sha256": array_sha256(assumed_probe),
        "probe_consumption_smoke_path": str(Path(output_root) / "probe_consumption_smoke.json"),
        "child_invocation_path": str(smoke_dir / "child_invocation.json"),
        "stdout_log_path": str(smoke_dir / "child_stdout.log"),
        "stderr_log_path": str(smoke_dir / "child_stderr.log"),
        "grid_config": _config_payload(cfg),
        "condition": asdict(smoke_condition),
        "perturbation_metadata": perturbation_metadata,
        "canonical_checksum_policy": "fixed_true_measurement_arrays_must_match_manifest",
    }
    write_json(request_path, request)
    return {"request_path": request_path, "request": request}


def build_condition_child_request(
    *,
    output_root: Path,
    bundle_paths: dict[str, Any],
    condition: PerturbationCondition,
    assumed_probe: np.ndarray,
    perturbation_metadata: dict[str, Any],
    branch_decision: str,
    cfg: Any,
) -> dict[str, Any]:
    condition_dir = Path(output_root) / "conditions" / condition.condition_id
    condition_dir.mkdir(parents=True, exist_ok=True)
    assumed_probe_path = condition_dir / "assumed_probe.npz"
    np.savez(assumed_probe_path, probe=assumed_probe)
    request_path = condition_dir / "child_request.json"
    request = {
        "request_type": "condition",
        "condition_id": condition.condition_id,
        "run_root": str(output_root),
        "bundle_path": str(bundle_paths["npz_path"]),
        "bundle_manifest_path": str(bundle_paths["manifest_path"]),
        "assumed_probe_path": str(assumed_probe_path),
        "assumed_probe_sha256": array_sha256(assumed_probe),
        "metrics_path": str(condition_dir / "metrics.json"),
        "child_invocation_path": str(condition_dir / "child_invocation.json"),
        "stdout_log_path": str(condition_dir / "child_stdout.log"),
        "stderr_log_path": str(condition_dir / "child_stderr.log"),
        "branch_decision": branch_decision,
        "grid_config": _config_payload(cfg),
        "condition": asdict(condition),
        "perturbation_metadata": perturbation_metadata,
        "canonical_checksum_policy": "fixed_true_measurement_arrays_must_match_manifest",
    }
    write_json(request_path, request)
    return {"request_path": request_path, "request": request}


def launch_child_process(mode_flag: str, request_path: Path) -> dict[str, Any]:
    request_path = Path(request_path)
    request = _load_child_request(request_path)
    stdout_path = Path(request["stdout_log_path"])
    stderr_path = Path(request["stderr_log_path"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        SCRIPT_PATH,
        mode_flag,
        "--child-request-json",
        str(request_path),
    ]
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(REPO_ROOT)
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    start_utc = _utc_now()
    proc = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
    )
    stdout_text, stderr_text = proc.communicate()
    return_code = proc.wait()
    stdout_path.write_text(stdout_text or "")
    stderr_path.write_text(stderr_text or "")
    end_utc = _utc_now()
    record = {
        "command": command,
        "cwd": str(REPO_ROOT),
        "pid": proc.pid,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "return_code": int(return_code),
        "request_path": str(request_path),
        "request_sha256": file_sha256(request_path),
        "stdout_log_path": str(stdout_path),
        "stderr_log_path": str(stderr_path),
        "child_invocation_path": request.get("child_invocation_path"),
    }
    child_invocation_path = request.get("child_invocation_path")
    return record


def run_smoke_child_from_request(request_path: Path) -> int:
    from ptycho.workflows.grid_lines_workflow import configure_legacy_params, train_pinn_model

    request_path = Path(request_path)
    request = _load_child_request(request_path)
    write_child_invocation_artifact(request_path, request, "--child-smoke-runner")
    cfg = _grid_config_from_payload(request["grid_config"])
    true_probe_path = Path(request["true_probe_path"])
    true_probe = np.asarray(_probe_from_npz(true_probe_path), dtype=np.complex64)
    loaded = load_condition_inputs_for_child(
        request["bundle_path"],
        request["bundle_manifest_path"],
        true_probe_path,
    )
    large_perturbed_probe = np.asarray(_probe_from_npz(Path(request["assumed_probe_path"])), dtype=np.complex64)
    reset_tf_state(clear_model_cache=True)
    configure_legacy_params(cfg, true_probe)
    model, history = train_pinn_model(loaded["train_container"])
    smoke = run_probe_consumption_smoke(model, loaded["sim"], true_probe, large_perturbed_probe, cfg)
    smoke["perturbation_metadata"] = request.get("perturbation_metadata", {})
    smoke["child_preflight"] = loaded["preflight"]
    smoke["baseline_history"] = _history_payload(history)
    write_json(Path(request["probe_consumption_smoke_path"]), smoke)
    write_json(
        Path(request["child_invocation_path"]).with_name("child_status.json"),
        {"status": "ok", "return_code": 0, "probe_consumption_smoke_path": request["probe_consumption_smoke_path"]},
    )
    return 0


def run_condition_child_from_request(request_path: Path) -> int:
    request_path = Path(request_path)
    request = _load_child_request(request_path)
    write_child_invocation_artifact(request_path, request, "--child-condition-runner")
    cfg = _grid_config_from_payload(request["grid_config"])
    condition = _condition_from_payload(request["condition"])
    loaded = load_condition_inputs_for_child(
        request["bundle_path"],
        request["bundle_manifest_path"],
        request["assumed_probe_path"],
    )
    result = run_condition(
        condition=condition,
        sim=loaded["sim"],
        true_probe=np.asarray(_probe_from_npz(Path(request["assumed_probe_path"])), dtype=np.complex64),
        assumed_probe=loaded["assumed_probe"],
        branch_decision=request.get("branch_decision", "fixed_wrong_probe_training"),
        cfg=cfg,
        output_root=Path(request["run_root"]),
        baseline_model=None,
        canonical_data_checksums=loaded["preflight"]["canonical_data_checksums"],
    )
    result.setdefault("preflight", {})
    result["preflight"]["child_bundle_preflight"] = loaded["preflight"]
    write_json(Path(request["metrics_path"]), result)
    return 0 if result.get("status") == "ok" else 1


def run_full_or_smoke(
    args: argparse.Namespace,
    conditions: list[PerturbationCondition],
    runtime_provenance: dict[str, Any],
) -> int:
    from ptycho.workflows.grid_lines_workflow import simulate_grid_data

    output_root = Path(args.output_root)
    true_probe, probe_meta = build_canonical_probe(
        args.probe_npz,
        args.N,
        args.probe_scale_mode,
        args.probe_smoothing_sigma,
        probe_mask_diameter=args.probe_mask_diameter,
        probe_transform_pipeline=args.probe_transform_pipeline,
    )
    provenance = discover_provenance(args, probe_meta, runtime_provenance=runtime_provenance)
    write_json(output_root / "provenance_discovery.json", provenance)

    manifest = base_manifest(args, conditions, runtime_provenance=runtime_provenance)
    manifest["canonical_probe"] = probe_meta
    write_json(output_root / "manifest.json", manifest)

    cfg, execution_config = execution_config_from_args(args)
    manifest["execution_config"] = execution_config
    write_json(output_root / "manifest.json", manifest)
    sim = simulate_grid_data(cfg, true_probe)
    persisted_inputs = persist_true_measurements(output_root, sim, true_probe)
    bundle_paths = write_canonical_condition_bundle(output_root, sim, true_probe, cfg)
    canonical_data_checksums = build_canonical_data_checksums(sim)
    manifest["canonical_data_checksums"] = canonical_data_checksums
    manifest["canonical_condition_bundle"] = {
        "npz_path": str(bundle_paths["npz_path"]),
        "manifest_path": str(bundle_paths["manifest_path"]),
    }
    write_json(output_root / "manifest.json", manifest)

    smoke_condition = PerturbationCondition(
        condition_id="smoke_phase_noise_sigma_rad_0p4pi_seed11",
        perturbation_type="phase_noise_sigma_rad",
        value=0.4 * math.pi,
        seed=11,
    )
    large_perturbed_probe, smoke_perturb_meta = apply_probe_perturbation(true_probe, smoke_condition)
    smoke_request = build_smoke_child_request(
        output_root=output_root,
        bundle_paths=bundle_paths,
        true_probe_path=Path(persisted_inputs["true_probe"]),
        assumed_probe=large_perturbed_probe,
        smoke_condition=smoke_condition,
        perturbation_metadata=smoke_perturb_meta,
        cfg=cfg,
    )
    smoke_child_run = launch_child_process("--child-smoke-runner", smoke_request["request_path"])
    manifest["child_runs"]["smoke"] = smoke_child_run
    smoke_path = output_root / "probe_consumption_smoke.json"
    if smoke_child_run["return_code"] != 0 or not smoke_path.exists():
        stderr_excerpt = ""
        stderr_log = Path(smoke_child_run["stderr_log_path"])
        if stderr_log.exists():
            stderr_excerpt = stderr_log.read_text()[-4000:]
        smoke = {
            "decision": "pivot_text_only",
            "status": "failed",
            "child_return_code": smoke_child_run["return_code"],
            "stderr_excerpt": stderr_excerpt,
            "reason": "isolated smoke child failed or did not write probe_consumption_smoke.json",
        }
        write_json(smoke_path, smoke)
    else:
        smoke = json.loads(smoke_path.read_text())

    manifest["probe_consumption_smoke"] = smoke
    manifest["branch_decision"] = smoke["decision"]
    manifest["tf_memory_mitigation"] = {
        "smoke_baseline_subprocess_isolation": True,
        "per_condition_subprocess_isolation": True,
        "child_launch_policy": "smoke_then_conditions_sequential",
        "parent_trains_tensorflow_after_bundle_write": False,
        "reset_tf_state_deletes_model_singletons": True,
        "finding": "TF-REPEATED-MODEL-OOM-001",
    }
    manifest["condition_manifests"]["smoke"] = build_condition_manifest_entry(
        condition=smoke_condition,
        source_probe=true_probe,
        true_probe=true_probe,
        assumed_probe=large_perturbed_probe,
        perturbation_metadata=smoke_perturb_meta,
    )

    if args.smoke_only or smoke["decision"] == "pivot_text_only":
        if smoke["decision"] == "pivot_text_only":
            manifest["pivots_or_stop_conditions"].append("probe_consumption_smoke_pivot_text_only")
        write_json(output_root / "manifest.json", manifest)
        write_artifact_manifest(output_root)
        return 0 if args.smoke_only and smoke_child_run["return_code"] == 0 else 2

    if smoke["decision"] != "fixed_wrong_probe_training":
        manifest["pivots_or_stop_conditions"].append("unsupported_process_isolated_smoke_branch")
        write_json(output_root / "manifest.json", manifest)
        write_artifact_manifest(output_root)
        return 2

    condition_results: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        assumed_probe, perturb_meta = apply_probe_perturbation(true_probe, condition)
        save_probe_visuals(output_root / "probes", condition.condition_id, assumed_probe)
        manifest["condition_manifests"][condition.condition_id] = build_condition_manifest_entry(
            condition=condition,
            source_probe=true_probe,
            true_probe=true_probe,
            assumed_probe=assumed_probe,
            perturbation_metadata=perturb_meta,
        )
        child_request = build_condition_child_request(
            output_root=output_root,
            bundle_paths=bundle_paths,
            condition=condition,
            assumed_probe=assumed_probe,
            perturbation_metadata=perturb_meta,
            branch_decision=smoke["decision"],
            cfg=cfg,
        )
        child_run = launch_child_process("--child-condition-runner", child_request["request_path"])
        manifest["child_runs"][condition.condition_id] = child_run
        condition_metrics_path = output_root / "conditions" / condition.condition_id / "metrics.json"
        if child_run["return_code"] == 0 and condition_metrics_path.exists():
            result = json.loads(condition_metrics_path.read_text())
            if result.get("preflight") is not None:
                preflight = result["preflight"].get("child_bundle_preflight", result["preflight"])
                manifest["condition_manifests"][condition.condition_id] = build_condition_manifest_entry(
                    condition=condition,
                    source_probe=true_probe,
                    true_probe=true_probe,
                    assumed_probe=assumed_probe,
                    perturbation_metadata=perturb_meta,
                    preflight=preflight,
                )
        else:
            stderr_excerpt = ""
            stderr_log = Path(child_run["stderr_log_path"])
            if stderr_log.exists():
                stderr_excerpt = stderr_log.read_text()[-4000:]
            result = {
                "status": "failed",
                "condition_id": condition.condition_id,
                "perturbation_type": condition.perturbation_type,
                "value": condition.value,
                "seed": condition.seed,
                "reviewer_facing": condition.reviewer_facing,
                "error": "child_condition_process_failed_or_missing_metrics",
                "child_return_code": child_run["return_code"],
                "stderr_excerpt": stderr_excerpt,
            }
            write_json(condition_metrics_path, result)
        condition_results[condition.condition_id] = result
        write_json(output_root / "manifest.json", manifest)

    write_metrics_outputs(output_root, condition_results)
    infrastructure_gate = evaluate_infrastructure_failure_gate(condition_results)
    manifest["infrastructure_failure_gate"] = infrastructure_gate
    if infrastructure_gate["status"] == "stop_full_grid":
        manifest["pivots_or_stop_conditions"].append("infrastructure_failure_gate_stop_full_grid")
        write_json(output_root / "manifest.json", manifest)
        write_artifact_manifest(output_root)
        return 2

    baseline_gate = evaluate_baseline_comparability(
        condition_results,
        adopt_rerun_baseline=args.adopt_rerun_baseline,
    )
    manifest["baseline_comparability_gate"] = baseline_gate
    manifest["baseline_policy"] = baseline_gate["baseline_policy"]
    if not baseline_gate["claim_safe"]:
        manifest["pivots_or_stop_conditions"].append("baseline_comparability_gate_pivot_no_numeric_stress_table")
        write_json(output_root / "manifest.json", manifest)
        write_artifact_manifest(output_root)
        return 2

    mild_gate = evaluate_mild_perturbation_gate(condition_results)
    manifest["mild_perturbation_gate"] = mild_gate
    if not mild_gate["export_allowed"]:
        manifest["pivots_or_stop_conditions"].append("mild_perturbation_gate_blocked_paper_export")
        write_json(output_root / "manifest.json", manifest)
        write_artifact_manifest(output_root)
        return 2
    if not mild_gate["robustness_claim_safe"]:
        manifest["pivots_or_stop_conditions"].append("mild_perturbation_sensitivity_language_required")

    fig_path = write_stress_figure(output_root, condition_results)
    if fig_path is not None:
        manifest["stress_figure"] = str(fig_path)
    if smoke.get("baseline_history") is not None:
        manifest["baseline_history"] = smoke["baseline_history"]
    write_json(output_root / "manifest.json", manifest)
    write_artifact_manifest(output_root)

    if args.export_paper_assets:
        export_paper_assets(
            args,
            output_root,
            smoke,
            condition_results,
            baseline_gate=baseline_gate,
            mild_perturbation_gate=mild_gate,
        )
        write_artifact_manifest(output_root)
    return 0


def export_paper_assets(
    args: argparse.Namespace,
    output_root: Path,
    smoke: dict[str, Any],
    condition_results: dict[str, dict[str, Any]],
    *,
    baseline_gate: dict[str, Any],
    mild_perturbation_gate: dict[str, Any],
) -> None:
    if smoke.get("decision") == "pivot_text_only":
        raise RuntimeError("paper asset export blocked because probe smoke gate pivoted to text-only")
    if not baseline_gate.get("claim_safe"):
        raise RuntimeError("paper asset export blocked because baseline comparability gate is not claim-safe")
    if not mild_perturbation_gate.get("export_allowed"):
        raise RuntimeError("paper asset export blocked because mild perturbation gate did not permit export")
    source_figure = output_root / "figures" / "probe_mischaracterization_stress.png"
    if not source_figure.exists():
        raise RuntimeError(f"paper asset export blocked because required stress figure is missing: {source_figure}")
    paper_root = Path(args.paper_root)
    data_dir = paper_root / "data"
    figures_dir = paper_root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_root": str(output_root),
        "git_commit": _git_commit(),
        "branch_decision": smoke["decision"],
        "baseline_policy": baseline_gate["baseline_policy"],
        "baseline_comparability_gate": baseline_gate,
        "mild_perturbation_gate": mild_perturbation_gate,
        "claim_boundaries": {
            "no_trainable_probe_variant_added": True,
            "robustness_claim_safe": bool(mild_perturbation_gate.get("robustness_claim_safe")),
            "branch_language_required": (
                "fixed_wrong_probe_training"
                if smoke.get("decision") == "fixed_wrong_probe_training"
                else smoke.get("decision")
            ),
        },
        "no_trainable_probe_variant_added": True,
        "conditions": condition_results,
    }
    write_json(data_dir / "probe_mischaracterization_metrics.json", payload)
    shutil.copy2(source_figure, figures_dir / "probe_mischaracterization_stress.png")


def _git_commit() -> str | None:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(raw_argv)
    if args.child_smoke_runner:
        return run_smoke_child_from_request(args.child_request_json)
    if args.child_condition_runner:
        return run_condition_child_from_request(args.child_request_json)
    conditions = select_conditions(args.conditions)
    prepare_output_root(args.output_root, force=args.force)
    runtime_provenance = capture_study_runtime_provenance()
    write_invocation_artifacts(
        output_dir=args.output_root,
        script_path=SCRIPT_PATH,
        argv=raw_argv,
        parsed_args=vars(args),
        extra={"runtime_provenance": runtime_provenance},
    )

    if args.dry_run:
        true_probe, probe_meta = build_canonical_probe(
            args.probe_npz,
            args.N,
            args.probe_scale_mode,
            args.probe_smoothing_sigma,
            probe_mask_diameter=args.probe_mask_diameter,
            probe_transform_pipeline=args.probe_transform_pipeline,
        )
        provenance = discover_provenance(args, probe_meta, runtime_provenance=runtime_provenance)
        write_json(args.output_root / "provenance_discovery.json", provenance)
        manifest = base_manifest(args, conditions, runtime_provenance=runtime_provenance)
        manifest["canonical_probe"] = probe_meta
        manifest["dry_run"] = True
        manifest["reviewer_facing_metrics"] = False
        manifest["canonical_condition_bundle"] = {
            "dry_run_metadata_only": True,
            "npz_path": str(args.output_root / CANONICAL_BUNDLE_NPZ),
            "manifest_path": str(args.output_root / CANONICAL_BUNDLE_MANIFEST),
            "not_written_reason": "dry_run_does_not_simulate_or_launch_training",
        }
        manifest["tf_memory_mitigation"] = {
            "smoke_baseline_subprocess_isolation": True,
            "per_condition_subprocess_isolation": True,
            "child_launch_policy": "smoke_then_conditions_sequential",
            "parent_trains_tensorflow_after_bundle_write": False,
            "finding": "TF-REPEATED-MODEL-OOM-001",
        }
        condition_manifests = {}
        for condition in conditions:
            assumed_probe, perturb_meta = apply_probe_perturbation(true_probe, condition)
            condition_manifests[condition.condition_id] = build_condition_manifest_entry(
                condition=condition,
                source_probe=true_probe,
                true_probe=true_probe,
                assumed_probe=assumed_probe,
                perturbation_metadata=perturb_meta,
            )
        manifest["condition_manifests"] = condition_manifests
        write_json(args.output_root / "manifest.json", manifest)
        write_artifact_manifest(args.output_root)
        return 0

    return run_full_or_smoke(args, conditions, runtime_provenance)


if __name__ == "__main__":
    raise SystemExit(main())
