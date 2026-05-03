"""Dataset contract surface for the BRDT smoke/preflight dataset.

This module owns the locked physical-target, normalization, split, and
manifest contracts for the BRDT Born diffraction-tomography preflight
dataset. The physical forward operator (``BornRytovForward2D``) always
consumes physical scattering potential ``q``; the dataset stores both
``q_true_physical`` (canonical training/metric target) and
``q_true_norm`` (model-output/storage convenience). Any future physics
loss must unnormalize the model output before evaluating
``A(unnormalize(q_norm))``. See
``docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md``
and the operator-validation report for the authoritative contract this
module encodes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Locked smoke-dataset geometry, copied from the operator-validation
# contract. Downstream items must consume these values rather than
# redefining them.
LOCKED_GRID_SIZE: int = 128
LOCKED_DETECTOR_SIZE: int = 128
LOCKED_ANGLE_COUNT: int = 64
LOCKED_WAVELENGTH_PX: float = 8.0
LOCKED_MEDIUM_RI: float = 1.333
LOCKED_OPERATOR_MODE: str = "born"
LOCKED_NORMALIZE: str = "unitary_fft"
LOCKED_OPERATOR_MODULE: str = "ptycho_torch.physics.born_rytov_dt"
LOCKED_OPERATOR_CLASS: str = "BornRytovForward2D"

# Q formula: q(x,z) = k_m^2 * ((n / n_m)^2 - 1)
PHYSICAL_TARGET_FORMULA: str = "q(x,z) = k_m^2 * ((n(x,z) / n_m)^2 - 1)"
PHYSICS_LOSS_RULE: str = (
    "L_phys = || A(unnormalize(q_pred_norm)) - y || ; the operator always "
    "consumes physical q. Predictions in normalized space MUST be "
    "unnormalized via mean+std before being passed to A(...)."
)

DATASET_NAME: str = "brdt128_sparse_fullview_preflight"
DATASET_TIER: str = "feasibility"
DRY_RUN_MANIFEST_NAME: str = "dry_run_manifest.json"

# Phantom-family roster locked for this preflight (non-CDI families).
PHANTOM_FAMILIES: Tuple[str, ...] = (
    "overlapping_ellipses",
    "soft_blobs",
    "sparse_inclusions",
)

# Refractive-index contrast envelope kept inside the weak-scattering
# regime per the candidate-lane design.
DELTA_N_MIN: float = 0.002
DELTA_N_MAX: float = 0.03


def locked_angles() -> np.ndarray:
    """Return the locked ``A=64`` full-view angle grid in radians."""
    return np.linspace(0.0, 2.0 * math.pi, LOCKED_ANGLE_COUNT, endpoint=False)


def k_m() -> float:
    """Return the locked medium wave number ``k_m = 2*pi / lambda_px``."""
    return 2.0 * math.pi / LOCKED_WAVELENGTH_PX


def refractive_index_to_q(n: np.ndarray, n_m: float = LOCKED_MEDIUM_RI) -> np.ndarray:
    """Convert refractive index ``n(x,z)`` to physical scattering potential ``q``.

    Uses the canonical object-function definition
    ``q = k_m^2 * ((n / n_m)^2 - 1)``.
    """
    if n.dtype.kind != "f":
        n = n.astype(np.float64)
    ratio = n / float(n_m)
    return (k_m() ** 2) * (ratio * ratio - 1.0)


@dataclass(frozen=True)
class NormalizationStats:
    """Train-only normalization statistics for physical ``q``."""

    mean: float
    std: float
    qmin: float
    qmax: float

    @property
    def safe_std(self) -> float:
        return float(self.std) if abs(self.std) > 1e-30 else 1.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "q_mean_train": float(self.mean),
            "q_std_train": float(self.std),
            "q_min_train": float(self.qmin),
            "q_max_train": float(self.qmax),
        }


def compute_train_normalization(q_train: np.ndarray) -> NormalizationStats:
    """Compute mean/std/min/max from the train split only."""
    if q_train.size == 0:
        raise ValueError("compute_train_normalization requires a non-empty array")
    arr = q_train.astype(np.float64, copy=False)
    return NormalizationStats(
        mean=float(arr.mean()),
        std=float(arr.std()),
        qmin=float(arr.min()),
        qmax=float(arr.max()),
    )


def normalize_q(q: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Apply ``(q - mean) / std`` using train-only stats."""
    return (q - stats.mean) / stats.safe_std


def unnormalize_q(q_norm: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Invert ``normalize_q``: ``q_norm * std + mean``."""
    return q_norm * stats.safe_std + stats.mean


@dataclass(frozen=True)
class SplitCounts:
    """Locked train/val/test split counts for the smoke dataset."""

    train: int = 16
    val: int = 4
    test: int = 4

    @property
    def total(self) -> int:
        return self.train + self.val + self.test

    def as_dict(self) -> Dict[str, int]:
        return {"train": self.train, "val": self.val, "test": self.test}


def deterministic_object_seeds(
    counts: SplitCounts, split_seed: int
) -> Dict[str, List[int]]:
    """Generate disjoint per-sample object seeds for each split.

    Object seeds are drawn from a deterministic pool so train/val/test
    object-seed sets are disjoint by construction.
    """
    rng = np.random.default_rng(int(split_seed))
    pool_size = counts.total * 64
    pool = rng.choice(pool_size, size=counts.total, replace=False)
    pool = np.asarray(pool, dtype=np.int64).tolist()
    train = pool[: counts.train]
    val = pool[counts.train : counts.train + counts.val]
    test = pool[counts.train + counts.val : counts.total]
    return {"train": list(train), "val": list(val), "test": list(test)}


def deterministic_noise_seed(split_seed: int, split: str) -> int:
    """Return a stable per-split noise seed.

    Live generation must be reproducible across fresh interpreter
    processes, so this helper avoids Python's process-randomized string
    hash and uses a fixed split-to-offset mapping instead.
    """
    split_offsets = {"train": 0, "val": 1, "test": 2}
    if split not in split_offsets:
        raise ValueError(f"unknown split {split!r}")
    return int(split_seed) * 101 + split_offsets[split]


def assign_phantom_families(
    counts: SplitCounts, split_seed: int
) -> Dict[str, List[str]]:
    """Assign one phantom family per sample, balanced across each split."""
    families = list(PHANTOM_FAMILIES)
    rng = np.random.default_rng(int(split_seed) + 7919)
    out: Dict[str, List[str]] = {}
    for split, n in (("train", counts.train), ("val", counts.val), ("test", counts.test)):
        # Round-robin over families, then permute deterministically so
        # the per-sample family assignment is reproducible.
        repeats = (n + len(families) - 1) // len(families)
        ordered = (families * repeats)[:n]
        idx = rng.permutation(n)
        out[split] = [ordered[i] for i in idx]
    return out


def build_operator_block(operator_validation_path: str) -> Dict[str, Any]:
    """Build the operator section of the manifest from locked authority."""
    return {
        "module": LOCKED_OPERATOR_MODULE,
        "class": LOCKED_OPERATOR_CLASS,
        "mode": LOCKED_OPERATOR_MODE,
        "normalize": LOCKED_NORMALIZE,
        "grid_size": LOCKED_GRID_SIZE,
        "detector_size": LOCKED_DETECTOR_SIZE,
        "angle_count": LOCKED_ANGLE_COUNT,
        "angle_coverage": "full_view_0_to_2pi",
        "wavelength_px": LOCKED_WAVELENGTH_PX,
        "medium_ri": LOCKED_MEDIUM_RI,
        "k_m": k_m(),
        "output_layout": "(B, A, D, 2) real/imag",
        "validation_report": (
            "docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md"
        ),
        "validation_artifact": operator_validation_path,
    }


def build_manifest(
    *,
    output_root: str,
    operator_validation_path: str,
    counts: SplitCounts,
    split_seed: int,
    object_seeds: Dict[str, List[int]],
    families: Dict[str, List[str]],
    normalization: Optional[NormalizationStats],
    noise_sigma: float,
    measured_snr: Optional[Dict[str, float]],
    git_sha: str,
    git_dirty: bool,
    generation_command: str,
    environment: Dict[str, Any],
    artifact_paths: Dict[str, str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the dataset manifest with stable keys.

    The keys returned here are the dataset contract surface that
    downstream BRDT items (`brdt-task-adapters`, `brdt-four-row-preflight`)
    are allowed to depend on. New keys may be added; existing keys must
    not be renamed without an approved follow-up.
    """
    manifest: Dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_identity": {
            "name": DATASET_NAME,
            "tier": DATASET_TIER,
            "backlog_item": "2026-04-29-brdt-dataset-preflight",
            "output_root": output_root,
            "generation_command": generation_command,
            "git_sha": git_sha,
            "git_dirty": bool(git_dirty),
        },
        "physical_target": {
            "formula": PHYSICAL_TARGET_FORMULA,
            "q_units": "1/pixel^2 (k_m in rad/pixel)",
            "forward_input_is_physical_q": True,
            "model_output_space": "normalized_q",
            "physics_loss_rule": PHYSICS_LOSS_RULE,
        },
        "operator": build_operator_block(operator_validation_path),
        "split": {
            "split_seed": int(split_seed),
            "counts": counts.as_dict(),
            "object_seeds": object_seeds,
            "object_seeds_disjoint": True,
        },
        "phantom_distribution": {
            "families": list(PHANTOM_FAMILIES),
            "delta_n_min": DELTA_N_MIN,
            "delta_n_max": DELTA_N_MAX,
            "per_sample_family": families,
        },
        "noise": {
            "model": "complex_gaussian_iid",
            "noise_sigma_physical_units": float(noise_sigma),
            "measured_snr": measured_snr,
        },
        "normalization": (normalization.as_dict() if normalization is not None else None),
        "environment": environment,
        "artifacts": artifact_paths,
        "claim_boundary": (
            "Feasibility-only smoke dataset. NOT manuscript evidence. The "
            "later larger decision-support split, BRDT task adapters, and "
            "four-row preflight remain out of scope for this item."
        ),
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def manifest_required_keys() -> Tuple[str, ...]:
    """Stable top-level keys downstream consumers may rely on."""
    return (
        "schema_version",
        "dataset_identity",
        "physical_target",
        "operator",
        "split",
        "phantom_distribution",
        "noise",
        "normalization",
        "environment",
        "artifacts",
        "claim_boundary",
    )


def write_manifest(manifest: Dict[str, Any], path: str) -> None:
    """Serialize a manifest dict to JSON with sorted keys."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def validate_geometry_against_operator_authority(
    operator_authority: Dict[str, Any],
) -> List[str]:
    """Return a list of geometry mismatches against the locked operator authority.

    ``operator_authority`` is the ``operator`` block from the operator
    validation JSON. The smoke dataset must match the locked geometry
    exactly; any mismatch is a hard contract violation.
    """
    mismatches: List[str] = []
    expected = {
        "mode": LOCKED_OPERATOR_MODE,
        "normalize": LOCKED_NORMALIZE,
        "grid_size": LOCKED_GRID_SIZE,
        "detector_size": LOCKED_DETECTOR_SIZE,
        "wavelength_px": LOCKED_WAVELENGTH_PX,
        "medium_ri": LOCKED_MEDIUM_RI,
    }
    for key, want in expected.items():
        got = operator_authority.get(key)
        if got is None:
            mismatches.append(f"operator authority missing {key}")
            continue
        if isinstance(want, float):
            if not math.isclose(float(got), want, rel_tol=0.0, abs_tol=1e-9):
                mismatches.append(f"{key}: locked={want} authority={got}")
        else:
            if got != want:
                mismatches.append(f"{key}: locked={want!r} authority={got!r}")
    angle_count = operator_authority.get("angle_count")
    if angle_count is not None and int(angle_count) != LOCKED_ANGLE_COUNT:
        mismatches.append(
            f"angle_count: locked={LOCKED_ANGLE_COUNT} authority={angle_count}"
        )
    return mismatches


def reject_normalized_q_to_operator(routing: str) -> None:
    """Hard-stop guard against routing normalized q into the operator.

    Downstream code that wires the physics loss must call this guard with
    the symbolic name of the tensor being passed to the forward operator.
    Anything other than ``physical_q`` raises ``ValueError`` so the
    contract violation surfaces at the call site rather than as silent
    physical-units drift.
    """
    if routing != "physical_q":
        raise ValueError(
            "BRDT operator must be called on physical q. Got routing="
            f"{routing!r}. Unnormalize before calling A(...). "
            "See PHYSICS_LOSS_RULE."
        )
