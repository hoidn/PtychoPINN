"""BRDT four-row preflight orchestrator.

Runs the bounded decision-support preflight for the four-row roster
(`classical_born_backprop`, `unet`, `fno_vanilla`, `hybrid_resnet` or
`sru_net`) under one shared dataset, operator, input, split, loss, and
metric contract. Produces:

- a top-level ``preflight_manifest.json`` recording dataset identity,
  operator pointer, row roster, fixed sample IDs, training contract, and
  claim boundary;
- per-row directories with invocation provenance, training/eval
  summaries, and saved model state for the trained rows;
- aggregated ``metrics.json`` / ``metrics.csv`` / ``metric_schema.json``;
- fixed-sample ``q`` comparison, error-map, and sinogram-residual figures
  with matching ``.npy`` source arrays under ``visuals/`` and
  ``figures/source_arrays/``;
- a ``visual_manifest.json`` listing every saved figure/source pair.

This entrypoint is task-local and does NOT register BRDT in the CDI
generator registry.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ptycho_torch.physics import BornRytovForward2D
from scripts.studies.born_rytov_dt import comparison as comparison_mod
from scripts.studies.born_rytov_dt import dataset_contract as dc
from scripts.studies.born_rytov_dt import preflight_metrics as metrics_mod
from scripts.studies.born_rytov_dt import preflight_visuals as visuals_mod
from scripts.studies.born_rytov_dt.classical import (
    ClassicalBackendInfo,
    derive_born_init_image,
)
from scripts.studies.born_rytov_dt.data import (
    BRDTSmokeSplit,
    DatasetAuthority,
    brdt_collate,
    load_dataset_authority,
    sinogram_to_channels_first,
)
from scripts.studies.born_rytov_dt.lightning_module import BRDTTrainingModule
from scripts.studies.born_rytov_dt.model_based_inverse import (
    ModelBasedInverseConfig,
    optimize_born_inverse_batch,
)
from scripts.studies.born_rytov_dt.models import (
    AdapterBuildError,
    build_neural_adapter,
    build_sinogram_input_adapter,
)
from scripts.studies.born_rytov_dt.run_config import (
    DEFAULT_TRAINING_LABEL,
    HYBRID_FAMILY_ROW_IDS,
    LossWeights,
    OBJECTIVE_PRESETS,
    RowConfig,
    default_row_roster,
    make_blocked_row,
    resolve_objective_preset,
)
from scripts.studies.invocation_logging import (
    capture_runtime_provenance,
    write_invocation_artifacts,
)


SCRIPT_PATH = "scripts/studies/born_rytov_dt/run_preflight.py"
BACKLOG_ITEM = "2026-04-29-brdt-four-row-preflight"
NEXT_BACKLOG_ITEM = "2026-04-29-brdt-preflight-summary-promotion-decision"
PHYSICS_ONLY_BACKLOG_ITEM = "2026-05-04-brdt-physics-only-objective-ablation"
PREFLIGHT_MANIFEST_NAME = "preflight_manifest.json"
WRITER_LOCK_NAME = ".preflight.lock"
ROW_SUMMARY_NAME = "row_summary.json"
MODEL_BASED_INVERSE_VERSION = "model_based_born_inverse_v1"
MODEL_BASED_INVERSE_EXECUTION_PATH = "model_based_born_inverse"


# ---------------------------------------------------------------------------
# Writer lock (duplicate-writer protection)
# ---------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


@contextlib.contextmanager
def writer_lock(output_root: Path) -> Iterator[Path]:
    """Refuse to launch a duplicate writer against the same output root.

    A ``.preflight.lock`` file inside ``output_root`` records the active
    writer's PID and start timestamp. If the file already exists and the
    referenced PID is alive, this raises ``ValueError`` so the caller
    cannot silently overwrite artifacts being produced by a concurrent
    run. Stale locks (whose PID is gone) are reclaimed.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / WRITER_LOCK_NAME
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text())
        except Exception:
            existing = None
        if isinstance(existing, dict):
            other_pid = existing.get("pid")
            if isinstance(other_pid, int) and other_pid != os.getpid() and _pid_alive(other_pid):
                raise ValueError(
                    "active BRDT preflight writer detected at "
                    f"{lock_path} (pid={other_pid}, started_at="
                    f"{existing.get('started_at')!r}); refusing to launch "
                    "a duplicate writer against the same --output-root."
                )
    lock_payload = {
        "pid": int(os.getpid()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": (os.uname().nodename if hasattr(os, "uname") else "unknown"),
        "script": SCRIPT_PATH,
    }
    lock_path.write_text(json.dumps(lock_payload, indent=2) + "\n")
    try:
        yield lock_path
    finally:
        try:
            current = json.loads(lock_path.read_text())
        except Exception:
            current = None
        if isinstance(current, dict) and current.get("pid") == os.getpid():
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# Contract fingerprint (resume / skip)
# ---------------------------------------------------------------------------
def _stable_fingerprint(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_contract_fingerprint(
    *,
    row_id: str,
    model: str,
    training: str,
    input_mode: str,
    dataset_id: str,
    operator_pointer: str,
    training_contract: Mapping[str, Any],
    fixed_sample_ids: List[int],
    in_channels: int,
    classical_backend_name: str,
    execution_path: Optional[str] = None,
    solver_version: Optional[str] = None,
    solver_config: Optional[Mapping[str, Any]] = None,
) -> str:
    """Stable hash for one row's effective execution contract."""
    payload: Dict[str, Any] = {
        "row_id": row_id,
        "model": model,
        "training": training,
        "input_mode": input_mode,
        "dataset_id": dataset_id,
        "operator_pointer": operator_pointer,
        "training_contract": dict(training_contract),
        "fixed_sample_ids": [int(i) for i in fixed_sample_ids],
        "in_channels": int(in_channels),
        "classical_backend_name": classical_backend_name,
    }
    if execution_path is not None:
        payload["execution_path"] = execution_path
    if solver_version is not None:
        payload["solver_version"] = solver_version
    if solver_config is not None:
        payload["solver_config"] = dict(solver_config)
    return _stable_fingerprint(payload)


# ---------------------------------------------------------------------------
# Classical backend with narrow fix attempt
# ---------------------------------------------------------------------------
def attempt_classical_backend_with_fix(
    prefer_odtbrain: bool = True,
) -> Tuple[ClassicalBackendInfo, List[Dict[str, str]]]:
    """Detect the classical backend and document any narrow fix attempt.

    Returns ``(backend, attempts)`` where ``attempts`` is a list of
    structured records suitable for the row-level blocker payload. The
    only narrow fix in scope is to invalidate import caches and retry
    the import once; we never mutate the environment, install packages,
    or rewrite ``sys.path``. The captured ImportError text is recorded
    so the row-level blocker explicitly documents what was tried.
    """
    attempts: List[Dict[str, str]] = []
    if prefer_odtbrain:
        try:
            importlib.import_module("odtbrain")
            attempts.append({"step": "import_odtbrain", "outcome": "succeeded"})
            return (
                ClassicalBackendInfo(
                    name="odtbrain",
                    reason="odtbrain_import_succeeded",
                    claim_boundary="external_oracle",
                ),
                attempts,
            )
        except Exception as exc:
            attempts.append(
                {
                    "step": "import_odtbrain",
                    "outcome": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        try:
            importlib.invalidate_caches()
            importlib.import_module("odtbrain")
            attempts.append(
                {"step": "retry_after_invalidate_caches", "outcome": "succeeded"}
            )
            return (
                ClassicalBackendInfo(
                    name="odtbrain",
                    reason="odtbrain_import_succeeded_after_cache_invalidation",
                    claim_boundary="external_oracle",
                ),
                attempts,
            )
        except Exception as exc:
            attempts.append(
                {
                    "step": "retry_after_invalidate_caches",
                    "outcome": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return (
        ClassicalBackendInfo(
            name="local_adjoint",
            reason="odtbrain_unavailable_after_narrow_fix",
            claim_boundary="feasibility_only",
        ),
        attempts,
    )


def _resolve_manifest_pointer(manifest_path: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def classical_inverse_authorization(
    manifest: Mapping[str, Any], *, manifest_path: Path
) -> Dict[str, Any]:
    """Authorize ODTbrain classical rows only after inverse validation passes."""
    operator_block = manifest.get("operator") or {}
    pointer = operator_block.get("validation_artifact") or operator_block.get(
        "validation_report"
    )
    if not pointer:
        return {
            "may_run_classical_row": False,
            "status": "missing_validation_pointer",
            "blocker_reason": "odtbrain_inverse_consistency_unverified",
            "blocker_message": (
                "Classical ODTbrain backprop is not authorized because the "
                "dataset manifest does not point to an operator-validation artifact."
            ),
        }
    validation_path = _resolve_manifest_pointer(Path(manifest_path), str(pointer))
    if not validation_path.exists():
        return {
            "may_run_classical_row": False,
            "status": "missing_validation_artifact",
            "validation_artifact": str(validation_path),
            "blocker_reason": "odtbrain_inverse_consistency_unverified",
            "blocker_message": (
                "Classical ODTbrain backprop is not authorized because the "
                f"operator-validation artifact is missing: {validation_path}"
            ),
        }
    try:
        payload = json.loads(validation_path.read_text())
    except Exception as exc:  # noqa: BLE001 - provenance artifact may be corrupt
        return {
            "may_run_classical_row": False,
            "status": "invalid_validation_artifact",
            "validation_artifact": str(validation_path),
            "blocker_reason": "odtbrain_inverse_consistency_unverified",
            "blocker_message": (
                "Classical ODTbrain backprop is not authorized because the "
                f"operator-validation artifact could not be parsed: {exc}"
            ),
        }
    checks = payload.get("checks") or []
    odtbrain_check = next(
        (
            check
            for check in checks
            if isinstance(check, Mapping)
            and check.get("name") == "odtbrain_inverse_consistency"
        ),
        None,
    )
    if not isinstance(odtbrain_check, Mapping):
        return {
            "may_run_classical_row": False,
            "status": "missing_odtbrain_inverse_check",
            "validation_artifact": str(validation_path),
            "blocker_reason": "odtbrain_inverse_consistency_unverified",
            "blocker_message": (
                "Classical ODTbrain backprop is not authorized because the "
                "operator-validation artifact has no odtbrain_inverse_consistency check."
            ),
        }
    status = str(odtbrain_check.get("status", "unknown"))
    if status == "pass":
        return {
            "may_run_classical_row": True,
            "status": "pass",
            "validation_artifact": str(validation_path),
        }
    return {
        "may_run_classical_row": False,
        "status": status,
        "validation_artifact": str(validation_path),
        "metric": odtbrain_check.get("metric"),
        "tolerance": odtbrain_check.get("tolerance"),
        "blocker_reason": (
            "odtbrain_inverse_consistency_failed"
            if status == "fail"
            else f"odtbrain_inverse_consistency_{status}"
        ),
        "blocker_message": (
            "Classical ODTbrain backprop is not authorized because the "
            f"operator-validation check status is {status!r}."
        ),
    }


# ---------------------------------------------------------------------------
# Manifest gates
# ---------------------------------------------------------------------------
def assert_decision_support_manifest(manifest: Mapping[str, Any]) -> None:
    """Refuse smoke manifests as decision-support authority.

    Raises ``ValueError`` if the manifest is not a decision-support
    profile artifact.
    """
    identity = manifest.get("dataset_identity", {})
    tier = identity.get("tier")
    name = identity.get("name")
    backlog = identity.get("backlog_item")
    if tier != "decision_support":
        raise ValueError(
            "BRDT four-row preflight requires a decision_support manifest; "
            f"got tier={tier!r}, name={name!r}, backlog_item={backlog!r}. "
            "Generate the larger split with --dataset-profile decision_support."
        )
    if name != dc.DECISION_SUPPORT_DATASET_NAME:
        raise ValueError(
            "decision_support manifest has unexpected dataset name: "
            f"{name!r} (expected {dc.DECISION_SUPPORT_DATASET_NAME!r})"
        )


# ---------------------------------------------------------------------------
# Row roster + fixed sample IDs
# ---------------------------------------------------------------------------
def resolve_row_roster(
    *,
    manifest_path: Path,
    hybrid_label: str,
    neural_training_label: str = DEFAULT_TRAINING_LABEL,
    selected_row_ids: Optional[List[str]] = None,
) -> List[RowConfig]:
    """Return the locked four-row roster against a decision-support manifest.

    ``selected_row_ids`` optionally restricts the returned roster to a
    subset (preserving order). Unknown row IDs raise ``ValueError`` so the
    physics-only ablation cannot silently include or skip a row.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    assert_decision_support_manifest(manifest)
    operator_block = manifest.get("operator", {})
    operator_version = (
        operator_block.get("validation_artifact")
        or operator_block.get("validation_report")
        or "unspecified"
    )
    dataset_id = manifest["dataset_identity"]["name"]
    full_roster = default_row_roster(
        dataset_id=str(dataset_id),
        operator_version=str(operator_version),
        hybrid_label=hybrid_label,
        neural_training_label=neural_training_label,
    )
    if selected_row_ids is None:
        return full_roster
    available = {row.row_id: row for row in full_roster}
    unknown = [rid for rid in selected_row_ids if rid not in available]
    if unknown:
        raise ValueError(
            f"unknown selected row_ids {unknown!r}; available: {list(available)}"
        )
    return [available[rid] for rid in selected_row_ids]


def choose_fixed_sample_ids(
    manifest_path: Path,
    *,
    count: int,
    seed: int,
) -> List[int]:
    """Pick deterministic test-split sample IDs to anchor the visual bundle."""
    manifest = json.loads(Path(manifest_path).read_text())
    counts = manifest["split"]["counts"]
    test_count = int(counts["test"])
    if test_count <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    pool = np.arange(test_count)
    rng.shuffle(pool)
    chosen = pool[: max(0, min(int(count), test_count))]
    return [int(i) for i in chosen]


# ---------------------------------------------------------------------------
# Operator + device
# ---------------------------------------------------------------------------
def _build_operator(device: torch.device) -> BornRytovForward2D:
    op = BornRytovForward2D(
        grid_size=dc.LOCKED_GRID_SIZE,
        detector_size=dc.LOCKED_DETECTOR_SIZE,
        angles=torch.from_numpy(dc.locked_angles()),
        wavelength_px=dc.LOCKED_WAVELENGTH_PX,
        medium_ri=dc.LOCKED_MEDIUM_RI,
        mode="born",
        normalize="unitary_fft",
    )
    return op.to(device)


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _device_name(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return device.type


def _model_based_inverse_config(authority: DatasetAuthority) -> ModelBasedInverseConfig:
    """Resolve the model-based inverse settings from env plus dataset bounds."""
    return ModelBasedInverseConfig(
        steps=int(os.environ.get("BRDT_MODEL_BASED_INVERSE_STEPS", "300")),
        learning_rate=float(os.environ.get("BRDT_MODEL_BASED_INVERSE_LR", "0.002")),
        tv_weight=float(os.environ.get("BRDT_MODEL_BASED_INVERSE_TV", "0.0001")),
        l2_weight=float(os.environ.get("BRDT_MODEL_BASED_INVERSE_L2", "0.000001")),
        clamp_min=float(authority.normalization.qmin),
        clamp_max=float(authority.normalization.qmax),
        init="zeros",
    )


def _born_init_backend() -> ClassicalBackendInfo:
    """Stable local adjoint used only to build neural-row input images."""
    return ClassicalBackendInfo(
        name="local_adjoint",
        reason="born_init_image_uses_locked_local_adjoint",
        claim_boundary="initialization_only",
    )


# ---------------------------------------------------------------------------
# Per-row execution
# ---------------------------------------------------------------------------
@dataclass
class TrainingContract:
    """Locked training contract shared across neural rows."""

    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 2e-4
    optimizer: str = "Adam"
    loss_weights: LossWeights = field(default_factory=LossWeights)
    seed: int = 42
    scheduler: Optional[str] = None
    plateau_factor: Optional[float] = None
    plateau_patience: Optional[int] = None
    plateau_threshold: Optional[float] = None
    plateau_min_lr: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "optimizer": self.optimizer,
            "loss_weights": self.loss_weights.as_dict(),
            "seed": int(self.seed),
        }
        if self.scheduler is not None:
            payload["scheduler"] = str(self.scheduler)
            payload["plateau_factor"] = float(
                0.5 if self.plateau_factor is None else self.plateau_factor
            )
            payload["plateau_patience"] = int(
                2 if self.plateau_patience is None else self.plateau_patience
            )
            payload["plateau_threshold"] = float(
                0.0 if self.plateau_threshold is None else self.plateau_threshold
            )
            payload["plateau_min_lr"] = float(
                1e-5 if self.plateau_min_lr is None else self.plateau_min_lr
            )
        return payload


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs so model init + shuffling are deterministic.

    Used before each neural-row build/train so the bundle is regenerable
    under the design's "fixed seeds for local baseline and Hybrid ResNet
    runs" requirement (`docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`).
    """
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)


def _prepare_input(
    sinogram: torch.Tensor,
    *,
    operator: BornRytovForward2D,
    backend: ClassicalBackendInfo,
    in_channels: int,
    input_mode: str = "born_init_image",
) -> torch.Tensor:
    if input_mode == "sinogram":
        if in_channels != 2:
            raise ValueError("sinogram input requires in_channels=2")
        return sinogram_to_channels_first(sinogram)
    if input_mode != "born_init_image":
        raise ValueError(f"unsupported input_mode={input_mode!r}")
    init = derive_born_init_image(sinogram, operator=operator, backend=backend)
    if in_channels == 1:
        return init
    if in_channels == 2:
        return torch.cat([init, torch.zeros_like(init)], dim=1)
    raise ValueError(f"unsupported in_channels={in_channels}")


def _train_neural_row(
    *,
    row: RowConfig,
    authority: DatasetAuthority,
    operator: BornRytovForward2D,
    backend: ClassicalBackendInfo,
    device: torch.device,
    contract: TrainingContract,
    in_channels: int,
    output_dir: Path,
) -> Tuple[Optional[BRDTTrainingModule], Dict[str, Any], float]:
    """Train one neural row. Returns (module, runtime_meta, train_seconds)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(contract.seed)
    try:
        if row.input_mode == "sinogram":
            coordinate_channels = None
            if isinstance(row.extra, Mapping):
                coordinate_channels = row.extra.get("coordinate_channels")
            adapter = build_sinogram_input_adapter(
                architecture=row.model,
                out_channels=1,
                grid_size=dc.LOCKED_GRID_SIZE,
                coordinate_channels=coordinate_channels,
            ).to(device)
        else:
            adapter = build_neural_adapter(
                architecture=row.model,
                in_channels=in_channels,
                out_channels=1,
                grid_size=dc.LOCKED_GRID_SIZE,
            ).to(device)
    except AdapterBuildError as exc:
        return None, {"adapter_build_error": exc.to_payload()}, 0.0

    module = BRDTTrainingModule(
        model=adapter,
        operator=operator,
        normalization=authority.normalization,
        weights=contract.loss_weights,
        output_space="normalized_q",
    ).to(device)

    train_split = BRDTSmokeSplit(
        authority.split_paths["train"], normalization=authority.normalization
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(contract.seed))
    loader = DataLoader(
        train_split,
        batch_size=int(contract.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=brdt_collate,
        generator=loader_generator,
    )
    optim = torch.optim.Adam(module.parameters(), lr=float(contract.learning_rate))
    scheduler = None
    scheduler_name = contract.scheduler
    if scheduler_name is not None:
        if scheduler_name != "reduce_on_plateau":
            raise ValueError(
                f"unsupported scheduler={scheduler_name!r}; only 'reduce_on_plateau' is supported"
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="min",
            factor=float(
                0.5 if contract.plateau_factor is None else contract.plateau_factor
            ),
            patience=int(
                2
                if contract.plateau_patience is None
                else contract.plateau_patience
            ),
            threshold=float(
                0.0
                if contract.plateau_threshold is None
                else contract.plateau_threshold
            ),
            min_lr=float(
                1e-5 if contract.plateau_min_lr is None else contract.plateau_min_lr
            ),
        )
    started = time.perf_counter()
    last_breakdown: Dict[str, float] = {}
    train_steps = 0
    history_rows: List[Dict[str, Any]] = []
    module.train()
    for epoch_idx in range(int(contract.epochs)):
        epoch_total_sum = 0.0
        epoch_batches = 0
        component_sums: Dict[str, float] = {}
        for batch in loader:
            sino = batch["sinogram"].to(device)
            x = _prepare_input(
                sino,
                operator=operator,
                backend=backend,
                in_channels=in_channels,
                input_mode=row.input_mode,
            )
            q_pred = module(x)
            total, breakdown = module.compute_loss(
                q_pred=q_pred,
                q_true_norm=batch["q_true_norm"].to(device),
                q_true_physical=batch["q_true_physical"].to(device),
                sinogram_obs=sino,
            )
            optim.zero_grad(set_to_none=True)
            total.backward()
            optim.step()
            train_steps += 1
            last_breakdown = breakdown.to_dict()
            total_value = float(total.detach().item())
            epoch_total_sum += total_value
            epoch_batches += 1
            for key, value in last_breakdown.items():
                component_sums[key] = component_sums.get(key, 0.0) + float(value)
        avg_total = epoch_total_sum / max(1, epoch_batches)
        avg_components = {
            key: float(value / max(1, epoch_batches))
            for key, value in component_sums.items()
        }
        lr_before = float(optim.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step(avg_total)
        lr_after = float(optim.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": int(epoch_idx + 1),
                "train_total_loss": float(avg_total),
                "train_loss_components": avg_components,
                "learning_rate": float(lr_after),
                "scheduler_metric": float(avg_total),
                "lr_reduced": bool(lr_after < (lr_before - 1e-15)),
            }
        )
    elapsed = time.perf_counter() - started

    info = adapter.info()
    history_json_path = output_dir / "history.json"
    history_csv_path = output_dir / "history.csv"
    history_payload: Dict[str, Any] = {
        "schema_version": "brdt_training_history_v1",
        "row_id": row.row_id,
        "architecture": row.model,
        "epochs": history_rows,
    }
    if scheduler_name is not None:
        history_payload["scheduler"] = {
            "name": scheduler_name,
            "factor": float(
                0.5 if contract.plateau_factor is None else contract.plateau_factor
            ),
            "patience": int(
                2 if contract.plateau_patience is None else contract.plateau_patience
            ),
            "threshold": float(
                0.0
                if contract.plateau_threshold is None
                else contract.plateau_threshold
            ),
            "min_lr": float(
                1e-5 if contract.plateau_min_lr is None else contract.plateau_min_lr
            ),
        }
    history_json_path.write_text(
        json.dumps(history_payload, indent=2, sort_keys=True) + "\n"
    )
    with history_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "epoch",
                "train_total_loss",
                "train_image_loss",
                "train_physics_loss",
                "train_relative_physics_loss",
                "train_tv_loss",
                "train_positivity_loss",
                "learning_rate",
                "scheduler_metric",
                "lr_reduced",
            ],
        )
        writer.writeheader()
        for record in history_rows:
            comps = record.get("train_loss_components") or {}
            writer.writerow(
                {
                    "epoch": record["epoch"],
                    "train_total_loss": record["train_total_loss"],
                    "train_image_loss": comps.get("image", ""),
                    "train_physics_loss": comps.get("physics", ""),
                    "train_relative_physics_loss": comps.get(
                        "relative_physics", ""
                    ),
                    "train_tv_loss": comps.get("tv", ""),
                    "train_positivity_loss": comps.get("positivity", ""),
                    "learning_rate": record["learning_rate"],
                    "scheduler_metric": record["scheduler_metric"],
                    "lr_reduced": int(bool(record["lr_reduced"])),
                }
            )
    runtime_meta = {
        "parameter_count": int(info.parameter_count),
        "arch_kwargs": dict(info.arch_kwargs),
        "train_steps": int(train_steps),
        "final_loss_breakdown": last_breakdown,
        "history_json_path": str(history_json_path),
        "history_csv_path": str(history_csv_path),
        "history_length": int(len(history_rows)),
    }
    if scheduler_name is not None:
        runtime_meta["scheduler"] = dict(history_payload["scheduler"])
    # Save model state for reproducibility.
    state_path = output_dir / "model_state.pt"
    torch.save(
        {
            "architecture": row.model,
            "in_channels": in_channels,
            "arch_kwargs": dict(info.arch_kwargs),
            "state_dict": adapter.state_dict(),
        },
        state_path,
    )
    runtime_meta["model_state_path"] = str(state_path)
    return module, runtime_meta, elapsed


def _evaluate_split(
    *,
    module: Optional[BRDTTrainingModule],
    authority: DatasetAuthority,
    operator: BornRytovForward2D,
    backend: ClassicalBackendInfo,
    device: torch.device,
    in_channels: int,
    classical_only: bool,
    fixed_sample_ids: List[int],
    out_dir: Path,
    input_mode: str = "born_init_image",
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[int, Dict[str, np.ndarray]],
    Dict[str, float],
]:
    """Run an evaluation pass on the test split.

    Returns ``(image_metrics, measurement_metrics, fixed_sample_arrays,
    output_dynamic_range)``. ``fixed_sample_arrays[sample_id]`` is a dict
    with keys ``q_pred``, ``q_target``, ``sino_pred``, ``sino_obs``.
    """
    test_split = BRDTSmokeSplit(
        authority.split_paths["test"], normalization=authority.normalization
    )
    loader = DataLoader(
        test_split,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=brdt_collate,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if module is not None:
        module.eval()

    image_pred_chunks: List[np.ndarray] = []
    image_target_chunks: List[np.ndarray] = []
    meas_pred_chunks: List[np.ndarray] = []
    meas_obs_chunks: List[np.ndarray] = []
    sample_arrays: Dict[int, Dict[str, np.ndarray]] = {}
    sample_ids = set(int(i) for i in fixed_sample_ids)

    cursor = 0
    for batch in loader:
        sino_obs = batch["sinogram"].to(device)
        x = _prepare_input(
            sino_obs,
            operator=operator,
            backend=backend,
            in_channels=in_channels,
            input_mode="born_init_image" if classical_only else input_mode,
        ).detach()
        if classical_only or module is None:
            q_phys_pred = x[:, :1].detach()
        else:
            with torch.no_grad():
                q_pred_norm = module(x.to(device))
                q_phys_pred = module.to_physical_q(q_pred_norm)
        with torch.no_grad():
            sino_pred = operator(q_phys_pred.to(device))
        q_target_phys = batch["q_true_physical"].to(device)

        pred_np = q_phys_pred.detach().cpu().numpy()
        targ_np = q_target_phys.detach().cpu().numpy()
        sino_pred_np = sino_pred.detach().cpu().numpy()
        sino_obs_np = sino_obs.detach().cpu().numpy()

        image_pred_chunks.append(pred_np)
        image_target_chunks.append(targ_np)
        meas_pred_chunks.append(sino_pred_np)
        meas_obs_chunks.append(sino_obs_np)
        for offset in range(pred_np.shape[0]):
            global_id = cursor + offset
            if global_id in sample_ids:
                sample_arrays[global_id] = {
                    "q_pred": pred_np[offset, 0],
                    "q_target": targ_np[offset, 0],
                    "sino_pred": sino_pred_np[offset],
                    "sino_obs": sino_obs_np[offset],
                }
        cursor += pred_np.shape[0]

    image_pred = np.concatenate(image_pred_chunks, axis=0) if image_pred_chunks else np.zeros((0,))
    image_target = (
        np.concatenate(image_target_chunks, axis=0) if image_target_chunks else np.zeros((0,))
    )
    meas_pred = np.concatenate(meas_pred_chunks, axis=0) if meas_pred_chunks else np.zeros((0,))
    meas_obs = np.concatenate(meas_obs_chunks, axis=0) if meas_obs_chunks else np.zeros((0,))

    image_metrics = metrics_mod.image_space_metrics_phys(image_pred, image_target)
    meas_metrics = metrics_mod.measurement_space_metrics(meas_pred, meas_obs)
    image_metrics.update(metrics_mod.supporting_image_metrics(image_pred, image_target))
    return image_metrics, meas_metrics, sample_arrays, _output_dynamic_range_stats(image_pred)


def _output_dynamic_range_stats(image_pred: np.ndarray) -> Dict[str, float]:
    """Aggregate physical-q prediction stats for collapse detection.

    Returns ``min``, ``max``, ``mean``, ``std`` over the full evaluation
    prediction array. Empty inputs yield ``nan`` for each field.
    """
    if image_pred.size == 0:
        nan = float("nan")
        return {
            "physical_q_min": nan,
            "physical_q_max": nan,
            "physical_q_mean": nan,
            "physical_q_std": nan,
            "physical_q_ptp": nan,
        }
    arr = image_pred.astype(np.float64)
    return {
        "physical_q_min": float(np.min(arr)),
        "physical_q_max": float(np.max(arr)),
        "physical_q_mean": float(np.mean(arr)),
        "physical_q_std": float(np.std(arr)),
        "physical_q_ptp": float(np.max(arr) - np.min(arr)),
    }


def _evaluate_model_based_inverse_split(
    *,
    authority: DatasetAuthority,
    operator: BornRytovForward2D,
    device: torch.device,
    batch_size: int,
    fixed_sample_ids: List[int],
    config: ModelBasedInverseConfig,
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[int, Dict[str, np.ndarray]],
    Dict[str, Any],
]:
    """Optimize physical ``q`` for every test batch using the locked operator."""
    test_split = BRDTSmokeSplit(
        authority.split_paths["test"], normalization=authority.normalization
    )
    loader = DataLoader(
        test_split,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=brdt_collate,
    )

    image_pred_chunks: List[np.ndarray] = []
    image_target_chunks: List[np.ndarray] = []
    meas_pred_chunks: List[np.ndarray] = []
    meas_obs_chunks: List[np.ndarray] = []
    sample_arrays: Dict[int, Dict[str, np.ndarray]] = {}
    sample_ids = set(int(i) for i in fixed_sample_ids)
    batch_infos: List[Dict[str, Any]] = []

    cursor = 0
    for batch in loader:
        sino_obs = batch["sinogram"].to(device)
        q_phys_pred, info = optimize_born_inverse_batch(
            sinogram_obs=sino_obs,
            operator=operator,
            config=config,
        )
        with torch.no_grad():
            sino_pred = operator(q_phys_pred.to(device))
        q_target_phys = batch["q_true_physical"].to(device)

        pred_np = q_phys_pred.detach().cpu().numpy()
        targ_np = q_target_phys.detach().cpu().numpy()
        sino_pred_np = sino_pred.detach().cpu().numpy()
        sino_obs_np = sino_obs.detach().cpu().numpy()

        image_pred_chunks.append(pred_np)
        image_target_chunks.append(targ_np)
        meas_pred_chunks.append(sino_pred_np)
        meas_obs_chunks.append(sino_obs_np)
        batch_infos.append(info)
        for offset in range(pred_np.shape[0]):
            global_id = cursor + offset
            if global_id in sample_ids:
                sample_arrays[global_id] = {
                    "q_pred": pred_np[offset, 0],
                    "q_target": targ_np[offset, 0],
                    "sino_pred": sino_pred_np[offset],
                    "sino_obs": sino_obs_np[offset],
                }
        cursor += pred_np.shape[0]

    image_pred = (
        np.concatenate(image_pred_chunks, axis=0)
        if image_pred_chunks
        else np.zeros((0,))
    )
    image_target = (
        np.concatenate(image_target_chunks, axis=0)
        if image_target_chunks
        else np.zeros((0,))
    )
    meas_pred = (
        np.concatenate(meas_pred_chunks, axis=0)
        if meas_pred_chunks
        else np.zeros((0,))
    )
    meas_obs = (
        np.concatenate(meas_obs_chunks, axis=0)
        if meas_obs_chunks
        else np.zeros((0,))
    )

    image_metrics = metrics_mod.image_space_metrics_phys(image_pred, image_target)
    meas_metrics = metrics_mod.measurement_space_metrics(meas_pred, meas_obs)
    image_metrics.update(metrics_mod.supporting_image_metrics(image_pred, image_target))
    initial_values = [
        float(info["initial_relative_physics_l2"]) for info in batch_infos
    ]
    final_values = [float(info["final_relative_physics_l2"]) for info in batch_infos]
    solver_summary = {
        "solver": "adam_direct_q",
        "version": MODEL_BASED_INVERSE_VERSION,
        "config": config.as_dict(),
        "batch_count": len(batch_infos),
        "mean_initial_relative_physics_l2": (
            float(np.mean(initial_values)) if initial_values else float("nan")
        ),
        "mean_final_relative_physics_l2": (
            float(np.mean(final_values)) if final_values else float("nan")
        ),
    }
    return image_metrics, meas_metrics, sample_arrays, solver_summary


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _runtime_environment_block() -> Dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        "python": sys.version.split()[0],
    }


def _row_runtime_block(device: torch.device) -> Dict[str, Any]:
    """Per-row runtime/hardware metadata for invocation artifacts."""
    cuda_available = bool(torch.cuda.is_available())
    return {
        "device": str(device),
        "device_name": _device_name(device),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "python": sys.version.split()[0],
    }


def _save_fixed_sample_arrays(
    *,
    source_arrays_dir: Path,
    sample_arrays: Mapping[int, Mapping[str, np.ndarray]],
    row_id: str,
) -> Dict[int, Dict[str, str]]:
    """Persist per-row fixed-sample arrays so resume can rebuild the bundle.

    Returns ``{sample_id: {key: relative_path}}`` for the manifest.
    """
    paths: Dict[int, Dict[str, str]] = {}
    source_arrays_dir.mkdir(parents=True, exist_ok=True)
    for sid, arrays in sample_arrays.items():
        target_path = source_arrays_dir / f"sample_{int(sid):04d}_q_target.npy"
        sino_obs_path = source_arrays_dir / f"sample_{int(sid):04d}_sino_obs.npy"
        pred_path = source_arrays_dir / f"sample_{int(sid):04d}_{row_id}_q_pred.npy"
        sino_pred_path = (
            source_arrays_dir / f"sample_{int(sid):04d}_{row_id}_sino_pred.npy"
        )
        np.save(target_path, arrays["q_target"])
        np.save(sino_obs_path, arrays["sino_obs"])
        np.save(pred_path, arrays["q_pred"])
        np.save(sino_pred_path, arrays["sino_pred"])
        paths[int(sid)] = {
            "q_target": str(target_path),
            "sino_obs": str(sino_obs_path),
            "q_pred": str(pred_path),
            "sino_pred": str(sino_pred_path),
        }
    return paths


def _load_saved_row_arrays(
    *,
    source_arrays_dir: Path,
    row_id: str,
    fixed_sample_ids: List[int],
) -> Optional[Dict[int, Dict[str, np.ndarray]]]:
    """Load per-row fixed-sample arrays from disk, or ``None`` if any missing."""
    sample_arrays: Dict[int, Dict[str, np.ndarray]] = {}
    for sid in fixed_sample_ids:
        target_path = source_arrays_dir / f"sample_{int(sid):04d}_q_target.npy"
        sino_obs_path = source_arrays_dir / f"sample_{int(sid):04d}_sino_obs.npy"
        pred_path = source_arrays_dir / f"sample_{int(sid):04d}_{row_id}_q_pred.npy"
        sino_pred_path = (
            source_arrays_dir / f"sample_{int(sid):04d}_{row_id}_sino_pred.npy"
        )
        if not all(
            p.exists() for p in (target_path, sino_obs_path, pred_path, sino_pred_path)
        ):
            return None
        sample_arrays[int(sid)] = {
            "q_target": np.load(target_path),
            "sino_obs": np.load(sino_obs_path),
            "q_pred": np.load(pred_path),
            "sino_pred": np.load(sino_pred_path),
        }
    return sample_arrays


def _row_metrics_from_summary(
    summary: Mapping[str, Any], *, default_paper_label: str, default_arch: str
) -> metrics_mod.RowMetrics:
    """Reconstruct a ``RowMetrics`` from a stored ``row_summary.json`` payload."""
    image_metrics = dict(summary.get("image_metrics") or summary.get("image") or {})
    supporting = dict(summary.get("supporting") or {})
    for key in metrics_mod.SUPPORTING_METRICS:
        if key in image_metrics and key not in supporting:
            supporting[key] = image_metrics.pop(key)
    return metrics_mod.RowMetrics(
        row_id=str(summary.get("row_id")),
        paper_label=str(summary.get("paper_label", default_paper_label)),
        architecture=str(summary.get("architecture", default_arch)),
        row_status=str(summary.get("row_status", "completed")),
        image=image_metrics,
        measurement=dict(
            summary.get("measurement_metrics") or summary.get("measurement") or {}
        ),
        supporting=supporting,
        runtime=dict(summary.get("runtime") or {}),
        blocker_reason=summary.get("blocker_reason") or None,
        blocker_message=summary.get("blocker_message") or None,
    )


def _maybe_resume_row(
    *,
    row: "RowConfig",
    row_dir: Path,
    source_arrays_dir: Path,
    fixed_sample_ids: List[int],
    expected_fingerprint: str,
    expected_execution_path: Optional[str] = None,
) -> Optional[Tuple[metrics_mod.RowMetrics, Dict[str, Any], Dict[int, Dict[str, np.ndarray]]]]:
    """Return cached row outputs if a prior run with the same contract finished.

    Returns ``(row_metrics, row_payload, sample_arrays)`` if resume is
    valid, otherwise ``None``. Resume requires the row's
    ``row_summary.json`` to carry the same contract fingerprint, and
    every fixed-sample source array to exist on disk so the bundle can
    be reassembled deterministically.
    """
    summary_path = row_dir / ROW_SUMMARY_NAME
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        return None
    if summary.get("contract_fingerprint") != expected_fingerprint:
        return None
    if (
        expected_execution_path is not None
        and summary.get("execution_path") != expected_execution_path
    ):
        return None
    status = summary.get("row_status")
    if status not in ("completed", "blocked"):
        return None
    sample_arrays: Dict[int, Dict[str, np.ndarray]] = {}
    if status == "completed":
        loaded = _load_saved_row_arrays(
            source_arrays_dir=source_arrays_dir,
            row_id=row.row_id,
            fixed_sample_ids=fixed_sample_ids,
        )
        if loaded is None:
            return None
        sample_arrays = loaded
    rm = _row_metrics_from_summary(
        summary,
        default_paper_label=row.visible_label,
        default_arch=row.model,
    )
    return rm, dict(summary), sample_arrays


def _row_replay_argv(parent_argv: List[str], row_id: str) -> List[str]:
    """Build a parser-valid replay argv that scopes the run to ``row_id``.

    The parent argv may already include a ``--rows`` selector (e.g. for
    an ablation that excludes the classical row). Replace it with the
    single-row form so the per-row replay command, when executed, only
    re-runs that row. If the parent did not pass ``--rows``, append it.
    """
    out: List[str] = []
    skip_next = False
    saw_rows = False
    for token in parent_argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--rows":
            saw_rows = True
            skip_next = True
            out.extend(["--rows", row_id])
            continue
        if token.startswith("--rows="):
            saw_rows = True
            out.append(f"--rows={row_id}")
            continue
        out.append(token)
    if not saw_rows:
        out.extend(["--rows", row_id])
    return out


def _write_row_invocation_artifacts(
    *,
    row_dir: Path,
    row: "RowConfig",
    contract: "TrainingContract",
    fixed_sample_ids: List[int],
    in_channels: int,
    device: torch.device,
    classical_backend: ClassicalBackendInfo,
    contract_fingerprint: str,
    parent_argv: List[str],
    backlog_item: Optional[str] = None,
    extra_attempts: Optional[List[Dict[str, str]]] = None,
) -> None:
    """Persist per-row invocation provenance into the row directory.

    Writes ``invocation.json`` and ``invocation.sh`` under ``row_dir`` so
    each row carries a stand-alone provenance record (parent argv,
    parsed contract, runtime/hardware metadata, and any narrow-fix
    attempts that were made for that row).
    """
    row_argv = _row_replay_argv(list(parent_argv), row.row_id)
    parsed = {
        "row_id": row.row_id,
        "model": row.model,
        "training": row.training,
        "input_mode": row.input_mode,
        "dataset_id": row.dataset_id,
        "operator_version": row.operator_version,
        "in_channels": int(in_channels),
        "fixed_sample_ids": [int(i) for i in fixed_sample_ids],
        "training_contract": contract.as_dict(),
    }
    extras: Dict[str, Any] = {
        "backlog_item": backlog_item or BACKLOG_ITEM,
        "row": row.to_dict(),
        "runtime": _row_runtime_block(device),
        "runtime_provenance": capture_runtime_provenance(),
        "classical_backend": {
            "name": classical_backend.name,
            "reason": classical_backend.reason,
            "claim_boundary": classical_backend.claim_boundary,
        },
        "contract_fingerprint": contract_fingerprint,
    }
    if extra_attempts:
        extras["narrow_fix_attempts"] = list(extra_attempts)
    write_invocation_artifacts(
        output_dir=row_dir,
        script_path=SCRIPT_PATH,
        argv=row_argv,
        parsed_args=parsed,
        extra=extras,
    )


def _build_preflight_manifest(
    *,
    output_root: Path,
    authority: DatasetAuthority,
    rows: List[RowConfig],
    fixed_sample_ids: List[int],
    contract: TrainingContract,
    fixed_sample_seed: int,
    classical_backend: ClassicalBackendInfo,
    in_channels: int,
    notes: Optional[Dict[str, Any]] = None,
    backlog_item: Optional[str] = None,
    next_backlog_item: Optional[str] = None,
    claim_boundary: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": "brdt_preflight_v1",
        "backlog_item": backlog_item or BACKLOG_ITEM,
        "next_backlog_item": next_backlog_item or NEXT_BACKLOG_ITEM,
        "claim_boundary": claim_boundary or "decision_support_preflight_only",
        "output_root": str(output_root),
        "dataset": {
            "dataset_id": authority.dataset_id,
            "tier": authority.raw_manifest["dataset_identity"].get("tier"),
            "manifest_path": str(authority.manifest_path),
            "split_counts": authority.raw_manifest["split"]["counts"],
            "normalization": authority.normalization.as_dict(),
            "claim_boundary": authority.raw_manifest.get("claim_boundary"),
        },
        "operator": {
            "validation_artifact": authority.raw_manifest.get("operator", {}).get(
                "validation_artifact"
            ),
            "validation_report": authority.raw_manifest.get("operator", {}).get(
                "validation_report"
            ),
            "geometry": {
                "grid_size": dc.LOCKED_GRID_SIZE,
                "detector_size": dc.LOCKED_DETECTOR_SIZE,
                "angle_count": dc.LOCKED_ANGLE_COUNT,
                "wavelength_px": dc.LOCKED_WAVELENGTH_PX,
                "medium_ri": dc.LOCKED_MEDIUM_RI,
                "mode": dc.LOCKED_OPERATOR_MODE,
                "normalize": dc.LOCKED_NORMALIZE,
            },
        },
        "input_contract": {
            "input_mode": "born_init_image",
            "in_channels": int(in_channels),
            "classical_backend": {
                "name": classical_backend.name,
                "reason": classical_backend.reason,
                "claim_boundary": classical_backend.claim_boundary,
            },
        },
        "training_contract": contract.as_dict(),
        "metric_schema": {
            "version": metrics_mod.METRIC_SCHEMA_VERSION,
            "blocking": {
                "image_space_physical_q": list(metrics_mod.IMAGE_METRICS),
                "measurement_space": list(metrics_mod.MEASUREMENT_METRICS),
            },
            "supporting": list(metrics_mod.SUPPORTING_METRICS),
            "runtime_fields": list(metrics_mod.RUNTIME_FIELDS),
        },
        "fixed_sample_seed": int(fixed_sample_seed),
        "fixed_sample_ids": [int(i) for i in fixed_sample_ids],
        "rows": [r.to_dict() for r in rows],
        "environment": _runtime_environment_block(),
    }
    if notes:
        payload["notes"] = dict(notes)
    return payload


def _write_manifest(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def _resolve_baseline_lineage(
    *,
    baseline_root: Optional[Path],
    objective_preset: str,
    selected_row_ids: List[str],
) -> Dict[str, Any]:
    """Build the append-only lineage block linking this run to a prior bundle.

    The returned payload always names the ablation's own objective preset
    and selected rows. When ``baseline_root`` is supplied and the prior
    bundle exists, the lineage records the baseline's manifest/metrics
    paths and source root so reviewers can cross-check the comparison.
    """
    payload: Dict[str, Any] = {
        "claim_boundary": "decision_support_append_only",
        "ablation_objective_preset": objective_preset,
        "selected_row_ids": list(selected_row_ids),
        "baseline_root": None,
        "baseline_preflight_manifest": None,
        "baseline_metrics_json": None,
        "baseline_present": False,
    }
    if baseline_root is None:
        return payload
    base = Path(baseline_root).resolve()
    payload["baseline_root"] = str(base)
    manifest_path = base / PREFLIGHT_MANIFEST_NAME
    metrics_path = base / "metrics.json"
    payload["baseline_preflight_manifest"] = str(manifest_path)
    payload["baseline_metrics_json"] = str(metrics_path)
    payload["baseline_present"] = manifest_path.exists() and metrics_path.exists()
    return payload


def run_preflight(
    *,
    manifest_path: Path,
    output_root: Path,
    contract: TrainingContract,
    hybrid_label: str,
    fixed_sample_count: int,
    fixed_sample_seed: int,
    in_channels: int,
    device_choice: str,
    dry_run: bool,
    parent_argv: Optional[List[str]] = None,
    objective_preset: str = "supervised_plus_born",
    selected_row_ids: Optional[List[str]] = None,
    baseline_root: Optional[Path] = None,
) -> Dict[str, Any]:
    if objective_preset not in OBJECTIVE_PRESETS:
        raise ValueError(
            f"unknown objective preset {objective_preset!r}; allowed: {OBJECTIVE_PRESETS}"
        )
    resolved_weights, neural_training_label = resolve_objective_preset(objective_preset)
    # Replace contract loss weights with the preset-resolved values so the
    # manifest, fingerprints, and lightning module all observe one objective.
    contract = TrainingContract(
        epochs=contract.epochs,
        batch_size=contract.batch_size,
        learning_rate=contract.learning_rate,
        optimizer=contract.optimizer,
        loss_weights=resolved_weights,
        seed=contract.seed,
    )
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows_dir = output_root / "rows"
    visuals_dir = output_root / "visuals"
    source_arrays_dir = output_root / "figures" / "source_arrays"
    parent_argv = list(parent_argv) if parent_argv is not None else []

    with writer_lock(output_root):
        authority = load_dataset_authority(manifest_path)
        assert_decision_support_manifest(authority.raw_manifest)
        rows = resolve_row_roster(
            manifest_path=manifest_path,
            hybrid_label=hybrid_label,
            neural_training_label=neural_training_label,
            selected_row_ids=selected_row_ids,
        )
        fixed_sample_ids = choose_fixed_sample_ids(
            manifest_path,
            count=fixed_sample_count,
            seed=fixed_sample_seed,
        )

        backend, narrow_fix_attempts = attempt_classical_backend_with_fix()
        input_backend = _born_init_backend()
        inverse_config = _model_based_inverse_config(authority)
        classical_inverse_auth = classical_inverse_authorization(
            authority.raw_manifest,
            manifest_path=authority.manifest_path,
        )
        device = _select_device(device_choice)

        operator_pointer = (
            authority.raw_manifest.get("operator", {}).get("validation_artifact")
            or authority.raw_manifest.get("operator", {}).get("validation_report")
            or "unspecified"
        )
        per_row_fingerprints: Dict[str, str] = {}
        expected_execution_paths: Dict[str, str] = {}
        for row in rows:
            if row.row_id == "classical_born_backprop":
                backend_name = MODEL_BASED_INVERSE_EXECUTION_PATH
                execution_path = MODEL_BASED_INVERSE_EXECUTION_PATH
                solver_version = MODEL_BASED_INVERSE_VERSION
                solver_config = inverse_config.as_dict()
            else:
                backend_name = f"{backend.name}:{classical_inverse_auth.get('status')}"
                execution_path = "neural_train_eval"
                solver_version = None
                solver_config = None
            expected_execution_paths[row.row_id] = execution_path
            per_row_fingerprints[row.row_id] = row_contract_fingerprint(
                row_id=row.row_id,
                model=row.model,
                training=row.training,
                input_mode=row.input_mode,
                dataset_id=str(authority.dataset_id),
                operator_pointer=str(operator_pointer),
                training_contract=contract.as_dict(),
                fixed_sample_ids=fixed_sample_ids,
                in_channels=in_channels,
                classical_backend_name=backend_name,
                execution_path=(
                    execution_path
                    if row.row_id == "classical_born_backprop"
                    else None
                ),
                solver_version=solver_version,
                solver_config=solver_config,
            )

        is_ablation_run = objective_preset != "supervised_plus_born"
        manifest_backlog_item: Optional[str] = None
        manifest_next_backlog_item: Optional[str] = None
        manifest_claim_boundary: Optional[str] = None
        baseline_lineage: Optional[Dict[str, Any]] = None
        if objective_preset == "relative_physics_only":
            manifest_backlog_item = PHYSICS_ONLY_BACKLOG_ITEM
            manifest_next_backlog_item = "n/a"
            manifest_claim_boundary = "decision_support_append_only"
        resolved_claim_boundary = (
            manifest_claim_boundary or "decision_support_preflight_only"
        )
        manifest_notes: Dict[str, Any] = {
            "classical_narrow_fix_attempts": narrow_fix_attempts,
            "classical_inverse_authorization": classical_inverse_auth,
            "model_based_inverse": {
                "execution_path": MODEL_BASED_INVERSE_EXECUTION_PATH,
                "version": MODEL_BASED_INVERSE_VERSION,
                "config": inverse_config.as_dict(),
            },
            "row_contract_fingerprints": per_row_fingerprints,
        }
        if is_ablation_run:
            baseline_lineage = _resolve_baseline_lineage(
                baseline_root=baseline_root,
                objective_preset=objective_preset,
                selected_row_ids=[r.row_id for r in rows],
            )
            manifest_notes["objective_preset"] = objective_preset
            manifest_notes["selected_row_ids"] = [r.row_id for r in rows]
            manifest_notes["baseline_lineage"] = baseline_lineage
        preflight_manifest = _build_preflight_manifest(
            output_root=output_root,
            authority=authority,
            rows=rows,
            fixed_sample_ids=fixed_sample_ids,
            contract=contract,
            fixed_sample_seed=fixed_sample_seed,
            classical_backend=input_backend,
            in_channels=in_channels,
            backlog_item=manifest_backlog_item,
            next_backlog_item=manifest_next_backlog_item,
            claim_boundary=manifest_claim_boundary,
            notes=manifest_notes,
        )

        manifest_path_out = output_root / PREFLIGHT_MANIFEST_NAME
        _write_manifest(preflight_manifest, manifest_path_out)

        if dry_run:
            # Dry-run still emits a metric schema so downstream consumers see
            # the bundle layout even before live execution.
            metrics_mod.write_metric_schema(
                output_root / "metric_schema.json",
                claim_boundary=resolved_claim_boundary,
            )
            return {
                "dry_run": True,
                "preflight_manifest_path": str(manifest_path_out),
            }

        operator = _build_operator(device)
        rows_dir.mkdir(parents=True, exist_ok=True)
        visuals_dir.mkdir(parents=True, exist_ok=True)
        source_arrays_dir.mkdir(parents=True, exist_ok=True)

        row_metrics: List[metrics_mod.RowMetrics] = []
        row_status_updates: Dict[str, Dict[str, Any]] = {}
        fixed_q_pred_by_row: Dict[int, Dict[str, np.ndarray]] = {
            sid: {} for sid in fixed_sample_ids
        }
        fixed_sino_pred_by_row: Dict[int, Dict[str, np.ndarray]] = {
            sid: {} for sid in fixed_sample_ids
        }
        fixed_targets: Dict[int, Dict[str, np.ndarray]] = {}
        resumed_rows: List[str] = []

        def _record_completed_arrays(
            row_id: str, sample_arrays: Mapping[int, Mapping[str, np.ndarray]]
        ) -> None:
            for sid, arrays in sample_arrays.items():
                fixed_q_pred_by_row[sid][row_id] = arrays["q_pred"]
                fixed_sino_pred_by_row[sid][row_id] = arrays["sino_pred"]
                fixed_targets.setdefault(
                    sid,
                    {
                        "q_target": arrays["q_target"],
                        "sino_obs": arrays["sino_obs"],
                    },
                )

        for row in rows:
            row_dir = rows_dir / row.row_id
            row_dir.mkdir(parents=True, exist_ok=True)
            row_payload: Dict[str, Any] = {"row_id": row.row_id}
            row_fp = per_row_fingerprints[row.row_id]
            row_backend = (
                ClassicalBackendInfo(
                    name=MODEL_BASED_INVERSE_EXECUTION_PATH,
                    reason="direct_optimization_through_locked_forward_operator",
                    claim_boundary="model_based_inverse",
                )
                if row.row_id == "classical_born_backprop"
                else input_backend
            )

            # ---- Resume protocol: skip if a matching prior run exists ----
            resumed = _maybe_resume_row(
                row=row,
                row_dir=row_dir,
                source_arrays_dir=source_arrays_dir,
                fixed_sample_ids=fixed_sample_ids,
                expected_fingerprint=row_fp,
                expected_execution_path=expected_execution_paths[row.row_id],
            )
            if resumed is not None:
                cached_metrics, cached_payload, cached_arrays = resumed
                row_metrics.append(cached_metrics)
                if cached_metrics.row_status == "completed":
                    _record_completed_arrays(row.row_id, cached_arrays)
                row_status_updates[row.row_id] = cached_payload
                resumed_rows.append(row.row_id)
                # Refresh per-row invocation provenance to record this resume pass.
                _write_row_invocation_artifacts(
                    row_dir=row_dir,
                    row=row,
                    contract=contract,
                    fixed_sample_ids=fixed_sample_ids,
                    in_channels=in_channels,
                    device=device,
                    classical_backend=row_backend,
                    contract_fingerprint=row_fp,
                    parent_argv=parent_argv,
                    backlog_item=manifest_backlog_item,
                    extra_attempts=(
                        narrow_fix_attempts
                        if row.row_id == "classical_born_backprop"
                        else None
                    ),
                )
                continue

            # ---- Per-row invocation provenance for this fresh execution ----
            _write_row_invocation_artifacts(
                row_dir=row_dir,
                row=row,
                contract=contract,
                fixed_sample_ids=fixed_sample_ids,
                in_channels=in_channels,
                device=device,
                classical_backend=row_backend,
                contract_fingerprint=row_fp,
                parent_argv=parent_argv,
                backlog_item=manifest_backlog_item,
                extra_attempts=(
                    narrow_fix_attempts
                    if row.row_id == "classical_born_backprop"
                    else None
                ),
            )

            if row.row_id == "classical_born_backprop":
                # Classical row: no learned parameters. Optimize physical q
                # directly through the locked local forward operator.
                eval_started = time.perf_counter()
                (
                    image_metrics,
                    meas_metrics,
                    sample_arrays,
                    solver_summary,
                ) = _evaluate_model_based_inverse_split(
                    authority=authority,
                    operator=operator,
                    device=device,
                    batch_size=int(contract.batch_size),
                    fixed_sample_ids=fixed_sample_ids,
                    config=inverse_config,
                )
                eval_seconds = time.perf_counter() - eval_started
                _record_completed_arrays(row.row_id, sample_arrays)
                _save_fixed_sample_arrays(
                    source_arrays_dir=source_arrays_dir,
                    sample_arrays=sample_arrays,
                    row_id=row.row_id,
                )
                runtime_block = metrics_mod.collect_runtime_metadata(
                    device=str(device),
                    device_name=_device_name(device),
                    epochs=0,
                    batch_size=int(contract.batch_size),
                    learning_rate=float(contract.learning_rate),
                    parameter_count=0,
                    wall_time_train_s=0.0,
                    wall_time_eval_s=eval_seconds,
                    row_status="completed",
                    extras={
                        "solver": solver_summary,
                        "odtbrain_diagnostic": {
                            "backend_detection": {
                                "name": backend.name,
                                "reason": backend.reason,
                                "claim_boundary": backend.claim_boundary,
                            },
                            "narrow_fix_attempts": list(narrow_fix_attempts),
                            "inverse_authorization": dict(classical_inverse_auth),
                        },
                    },
                )
                row_metrics.append(
                    metrics_mod.RowMetrics(
                        row_id=row.row_id,
                        paper_label=row.visible_label,
                        architecture=row.model,
                        row_status="completed",
                        image={
                            k: v
                            for k, v in image_metrics.items()
                            if k in metrics_mod.IMAGE_METRICS
                        },
                        measurement=meas_metrics,
                        supporting={
                            k: v
                            for k, v in image_metrics.items()
                            if k in metrics_mod.SUPPORTING_METRICS
                        },
                        runtime=runtime_block,
                    )
                )
                row_payload["row_status"] = "completed"
                row_payload["paper_label"] = row.visible_label
                row_payload["architecture"] = row.model
                row_payload["execution_path"] = MODEL_BASED_INVERSE_EXECUTION_PATH
                row_payload["solver_version"] = MODEL_BASED_INVERSE_VERSION
                row_payload["solver_config"] = inverse_config.as_dict()
                row_payload["solver_summary"] = solver_summary
                row_payload["image_metrics"] = image_metrics
                row_payload["measurement_metrics"] = meas_metrics
                row_payload["runtime"] = runtime_block
                row_payload["contract_fingerprint"] = row_fp
                row_payload["narrow_fix_attempts"] = list(narrow_fix_attempts)
                row_payload["classical_inverse_authorization"] = dict(
                    classical_inverse_auth
                )
                row_status_updates[row.row_id] = row_payload
                (row_dir / ROW_SUMMARY_NAME).write_text(
                    json.dumps(row_payload, indent=2, sort_keys=True) + "\n"
                )
                continue

            # Neural rows.
            module, runtime_meta, train_seconds = _train_neural_row(
                row=row,
                authority=authority,
                operator=operator,
                backend=input_backend,
                device=device,
                contract=contract,
                in_channels=in_channels,
                output_dir=row_dir,
            )
            if module is None:
                err = runtime_meta.get("adapter_build_error", {})
                blocker = make_blocked_row(
                    row_id=row.row_id,
                    model=row.model,
                    training=row.training,
                    dataset_id=row.dataset_id,
                    operator_version=row.operator_version,
                    blocker_reason=str(err.get("reason", "adapter_build_error")),
                    blocker_message=str(err.get("message", "adapter_build_error")),
                    paper_label=row.paper_label,
                )
                row_payload.update(blocker.to_dict())
                row_payload["contract_fingerprint"] = row_fp
                row_payload["execution_path"] = "neural_blocked"
                blocked_runtime = metrics_mod.collect_runtime_metadata(
                    device=str(device),
                    device_name=_device_name(device),
                    epochs=0,
                    batch_size=int(contract.batch_size),
                    learning_rate=float(contract.learning_rate),
                    parameter_count=0,
                    wall_time_train_s=0.0,
                    wall_time_eval_s=0.0,
                    row_status="blocked",
                )
                row_metrics.append(
                    metrics_mod.RowMetrics(
                        row_id=row.row_id,
                        paper_label=row.visible_label,
                        architecture=row.model,
                        row_status="blocked",
                        blocker_reason=blocker.blocker_reason or "",
                        blocker_message=blocker.blocker_message or "",
                        runtime=blocked_runtime,
                    )
                )
                row_payload["paper_label"] = row.visible_label
                row_payload["architecture"] = row.model
                row_payload["runtime"] = blocked_runtime
                row_status_updates[row.row_id] = row_payload
                (row_dir / ROW_SUMMARY_NAME).write_text(
                    json.dumps(row_payload, indent=2, sort_keys=True) + "\n"
                )
                continue
            eval_started = time.perf_counter()
            (
                image_metrics,
                meas_metrics,
                sample_arrays,
                output_dynamic_range,
            ) = _evaluate_split(
                module=module,
                authority=authority,
                operator=operator,
                backend=input_backend,
                device=device,
                in_channels=in_channels,
                classical_only=False,
                fixed_sample_ids=fixed_sample_ids,
                out_dir=row_dir,
            )
            eval_seconds = time.perf_counter() - eval_started
            _record_completed_arrays(row.row_id, sample_arrays)
            _save_fixed_sample_arrays(
                source_arrays_dir=source_arrays_dir,
                sample_arrays=sample_arrays,
                row_id=row.row_id,
            )
            runtime_block = metrics_mod.collect_runtime_metadata(
                device=str(device),
                device_name=_device_name(device),
                epochs=int(contract.epochs),
                batch_size=int(contract.batch_size),
                learning_rate=float(contract.learning_rate),
                parameter_count=int(runtime_meta.get("parameter_count", 0)),
                wall_time_train_s=float(train_seconds),
                wall_time_eval_s=float(eval_seconds),
                row_status="completed",
                extras={
                    "train_steps": runtime_meta.get("train_steps"),
                    "final_loss_breakdown": runtime_meta.get(
                        "final_loss_breakdown"
                    ),
                    "model_state_path": runtime_meta.get("model_state_path"),
                    "history_json_path": runtime_meta.get("history_json_path"),
                    "history_csv_path": runtime_meta.get("history_csv_path"),
                    "history_length": runtime_meta.get("history_length"),
                    "scheduler": runtime_meta.get("scheduler"),
                    "output_dynamic_range": output_dynamic_range,
                },
            )
            row_metrics.append(
                metrics_mod.RowMetrics(
                    row_id=row.row_id,
                    paper_label=row.visible_label,
                    architecture=row.model,
                    row_status="completed",
                    image={
                        k: v
                        for k, v in image_metrics.items()
                        if k in metrics_mod.IMAGE_METRICS
                    },
                    measurement=meas_metrics,
                    supporting={
                        k: v
                        for k, v in image_metrics.items()
                        if k in metrics_mod.SUPPORTING_METRICS
                    },
                    runtime=runtime_block,
                )
            )
            row_payload["row_status"] = "completed"
            row_payload["paper_label"] = row.visible_label
            row_payload["architecture"] = row.model
            row_payload["execution_path"] = "neural_train_eval"
            row_payload["parameter_count"] = int(
                runtime_meta.get("parameter_count", 0)
            )
            row_payload["image_metrics"] = image_metrics
            row_payload["measurement_metrics"] = meas_metrics
            row_payload["wall_time_train_s"] = float(train_seconds)
            row_payload["wall_time_eval_s"] = float(eval_seconds)
            row_payload["model_state_path"] = runtime_meta.get("model_state_path")
            row_payload["history_json_path"] = runtime_meta.get("history_json_path")
            row_payload["history_csv_path"] = runtime_meta.get("history_csv_path")
            row_payload["history_length"] = runtime_meta.get("history_length")
            row_payload["scheduler"] = runtime_meta.get("scheduler")
            row_payload["runtime"] = runtime_block
            row_payload["contract_fingerprint"] = row_fp
            row_payload["output_dynamic_range"] = output_dynamic_range
            row_status_updates[row.row_id] = row_payload
            (row_dir / ROW_SUMMARY_NAME).write_text(
                json.dumps(row_payload, indent=2, sort_keys=True) + "\n"
            )

        # ---- Aggregate metrics outputs ----
        metrics_mod.write_metric_schema(
            output_root / "metric_schema.json",
            claim_boundary=resolved_claim_boundary,
        )
        metrics_mod.write_metrics_json(
            output_root / "metrics.json",
            row_metrics,
            claim_boundary=resolved_claim_boundary,
        )
        metrics_mod.write_metrics_csv(output_root / "metrics.csv", row_metrics)

        # ---- Visuals ----
        visual_entries: List[visuals_mod.VisualBundleEntry] = []
        figure_paths: List[str] = []
        for sid in fixed_sample_ids:
            per_row_q = fixed_q_pred_by_row.get(sid, {})
            per_row_sino = fixed_sino_pred_by_row.get(sid, {})
            targets = fixed_targets.get(sid)
            if not per_row_q or targets is None:
                continue
            target_array_path = source_arrays_dir / f"sample_{sid:04d}_q_target.npy"
            sino_obs_array_path = source_arrays_dir / f"sample_{sid:04d}_sino_obs.npy"
            np.save(target_array_path, targets["q_target"])
            np.save(sino_obs_array_path, targets["sino_obs"])
            for row_id, q_pred in per_row_q.items():
                pred_path = source_arrays_dir / f"sample_{sid:04d}_{row_id}_q_pred.npy"
                sino_pred_path = (
                    source_arrays_dir / f"sample_{sid:04d}_{row_id}_sino_pred.npy"
                )
                np.save(pred_path, q_pred)
                np.save(
                    sino_pred_path,
                    per_row_sino.get(row_id, np.zeros_like(targets["sino_obs"])),
                )
                visual_entries.append(
                    visuals_mod.VisualBundleEntry(
                        sample_id=int(sid),
                        row_id=row_id,
                        pred_array=str(pred_path.relative_to(output_root)),
                        target_array=str(target_array_path.relative_to(output_root)),
                        sino_pred_array=str(sino_pred_path.relative_to(output_root)),
                        sino_obs_array=str(sino_obs_array_path.relative_to(output_root)),
                    )
                )

        if fixed_sample_ids:
            first_id = fixed_sample_ids[0]
            per_row_q = fixed_q_pred_by_row.get(first_id, {})
            per_row_sino = fixed_sino_pred_by_row.get(first_id, {})
            targets = fixed_targets.get(first_id)
            if per_row_q and targets is not None:
                compare_path = visuals_dir / "brdt_compare_q.png"
                error_path = visuals_dir / "brdt_error_q.png"
                sino_path = visuals_dir / "brdt_sinogram_residual.png"
                visuals_mod.render_compare_q(
                    preds_by_row=per_row_q,
                    target=targets["q_target"],
                    out_path=compare_path,
                    sample_id=int(first_id),
                )
                visuals_mod.render_error_q(
                    preds_by_row=per_row_q,
                    target=targets["q_target"],
                    out_path=error_path,
                    sample_id=int(first_id),
                )
                visuals_mod.render_sinogram_residual(
                    sino_obs=targets["sino_obs"],
                    sino_preds_by_row=per_row_sino,
                    out_path=sino_path,
                    sample_id=int(first_id),
                )
                figure_paths = [
                    str(compare_path.relative_to(output_root)),
                    str(error_path.relative_to(output_root)),
                    str(sino_path.relative_to(output_root)),
                ]

        visuals_mod.write_visual_manifest(
            output_root / "visual_manifest.json",
            figures=figure_paths,
            entries=visual_entries,
        )

        # ---- Update preflight manifest with final row statuses ----
        preflight_manifest["rows"] = [
            {**r.to_dict(), **row_status_updates.get(r.row_id, {})}
            for r in rows
        ]
        preflight_manifest["row_metrics"] = [r.to_dict() for r in row_metrics]
        preflight_manifest["bundle_artifacts"] = {
            "metrics_json": "metrics.json",
            "metrics_csv": "metrics.csv",
            "metric_schema": "metric_schema.json",
            "visual_manifest": "visual_manifest.json",
            "visuals_dir": "visuals/",
            "source_arrays_dir": "figures/source_arrays/",
            "rows_dir": "rows/",
        }
        preflight_manifest["dependency_status"] = {
            "odtbrain": backend.name == "odtbrain",
            "neuralop": _neuralop_available(),
        }
        preflight_manifest["resumed_rows"] = list(resumed_rows)

        # ---- Comparison-to-baseline emission ----
        # For ablation runs whose baseline bundle is on disk, the
        # comparison artifacts are mandatory contract outputs and any
        # emission failure must surface as a run failure rather than a
        # quiet note.
        comparison_paths: Dict[str, str] = {}
        if (
            baseline_lineage is not None
            and baseline_lineage.get("baseline_present")
            and baseline_lineage.get("baseline_metrics_json")
        ):
            json_path, csv_path = comparison_mod.emit_comparison_artifacts(
                baseline_metrics_path=Path(
                    str(baseline_lineage["baseline_metrics_json"])
                ),
                ablation_metrics_path=output_root / "metrics.json",
                output_root=output_root,
                selected_row_ids=[r.row_id for r in rows],
                baseline_root=str(baseline_lineage.get("baseline_root") or ""),
                ablation_root=str(output_root),
                ablation_objective_preset=objective_preset,
            )
            comparison_paths = {
                "comparison_json": str(json_path.relative_to(output_root)),
                "comparison_csv": str(csv_path.relative_to(output_root)),
            }
            preflight_manifest["bundle_artifacts"].update(
                {
                    "comparison_to_supervised_plus_born_json": comparison_paths[
                        "comparison_json"
                    ],
                    "comparison_to_supervised_plus_born_csv": comparison_paths[
                        "comparison_csv"
                    ],
                }
            )

        _write_manifest(preflight_manifest, manifest_path_out)

        result: Dict[str, Any] = {
            "preflight_manifest_path": str(manifest_path_out),
            "metrics_json_path": str(output_root / "metrics.json"),
            "metrics_csv_path": str(output_root / "metrics.csv"),
            "metric_schema_path": str(output_root / "metric_schema.json"),
            "visual_manifest_path": str(output_root / "visual_manifest.json"),
            "rows": [r.to_dict() for r in row_metrics],
            "resumed_rows": list(resumed_rows),
        }
        if comparison_paths:
            result["comparison_paths"] = {
                k: str(output_root / v) for k, v in comparison_paths.items()
            }
        return result


def _neuralop_available() -> bool:
    try:  # pragma: no cover
        import neuralop  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brdt_run_preflight",
        description=(
            "BRDT four-row decision-support preflight. Runs the bounded "
            "row roster under one shared dataset/operator/input/split/loss "
            "contract and emits a machine-readable bundle."
        ),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--hybrid-label",
        default="hybrid_resnet",
        choices=list(HYBRID_FAMILY_ROW_IDS),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--in-channels", type=int, default=1, choices=[1, 2])
    parser.add_argument("--fixed-sample-count", type=int, default=4)
    parser.add_argument("--fixed-sample-seed", type=int, default=17)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Training seed: seeds NumPy/Torch (CPU+CUDA) before each neural "
            "row's model init and the DataLoader shuffle generator so the "
            "bundle is regenerable under a fixed contract."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest, write preflight_manifest.json + metric_schema.json, and exit.",
    )
    parser.add_argument(
        "--objective-preset",
        default="supervised_plus_born",
        choices=list(OBJECTIVE_PRESETS),
        help=(
            "Neural-row training objective preset. "
            "'supervised_plus_born' (default) reproduces the four-row "
            "preflight contract; 'relative_physics_only' resolves to "
            "image=0, physics=0, relative_physics=1, tv=0, positivity=0 "
            "for the bounded physics-only objective ablation."
        ),
    )
    parser.add_argument(
        "--rows",
        default=None,
        help=(
            "Optional comma-separated row_ids to execute (default: full "
            "four-row roster). Use to scope the run to neural rows only "
            "for the physics-only ablation, e.g. "
            "'unet,fno_vanilla,hybrid_resnet'."
        ),
    )
    parser.add_argument(
        "--baseline-root",
        default=None,
        type=Path,
        help=(
            "Optional path to a completed BRDT preflight bundle. When "
            "provided, the run records baseline lineage in the manifest "
            "and emits comparison_to_supervised_plus_born.{json,csv} "
            "comparing per-row metrics against that baseline."
        ),
    )
    return parser


def _resolve_backlog_item_for_invocation(objective_preset: str) -> str:
    """Resolve the backlog identity recorded in invocation provenance.

    Keeps the default supervised-plus-Born path on the original
    ``BACKLOG_ITEM`` while routing append-only objective ablations to
    their own backlog identity, so emitted ``invocation.json`` files
    truthfully name the backlog item that owns the artifact root.
    """
    if objective_preset == "relative_physics_only":
        return PHYSICS_ONLY_BACKLOG_ITEM
    return BACKLOG_ITEM


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    parent_argv = sys.argv[1:] if argv is None else list(argv)
    invocation_backlog_item = _resolve_backlog_item_for_invocation(
        str(args.objective_preset)
    )
    write_invocation_artifacts(
        output_dir=output_root,
        script_path=SCRIPT_PATH,
        argv=parent_argv,
        parsed_args=vars(args),
        extra={
            "backlog_item": invocation_backlog_item,
            "runtime_provenance": capture_runtime_provenance(),
        },
    )
    contract = TrainingContract(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
    )
    selected_row_ids: Optional[List[str]] = None
    if args.rows:
        selected_row_ids = [s.strip() for s in str(args.rows).split(",") if s.strip()]
    try:
        result = run_preflight(
            manifest_path=Path(args.manifest),
            output_root=output_root,
            contract=contract,
            hybrid_label=args.hybrid_label,
            fixed_sample_count=int(args.fixed_sample_count),
            fixed_sample_seed=int(args.fixed_sample_seed),
            in_channels=int(args.in_channels),
            device_choice=str(args.device),
            dry_run=bool(args.dry_run),
            parent_argv=parent_argv,
            objective_preset=str(args.objective_preset),
            selected_row_ids=selected_row_ids,
            baseline_root=Path(args.baseline_root) if args.baseline_root else None,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
