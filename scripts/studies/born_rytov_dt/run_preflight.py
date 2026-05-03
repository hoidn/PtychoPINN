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
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ptycho_torch.physics import BornRytovForward2D
from scripts.studies.born_rytov_dt import dataset_contract as dc
from scripts.studies.born_rytov_dt import preflight_metrics as metrics_mod
from scripts.studies.born_rytov_dt import preflight_visuals as visuals_mod
from scripts.studies.born_rytov_dt.classical import (
    ClassicalBackendInfo,
    derive_born_init_image,
    detect_classical_backend,
)
from scripts.studies.born_rytov_dt.data import (
    BRDTSmokeSplit,
    DatasetAuthority,
    brdt_collate,
    load_dataset_authority,
)
from scripts.studies.born_rytov_dt.lightning_module import BRDTTrainingModule
from scripts.studies.born_rytov_dt.models import (
    AdapterBuildError,
    build_neural_adapter,
)
from scripts.studies.born_rytov_dt.run_config import (
    HYBRID_FAMILY_ROW_IDS,
    LossWeights,
    RowConfig,
    default_row_roster,
    make_blocked_row,
)
from scripts.studies.invocation_logging import (
    capture_runtime_provenance,
    write_invocation_artifacts,
)


SCRIPT_PATH = "scripts/studies/born_rytov_dt/run_preflight.py"
BACKLOG_ITEM = "2026-04-29-brdt-four-row-preflight"
NEXT_BACKLOG_ITEM = "2026-04-29-brdt-preflight-summary-promotion-decision"
PREFLIGHT_MANIFEST_NAME = "preflight_manifest.json"


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
) -> List[RowConfig]:
    """Return the locked four-row roster against a decision-support manifest."""
    manifest = json.loads(Path(manifest_path).read_text())
    assert_decision_support_manifest(manifest)
    operator_block = manifest.get("operator", {})
    operator_version = (
        operator_block.get("validation_artifact")
        or operator_block.get("validation_report")
        or "unspecified"
    )
    dataset_id = manifest["dataset_identity"]["name"]
    return default_row_roster(
        dataset_id=str(dataset_id),
        operator_version=str(operator_version),
        hybrid_label=hybrid_label,
    )


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

    def as_dict(self) -> Dict[str, Any]:
        return {
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "learning_rate": float(self.learning_rate),
            "optimizer": self.optimizer,
            "loss_weights": self.loss_weights.as_dict(),
        }


def _prepare_input(
    sinogram: torch.Tensor,
    *,
    operator: BornRytovForward2D,
    backend: ClassicalBackendInfo,
    in_channels: int,
) -> torch.Tensor:
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
    try:
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
    loader = DataLoader(
        train_split,
        batch_size=int(contract.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=brdt_collate,
    )
    optim = torch.optim.Adam(module.parameters(), lr=float(contract.learning_rate))
    started = time.perf_counter()
    last_breakdown: Dict[str, float] = {}
    train_steps = 0
    module.train()
    for _epoch in range(int(contract.epochs)):
        for batch in loader:
            sino = batch["sinogram"].to(device)
            x = _prepare_input(
                sino,
                operator=operator,
                backend=backend,
                in_channels=in_channels,
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
    elapsed = time.perf_counter() - started

    info = adapter.info()
    runtime_meta = {
        "parameter_count": int(info.parameter_count),
        "train_steps": int(train_steps),
        "final_loss_breakdown": last_breakdown,
    }
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
) -> Tuple[Dict[str, float], Dict[str, float], Dict[int, Dict[str, np.ndarray]]]:
    """Run an evaluation pass on the test split.

    Returns (image_metrics, measurement_metrics, fixed_sample_arrays).
    ``fixed_sample_arrays[sample_id]`` is a dict with keys ``q_pred``,
    ``q_target``, ``sino_pred``, ``sino_obs``.
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
    return image_metrics, meas_metrics, sample_arrays


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
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": "brdt_preflight_v1",
        "backlog_item": BACKLOG_ITEM,
        "next_backlog_item": NEXT_BACKLOG_ITEM,
        "claim_boundary": "decision_support_preflight_only",
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
) -> Dict[str, Any]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows_dir = output_root / "rows"
    visuals_dir = output_root / "visuals"
    source_arrays_dir = output_root / "figures" / "source_arrays"

    authority = load_dataset_authority(manifest_path)
    assert_decision_support_manifest(authority.raw_manifest)
    rows = resolve_row_roster(manifest_path=manifest_path, hybrid_label=hybrid_label)
    fixed_sample_ids = choose_fixed_sample_ids(
        manifest_path,
        count=fixed_sample_count,
        seed=fixed_sample_seed,
    )

    backend = detect_classical_backend()
    device = _select_device(device_choice)

    preflight_manifest = _build_preflight_manifest(
        output_root=output_root,
        authority=authority,
        rows=rows,
        fixed_sample_ids=fixed_sample_ids,
        contract=contract,
        fixed_sample_seed=fixed_sample_seed,
        classical_backend=backend,
        in_channels=in_channels,
    )

    manifest_path_out = output_root / PREFLIGHT_MANIFEST_NAME
    _write_manifest(preflight_manifest, manifest_path_out)

    if dry_run:
        # Dry-run still emits a metric schema so downstream consumers see
        # the bundle layout even before live execution.
        metrics_mod.write_metric_schema(output_root / "metric_schema.json")
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

    for row in rows:
        row_dir = rows_dir / row.row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        row_payload: Dict[str, Any] = {"row_id": row.row_id}
        if row.row_id == "classical_born_backprop":
            # ODTbrain authoritative; if unavailable, mark row blocked.
            if backend.name != "odtbrain":
                blocker = make_blocked_row(
                    row_id=row.row_id,
                    model=row.model,
                    training=row.training,
                    dataset_id=row.dataset_id,
                    operator_version=row.operator_version,
                    blocker_reason="odtbrain_unavailable",
                    blocker_message=(
                        "Classical Born backprop reference requires ODTbrain "
                        "for the decision-support claim boundary; only the "
                        "local-adjoint fallback is available."
                    ),
                    paper_label=row.paper_label,
                )
                row_payload.update(blocker.to_dict())
                row_status_updates[row.row_id] = row_payload
                row_metrics.append(
                    metrics_mod.RowMetrics(
                        row_id=row.row_id,
                        paper_label=row.visible_label,
                        architecture=row.model,
                        row_status="blocked",
                        blocker_reason=blocker.blocker_reason or "",
                        blocker_message=blocker.blocker_message or "",
                        runtime=metrics_mod.collect_runtime_metadata(
                            device=str(device),
                            device_name=_device_name(device),
                            epochs=0,
                            batch_size=int(contract.batch_size),
                            learning_rate=float(contract.learning_rate),
                            parameter_count=0,
                            wall_time_train_s=0.0,
                            wall_time_eval_s=0.0,
                            row_status="blocked",
                        ),
                    )
                )
                (row_dir / "row_summary.json").write_text(
                    json.dumps(row_payload, indent=2, sort_keys=True) + "\n"
                )
                continue
            # Classical row: no training; eval uses the backend init directly.
            eval_started = time.perf_counter()
            image_metrics, meas_metrics, sample_arrays = _evaluate_split(
                module=None,
                authority=authority,
                operator=operator,
                backend=backend,
                device=device,
                in_channels=in_channels,
                classical_only=True,
                fixed_sample_ids=fixed_sample_ids,
                out_dir=row_dir,
            )
            eval_seconds = time.perf_counter() - eval_started
            for sid, arrays in sample_arrays.items():
                fixed_q_pred_by_row[sid][row.row_id] = arrays["q_pred"]
                fixed_sino_pred_by_row[sid][row.row_id] = arrays["sino_pred"]
                fixed_targets.setdefault(sid, {"q_target": arrays["q_target"], "sino_obs": arrays["sino_obs"]})
            row_metrics.append(
                metrics_mod.RowMetrics(
                    row_id=row.row_id,
                    paper_label=row.visible_label,
                    architecture=row.model,
                    row_status="completed",
                    image={k: v for k, v in image_metrics.items() if k in metrics_mod.IMAGE_METRICS},
                    measurement=meas_metrics,
                    supporting={
                        k: v for k, v in image_metrics.items()
                        if k in metrics_mod.SUPPORTING_METRICS
                    },
                    runtime=metrics_mod.collect_runtime_metadata(
                        device=str(device),
                        device_name=_device_name(device),
                        epochs=0,
                        batch_size=int(contract.batch_size),
                        learning_rate=float(contract.learning_rate),
                        parameter_count=0,
                        wall_time_train_s=0.0,
                        wall_time_eval_s=eval_seconds,
                        row_status="completed",
                    ),
                )
            )
            row_payload["row_status"] = "completed"
            row_payload["execution_path"] = "classical_only"
            row_payload["image_metrics"] = image_metrics
            row_payload["measurement_metrics"] = meas_metrics
            row_status_updates[row.row_id] = row_payload
            (row_dir / "row_summary.json").write_text(
                json.dumps(row_payload, indent=2, sort_keys=True) + "\n"
            )
            continue

        # Neural rows.
        module, runtime_meta, train_seconds = _train_neural_row(
            row=row,
            authority=authority,
            operator=operator,
            backend=backend,
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
            row_status_updates[row.row_id] = row_payload
            row_metrics.append(
                metrics_mod.RowMetrics(
                    row_id=row.row_id,
                    paper_label=row.visible_label,
                    architecture=row.model,
                    row_status="blocked",
                    blocker_reason=blocker.blocker_reason or "",
                    blocker_message=blocker.blocker_message or "",
                    runtime=metrics_mod.collect_runtime_metadata(
                        device=str(device),
                        device_name=_device_name(device),
                        epochs=0,
                        batch_size=int(contract.batch_size),
                        learning_rate=float(contract.learning_rate),
                        parameter_count=0,
                        wall_time_train_s=0.0,
                        wall_time_eval_s=0.0,
                        row_status="blocked",
                    ),
                )
            )
            (row_dir / "row_summary.json").write_text(
                json.dumps(row_payload, indent=2, sort_keys=True) + "\n"
            )
            continue
        eval_started = time.perf_counter()
        image_metrics, meas_metrics, sample_arrays = _evaluate_split(
            module=module,
            authority=authority,
            operator=operator,
            backend=backend,
            device=device,
            in_channels=in_channels,
            classical_only=False,
            fixed_sample_ids=fixed_sample_ids,
            out_dir=row_dir,
        )
        eval_seconds = time.perf_counter() - eval_started
        for sid, arrays in sample_arrays.items():
            fixed_q_pred_by_row[sid][row.row_id] = arrays["q_pred"]
            fixed_sino_pred_by_row[sid][row.row_id] = arrays["sino_pred"]
            fixed_targets.setdefault(sid, {"q_target": arrays["q_target"], "sino_obs": arrays["sino_obs"]})
        row_metrics.append(
            metrics_mod.RowMetrics(
                row_id=row.row_id,
                paper_label=row.visible_label,
                architecture=row.model,
                row_status="completed",
                image={k: v for k, v in image_metrics.items() if k in metrics_mod.IMAGE_METRICS},
                measurement=meas_metrics,
                supporting={
                    k: v for k, v in image_metrics.items()
                    if k in metrics_mod.SUPPORTING_METRICS
                },
                runtime=metrics_mod.collect_runtime_metadata(
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
                        "final_loss_breakdown": runtime_meta.get("final_loss_breakdown"),
                        "model_state_path": runtime_meta.get("model_state_path"),
                    },
                ),
            )
        )
        row_payload["row_status"] = "completed"
        row_payload["execution_path"] = "neural_train_eval"
        row_payload["parameter_count"] = int(runtime_meta.get("parameter_count", 0))
        row_payload["image_metrics"] = image_metrics
        row_payload["measurement_metrics"] = meas_metrics
        row_payload["wall_time_train_s"] = float(train_seconds)
        row_payload["wall_time_eval_s"] = float(eval_seconds)
        row_payload["model_state_path"] = runtime_meta.get("model_state_path")
        row_status_updates[row.row_id] = row_payload
        (row_dir / "row_summary.json").write_text(
            json.dumps(row_payload, indent=2, sort_keys=True) + "\n"
        )

    # ---- Aggregate metrics outputs ----
    metrics_mod.write_metric_schema(output_root / "metric_schema.json")
    metrics_mod.write_metrics_json(output_root / "metrics.json", row_metrics)
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
        # Source arrays.
        target_array_path = source_arrays_dir / f"sample_{sid:04d}_q_target.npy"
        sino_obs_array_path = source_arrays_dir / f"sample_{sid:04d}_sino_obs.npy"
        np.save(target_array_path, targets["q_target"])
        np.save(sino_obs_array_path, targets["sino_obs"])
        for row_id, q_pred in per_row_q.items():
            pred_path = source_arrays_dir / f"sample_{sid:04d}_{row_id}_q_pred.npy"
            sino_pred_path = source_arrays_dir / f"sample_{sid:04d}_{row_id}_sino_pred.npy"
            np.save(pred_path, q_pred)
            np.save(sino_pred_path, per_row_sino.get(row_id, np.zeros_like(targets["sino_obs"])))
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
    _write_manifest(preflight_manifest, manifest_path_out)

    return {
        "preflight_manifest_path": str(manifest_path_out),
        "metrics_json_path": str(output_root / "metrics.json"),
        "metrics_csv_path": str(output_root / "metrics.csv"),
        "metric_schema_path": str(output_root / "metric_schema.json"),
        "visual_manifest_path": str(output_root / "visual_manifest.json"),
        "rows": [r.to_dict() for r in row_metrics],
    }


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
        "--dry-run",
        action="store_true",
        help="Validate the manifest, write preflight_manifest.json + metric_schema.json, and exit.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_invocation_artifacts(
        output_dir=output_root,
        script_path=SCRIPT_PATH,
        argv=sys.argv[1:] if argv is None else list(argv),
        parsed_args=vars(args),
        extra={
            "backlog_item": BACKLOG_ITEM,
            "runtime_provenance": capture_runtime_provenance(),
        },
    )
    contract = TrainingContract(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
    )
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
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
