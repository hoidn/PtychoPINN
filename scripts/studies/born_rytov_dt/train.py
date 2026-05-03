"""BRDT task-local training entrypoint.

Narrow training surface for the bounded four-row preflight. It:

- accepts a row identifier (architecture), dataset-manifest authority,
  and an output root;
- builds the adapter, classical backend info, and locked operator;
- supports a small ``--fast-dev-run`` mode that runs a single batch on
  CPU/CUDA so adapter readiness can be proven without launching the
  benchmark-grade preflight here;
- writes invocation/provenance plus a sanity summary the four-row
  preflight can later read.

This entrypoint does NOT register BRDT as a CDI generator and does not
import :mod:`ptycho_torch.train`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from ptycho_torch.physics import BornRytovForward2D
from scripts.studies.born_rytov_dt import dataset_contract as dc
from scripts.studies.born_rytov_dt.classical import (
    ClassicalBackendInfo,
    derive_born_init_image,
    detect_classical_backend,
)
from scripts.studies.born_rytov_dt.data import (
    BRDTSmokeSplit,
    DatasetAuthority,
    assert_input_mode_supported,
    brdt_collate,
    load_dataset_authority,
)
from scripts.studies.born_rytov_dt.lightning_module import BRDTTrainingModule
from scripts.studies.born_rytov_dt.models import (
    AdapterBuildError,
    build_neural_adapter,
)
from scripts.studies.born_rytov_dt.reporting import (
    SanitySummary,
    build_adapter_contract,
    rows_with_sanity_summary,
    write_json,
)
from scripts.studies.born_rytov_dt.run_config import (
    HYBRID_FAMILY_ROW_IDS,
    LossWeights,
    RowConfig,
    default_row_roster,
    make_blocked_row,
)
from scripts.studies.invocation_logging import write_invocation_artifacts


SCRIPT_PATH = "scripts/studies/born_rytov_dt/train.py"


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


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _prepare_input(
    batch: Dict[str, Any],
    *,
    operator: BornRytovForward2D,
    backend: ClassicalBackendInfo,
    in_channels: int,
) -> torch.Tensor:
    sinogram = batch["sinogram"].to(operator.angles.device)
    init = derive_born_init_image(
        sinogram,
        operator=operator,
        backend=backend,
    )
    if in_channels == 1:
        return init
    if in_channels == 2:
        # Two-channel layout: (real backprop magnitude, zero placeholder).
        zero = torch.zeros_like(init)
        return torch.cat([init, zero], dim=1)
    raise ValueError(f"unsupported in_channels={in_channels} for born_init_image")


def run_training(
    *,
    architecture: str,
    manifest_path: Path,
    output_root: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device_choice: str,
    fast_dev_run: bool,
    in_channels: int,
    hybrid_label: str,
) -> Dict[str, Any]:
    """Run the bounded sanity training for a single row."""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    authority: DatasetAuthority = load_dataset_authority(manifest_path)
    device = _select_device(device_choice)
    operator = _build_operator(device)
    backend = detect_classical_backend()

    effective_hybrid_label = _resolve_hybrid_label(architecture, hybrid_label)
    rows: List[RowConfig] = default_row_roster(
        dataset_id=authority.dataset_id,
        operator_version=authority.operator_version,
        hybrid_label=effective_hybrid_label,
    )
    selected = next((r for r in rows if r.row_id == architecture), None)
    if selected is None:
        raise ValueError(
            f"architecture={architecture!r} is not part of the bounded roster: "
            f"{[r.row_id for r in rows]}"
        )

    assert_input_mode_supported(selected.input_mode)

    if architecture == "classical_born_backprop":
        row_summary = _classical_only(
            rows=rows,
            selected=selected,
            authority=authority,
            operator=operator,
            backend=backend,
            output_root=output_root,
        )
        return row_summary

    try:
        adapter = build_neural_adapter(
            architecture=selected.model,
            in_channels=in_channels,
            out_channels=1,
            grid_size=dc.LOCKED_GRID_SIZE,
        )
    except AdapterBuildError as exc:
        blocker_row = make_blocked_row(
            row_id=selected.row_id,
            model=selected.model,
            training=selected.training,
            dataset_id=authority.dataset_id,
            operator_version=authority.operator_version,
            blocker_reason=exc.reason,
            blocker_message=str(exc),
            paper_label=selected.paper_label,
        )
        return _emit_blocked_row(
            row=blocker_row,
            authority=authority,
            backend=backend,
            output_root=output_root,
        )
    adapter = adapter.to(device)
    info = adapter.info()

    module = BRDTTrainingModule(
        model=adapter,
        operator=operator,
        normalization=authority.normalization,
        weights=LossWeights(),
        output_space="normalized_q",
    ).to(device)

    train_split = BRDTSmokeSplit(
        authority.split_paths["train"], normalization=authority.normalization
    )
    loader = DataLoader(
        train_split,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=brdt_collate,
    )
    optim = torch.optim.Adam(module.parameters(), lr=float(learning_rate))

    train_steps = 0
    last_breakdown: Dict[str, float] = {}
    last_total = float("nan")
    module.train()
    for _ in range(max(1, int(epochs))):
        for batch in loader:
            x = _prepare_input(
                batch, operator=operator, backend=backend, in_channels=in_channels
            )
            x = x.to(device)
            q_pred = module(x)
            total, breakdown = module.compute_loss(
                q_pred=q_pred,
                q_true_norm=batch["q_true_norm"].to(device),
                q_true_physical=batch["q_true_physical"].to(device),
                sinogram_obs=batch["sinogram"].to(device),
            )
            optim.zero_grad(set_to_none=True)
            total.backward()
            optim.step()
            train_steps += 1
            last_breakdown = breakdown.to_dict()
            last_total = float(total.detach().item())
            if fast_dev_run:
                break
        if fast_dev_run:
            break

    module.eval()
    eval_batch = next(iter(loader))
    # Born-init derivation uses the operator's autograd path, so it must
    # NOT run under no_grad. The model forward + operator residual can.
    x = _prepare_input(
        eval_batch, operator=operator, backend=backend, in_channels=in_channels
    ).to(device).detach()
    with torch.no_grad():
        q_pred = module(x)
        eval_image_mae = float(
            (q_pred - eval_batch["q_true_norm"].to(device)).abs().mean().item()
        )
        q_phys = module.to_physical_q(q_pred)
        y_pred = operator(q_phys)
        sino_obs = eval_batch["sinogram"].to(device)
        rel_phys = float(
            ((y_pred - sino_obs).norm() / (sino_obs.norm() + 1e-8)).item()
        )

    summary = SanitySummary(
        row_id=selected.row_id,
        architecture=selected.model,
        parameter_count=info.parameter_count,
        train_steps=train_steps,
        final_loss_total=last_total,
        final_loss_breakdown=last_breakdown,
        eval_image_mae_norm=eval_image_mae,
        eval_relative_physics=rel_phys,
        row_status="completed" if not fast_dev_run else "feasibility_only",
        note="fast_dev_run sanity execution" if fast_dev_run else None,
    )

    rows_payload = rows_with_sanity_summary(
        rows, selected_row_id=selected.row_id, summary=summary
    )
    contract = build_adapter_contract(
        dataset_id=authority.dataset_id,
        operator_version=authority.operator_version,
        rows=rows_payload,
        classical_backend={
            "name": backend.name,
            "reason": backend.reason,
            "claim_boundary": backend.claim_boundary,
        },
        loss_contract=module.loss_contract(),
        extra={
            "device": str(device),
            "fast_dev_run": bool(fast_dev_run),
            "manifest_path": str(authority.manifest_path),
            "in_channels": int(in_channels),
        },
    )
    contract_path = output_root / "adapter_contract.json"
    write_json(contract_path, contract)
    eval_path = output_root / "eval_summary.json"
    write_json(eval_path, summary.to_dict())
    return {
        "summary": summary.to_dict(),
        "adapter_contract_path": str(contract_path),
        "eval_summary_path": str(eval_path),
        "row_status": summary.row_status,
    }


def _resolve_hybrid_label(architecture: str, hybrid_label: str) -> str:
    """Resolve the effective ``hybrid_label`` from the requested architecture.

    The Hybrid-family row may be surfaced as ``hybrid_resnet`` or
    ``sru_net``. When the architecture explicitly names one of those
    Hybrid-family rows the architecture choice is authoritative — the
    roster's ``row_id`` is forced to match so the requested visible row
    cannot silently fall back to the other label (implementation-review
    HIGH-2). For non-Hybrid-family architectures the ``hybrid_label``
    flag controls how the Hybrid-family row is presented in the roster
    that the contract emits.
    """
    if architecture in HYBRID_FAMILY_ROW_IDS:
        return architecture
    return hybrid_label


def _classical_only(
    *,
    rows: List[RowConfig],
    selected: RowConfig,
    authority: DatasetAuthority,
    operator: BornRytovForward2D,
    backend: ClassicalBackendInfo,
    output_root: Path,
) -> Dict[str, Any]:
    """Sanity reference path for the classical Born backprop row."""
    train_split = BRDTSmokeSplit(
        authority.split_paths["train"], normalization=authority.normalization
    )
    loader = DataLoader(
        train_split,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=brdt_collate,
    )
    batch = next(iter(loader))
    sinogram = batch["sinogram"].to(operator.angles.device)
    init = derive_born_init_image(sinogram, operator=operator, backend=backend)
    image_mae = float((init - batch["q_true_physical"].to(init.device)).abs().mean().item())
    summary = SanitySummary(
        row_id=selected.row_id,
        architecture=selected.model,
        parameter_count=0,
        train_steps=0,
        final_loss_total=float("nan"),
        final_loss_breakdown={},
        eval_image_mae_norm=None,
        eval_relative_physics=None,
        row_status="feasibility_only",
        note=(
            f"classical reference via backend={backend.name}; "
            f"physical_q image_mae={image_mae:.4e}"
        ),
    )
    eval_path = output_root / "eval_summary.json"
    write_json(eval_path, summary.to_dict())
    rows_payload = rows_with_sanity_summary(
        rows, selected_row_id=selected.row_id, summary=summary
    )
    contract = build_adapter_contract(
        dataset_id=authority.dataset_id,
        operator_version=authority.operator_version,
        rows=rows_payload,
        classical_backend={
            "name": backend.name,
            "reason": backend.reason,
            "claim_boundary": backend.claim_boundary,
        },
        loss_contract={"training_label": selected.training},
        extra={
            "manifest_path": str(authority.manifest_path),
            "split": "train",
            "execution_path": "classical_only",
        },
    )
    contract_path = output_root / "adapter_contract.json"
    write_json(contract_path, contract)
    return {
        "summary": summary.to_dict(),
        "adapter_contract_path": str(contract_path),
        "eval_summary_path": str(eval_path),
        "row_status": summary.row_status,
    }


def _emit_blocked_row(
    *,
    row: RowConfig,
    authority: DatasetAuthority,
    backend: ClassicalBackendInfo,
    output_root: Path,
) -> Dict[str, Any]:
    payload = build_adapter_contract(
        dataset_id=authority.dataset_id,
        operator_version=authority.operator_version,
        rows=[row.to_dict()],
        classical_backend={
            "name": backend.name,
            "reason": backend.reason,
            "claim_boundary": backend.claim_boundary,
        },
        loss_contract={"training_label": row.training},
        extra={"manifest_path": str(authority.manifest_path)},
    )
    write_json(output_root / "adapter_contract.json", payload)
    return {"summary": row.to_dict(), "row_status": row.row_status}


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brdt_task_adapters_train",
        description=(
            "Bounded BRDT adapter sanity training (NOT a benchmark run). "
            "Builds one adapter and runs a small training step against the "
            "locked smoke dataset/operator authorities."
        ),
    )
    parser.add_argument(
        "--architecture",
        required=True,
        choices=[
            "classical_born_backprop",
            "unet",
            "fno_vanilla",
            "hybrid_resnet",
            "sru_net",
        ],
        help="Row identifier from the bounded four-row roster.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to the BRDT smoke dataset_manifest.json.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory under which adapter_contract.json + eval_summary.json are written.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a single batch only (sanity-readiness mode).",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of channels in the born_init_image input (1 = real magnitude only).",
    )
    parser.add_argument(
        "--hybrid-label",
        default="hybrid_resnet",
        choices=["hybrid_resnet", "sru_net"],
        help="Visible label for the Hybrid-family row in adapter_contract.json.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_invocation_artifacts(
        output_root,
        SCRIPT_PATH,
        sys.argv[1:] if argv is None else list(argv),
        vars(args),
        extra={"backlog_item": "2026-04-29-brdt-task-adapters"},
    )
    result = run_training(
        architecture=args.architecture,
        manifest_path=args.manifest,
        output_root=output_root,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        device_choice=str(args.device),
        fast_dev_run=bool(args.fast_dev_run),
        in_channels=int(args.in_channels),
        hybrid_label=str(args.hybrid_label),
    )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
