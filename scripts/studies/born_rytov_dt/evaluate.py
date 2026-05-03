"""BRDT task-local evaluation entrypoint.

Narrow evaluation surface for the bounded four-row preflight. It loads
a split through the locked dataset/operator authorities, runs either
the classical reference path or a built (but freshly-initialized)
neural adapter on a small batch, and emits a row-status payload the
later preflight aggregator can consume.

Like :mod:`scripts.studies.born_rytov_dt.train`, this entrypoint does
NOT register BRDT as a CDI generator and is feasibility-only.
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
    derive_born_init_image,
    detect_classical_backend,
)
from scripts.studies.born_rytov_dt.data import (
    BRDTSmokeSplit,
    SUPPORTED_SPLITS,
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
    write_json,
)
from scripts.studies.born_rytov_dt.run_config import (
    LossWeights,
    default_row_roster,
    make_blocked_row,
)
from scripts.studies.invocation_logging import write_invocation_artifacts


SCRIPT_PATH = "scripts/studies/born_rytov_dt/evaluate.py"


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


def run_evaluation(
    *,
    architecture: str,
    manifest_path: Path,
    output_root: Path,
    split: str,
    batch_size: int,
    device_choice: str,
    in_channels: int,
    hybrid_label: str,
    dry_run: bool,
) -> Dict[str, Any]:
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unknown split={split!r}; allowed: {SUPPORTED_SPLITS}")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    authority = load_dataset_authority(manifest_path)
    device = _select_device(device_choice)
    operator = _build_operator(device)
    backend = detect_classical_backend()

    rows = default_row_roster(
        dataset_id=authority.dataset_id,
        operator_version=authority.operator_version,
        hybrid_label=hybrid_label,
    )
    selected = next((r for r in rows if r.row_id == architecture), None)
    if selected is None and architecture in {"hybrid_resnet", "sru_net"}:
        selected = next(r for r in rows if r.model in {"hybrid_resnet", "sru_net"})
    if selected is None:
        raise ValueError(
            f"architecture={architecture!r} not part of bounded roster: "
            f"{[r.row_id for r in rows]}"
        )
    assert_input_mode_supported(selected.input_mode)

    split_dataset = BRDTSmokeSplit(
        authority.split_paths[split], normalization=authority.normalization
    )
    loader = DataLoader(
        split_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=brdt_collate,
    )

    if dry_run:
        # Validate the contract surface without running any forward pass.
        rows_payload = [r.to_dict() for r in rows]
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
                "split": split,
                "dry_run": True,
            },
        )
        write_json(output_root / "adapter_contract.json", contract)
        return {
            "row_status": "feasibility_only",
            "summary": {"row_id": selected.row_id, "dry_run": True},
            "adapter_contract_path": str(output_root / "adapter_contract.json"),
        }

    if architecture == "classical_born_backprop":
        batch = next(iter(loader))
        sinogram = batch["sinogram"].to(device)
        # Local adjoint backend uses autograd; derive outside no_grad.
        init = derive_born_init_image(sinogram, operator=operator, backend=backend).detach()
        image_mae = float(
            (init - batch["q_true_physical"].to(init.device)).abs().mean().item()
        )
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
        write_json(output_root / "eval_summary.json", summary.to_dict())
        return {"row_status": summary.row_status, "summary": summary.to_dict()}

    try:
        adapter = build_neural_adapter(
            architecture=selected.model,
            in_channels=in_channels,
            out_channels=1,
            grid_size=dc.LOCKED_GRID_SIZE,
        ).to(device)
    except AdapterBuildError as exc:
        blocker = make_blocked_row(
            row_id=selected.row_id,
            model=selected.model,
            training=selected.training,
            dataset_id=authority.dataset_id,
            operator_version=authority.operator_version,
            blocker_reason=exc.reason,
            blocker_message=str(exc),
            paper_label=selected.paper_label,
        )
        contract = build_adapter_contract(
            dataset_id=authority.dataset_id,
            operator_version=authority.operator_version,
            rows=[blocker.to_dict()],
            classical_backend={
                "name": backend.name,
                "reason": backend.reason,
                "claim_boundary": backend.claim_boundary,
            },
            loss_contract={"training_label": selected.training},
            extra={"manifest_path": str(authority.manifest_path)},
        )
        write_json(output_root / "adapter_contract.json", contract)
        return {"row_status": blocker.row_status, "summary": blocker.to_dict()}

    module = BRDTTrainingModule(
        model=adapter,
        operator=operator,
        normalization=authority.normalization,
        weights=LossWeights(),
        output_space="normalized_q",
    ).to(device)

    module.eval()
    batch = next(iter(loader))
    sinogram = batch["sinogram"].to(device)
    # Born-init derivation uses operator autograd; run outside no_grad.
    init = derive_born_init_image(sinogram, operator=operator, backend=backend).detach()
    if in_channels == 2:
        init = torch.cat([init, torch.zeros_like(init)], dim=1)
    with torch.no_grad():
        q_pred = module(init.to(device))
        eval_image_mae = float(
            (q_pred - batch["q_true_norm"].to(device)).abs().mean().item()
        )
        q_phys = module.to_physical_q(q_pred)
        y_pred = operator(q_phys)
        rel_phys = float(
            ((y_pred - sinogram).norm() / (sinogram.norm() + 1e-8)).item()
        )

    summary = SanitySummary(
        row_id=selected.row_id,
        architecture=selected.model,
        parameter_count=adapter.info().parameter_count,
        train_steps=0,
        final_loss_total=float("nan"),
        final_loss_breakdown={},
        eval_image_mae_norm=eval_image_mae,
        eval_relative_physics=rel_phys,
        row_status="feasibility_only",
        note="evaluation-only sanity (untrained adapter)",
    )
    write_json(output_root / "eval_summary.json", summary.to_dict())
    return {"row_status": summary.row_status, "summary": summary.to_dict()}


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brdt_task_adapters_evaluate",
        description=(
            "Bounded BRDT adapter sanity evaluation (NOT a benchmark run). "
            "Loads the locked smoke dataset and runs a single batch through "
            "the requested row."
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
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=list(SUPPORTED_SPLITS))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--in-channels", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "--hybrid-label",
        default="hybrid_resnet",
        choices=["hybrid_resnet", "sru_net"],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit adapter_contract.json without running the model forward.",
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
    result = run_evaluation(
        architecture=args.architecture,
        manifest_path=args.manifest,
        output_root=output_root,
        split=args.split,
        batch_size=int(args.batch_size),
        device_choice=str(args.device),
        in_channels=int(args.in_channels),
        hybrid_label=str(args.hybrid_label),
        dry_run=bool(args.dry_run),
    )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
