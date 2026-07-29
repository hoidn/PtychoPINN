"""2x2 checkpoint x evaluation-path cross-eval matrix (post-training seam).

Diagnoses whether the mmap rungs' residual amp-SSIM regression lives in the
models or in the rungs' inference/reassembly/eval context. Sealed
checkpoints (identity-verified against sealed evidence hashes) are swapped
across the two evaluation paths while the historical grid-lines stitcher and
the ``eval_reconstruction`` metric convention are held fixed:

- ``reference_dictionary`` — rung 0's exact path
  (``runtime_reference_execution``: dictionary test container via
  ``load_cached_dataset_with_metadata``, reference-arm runner config).
- ``mmap_generic`` — the mmap rungs' exact path
  (``runtime_ladder_execution``: generic twin container via
  ``load_generic_test_dict`` with recipe-synthesized stitch metadata,
  rung-resolved runner config).

Every stage is import-linked to the sealed producers — inference
(``run_torch_inference``), stitching
(``_reassemble_predictions_for_metrics`` in ``grid_lines`` mode), and the
metric (``ptycho.evaluation.eval_reconstruction``); nothing is
reimplemented. Cells A/B recompute the sealed numbers as harness
validations; cells C/D are the cross terms.

Read-only over sealed artifacts: checkpoints are loaded, never written;
outputs land under the caller-provided diagnostics directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .dataset_provenance import canonical_array_sha256
from .runtime_checkpoint import restore_checkpoint_from_hparams
from .runtime_errors import (
    RuntimeExecutionError,
    StudyRequestError,
)
from .runtime_reference_execution import (
    DECLARED_GAUGE_HANDLING,
    _complex_patches,
    _ground_truth,
    _metric_pair,
)

MATRIX_SCHEMA_VERSION = "bridge_ladder_cross_eval_matrix_v1"

#: Convention provenance recorded in the matrix artifact. Both sealed
#: producers used exactly this stitcher+metric pair (provenance:
#: runtime_reference_execution.py stitch/metric stages and
#: runtime_ladder_execution.py dual-stitch/metric stages with
#: gated_evaluator == "historical" sealed on every mmap rung).
STITCHER_PROVENANCE = (
    "grid_lines_torch_runner._reassemble_predictions_for_metrics"
    "(reassembly_mode='grid_lines')"
)
METRIC_PROVENANCE = "ptycho.evaluation.eval_reconstruction"


_PATH_RUNNER_OPERANDS = ("train_npz", "test_npz", "output_dir")


def runner_config_delta(cfg_a: Any, cfg_b: Any) -> dict[str, list[Any]]:
    """Field-by-field delta of two TorchRunnerConfigs, path operands excluded
    (they always differ between contexts and are never metric-relevant)."""
    from dataclasses import fields as dataclass_fields

    delta: dict[str, list[Any]] = {}
    for field in dataclass_fields(cfg_a):
        if field.name in _PATH_RUNNER_OPERANDS:
            continue
        value_a = getattr(cfg_a, field.name)
        value_b = getattr(cfg_b, field.name)
        if value_a != value_b:
            delta[field.name] = [value_a, value_b]
    return delta


@dataclass(frozen=True)
class CellRequest:
    """One matrix cell: a sealed checkpoint driven through one eval path."""

    cell_id: str
    checkpoint_id: str
    checkpoint_path: Path
    checkpoint_sha256: str
    eval_path: str
    sealed_expected: Mapping[str, float] | None = None


def load_sealed_checkpoint(
    checkpoint_path: Path, *, expected_sha256: str, device: str
) -> Any:
    """Load a sealed checkpoint through the canonical Lightning loader.

    Fail-closed identity first: the file's sha256 must match the sealed
    evidence hash BEFORE any deserialization.

    Restoration uses the checkpoint's OWN persisted hyper_parameters (the
    Lightning module saves its data/model/training/inference config
    dataclasses into every checkpoint), via the same strict
    ``PtychoPINN_Lightning.load_from_checkpoint`` call the canonical
    ``lightning_utils.load_checkpoint_with_configs`` makes. That helper is
    not usable directly here — documented evidence: it requires a
    ``configs/`` sidecar directory that NEITHER sealed flow writes (both the
    reference qualification and the ladder train+infer in-process), so the
    requirement fails equally for both paths and is a loader precondition,
    not a cross-path architectural incompatibility. ``strict=True`` keeps
    the era-incompatibility fail-closed property: a weight-shape mismatch
    raises instead of loading loosely.
    """
    return restore_checkpoint_from_hparams(
        checkpoint_path,
        expected_sha256=expected_sha256,
        device=device,
    )


def evaluate_cell(
    *,
    model: Any,
    runner_cfg: Any,
    test_data: Mapping[str, Any],
    test_metadata: Mapping[str, Any] | None,
    label: str,
    trim_offset: int,
) -> dict[str, Any]:
    """Inference -> historical grid-lines stitch -> eval_reconstruction.

    The exact chain both sealed producers ran, import-linked
    (see module docstring); the stitch mode is forced to ``grid_lines``
    regardless of the incoming config so every cell scores the historical
    canvas — the canvas the sealed SSIMs were computed on.
    """
    from ptycho import evaluation

    from scripts.studies import grid_lines_torch_runner as runner_mod

    predictions = runner_mod.run_torch_inference(
        model, test_data, runner_cfg, metadata=test_metadata
    )
    patches = _complex_patches(runner_mod, predictions)
    ground_truth = np.squeeze(np.asarray(_ground_truth(test_data)))
    canvas, _, _ = runner_mod._reassemble_predictions_for_metrics(
        patches,
        ground_truth,
        test_data,
        test_metadata,
        replace(runner_cfg, reassembly_mode="grid_lines"),
    )
    canvas = np.squeeze(np.asarray(canvas))
    if canvas.shape != ground_truth.shape:
        raise RuntimeExecutionError(
            "no_resize",
            f"historical canvas shape {canvas.shape} != ground truth "
            f"{ground_truth.shape}; resizing is prohibited and would "
            "invalidate the cross-eval comparison",
        )
    metrics = evaluation.eval_reconstruction_explicit(
        canvas[..., None],
        ground_truth[..., None],
        label=label,
        trim_offset=trim_offset,
    )
    amp_mae, phase_mae = _metric_pair(metrics, "mae")
    amp_ssim, phase_ssim = _metric_pair(metrics, "ssim")
    return {
        "amp_ssim": amp_ssim,
        "phase_ssim": phase_ssim,
        "amp_mae": amp_mae,
        "phase_mae": phase_mae,
        "historical_canvas_sha256": canonical_array_sha256(canvas),
        "pre_stitch_patch_sha256": canonical_array_sha256(patches),
        "canvas": canvas,
    }


def assemble_matrix(
    requests: list[CellRequest],
    run_cell: Callable[[CellRequest], dict[str, Any]],
    *,
    output_dir: Path,
    eval_context_delta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every cell, persist canvases + the matrix JSON, return the matrix.

    ``run_cell`` is injected so the assembly (persistence, sealed-delta
    computation, provenance) is testable without GPU inference.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells: dict[str, dict[str, Any]] = {}
    for request in requests:
        if request.cell_id in cells:
            raise StudyRequestError(
                f"duplicate matrix cell id {request.cell_id!r}"
            )
        payload = dict(run_cell(request))
        canvas = np.asarray(payload.pop("canvas"))
        cell_dir = output_dir / request.cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        np.save(cell_dir / "canvas.npy", canvas)
        sealed_expected = (
            dict(request.sealed_expected)
            if request.sealed_expected is not None
            else None
        )
        sealed_delta = None
        if sealed_expected is not None:
            sealed_delta = {
                key: float(payload[key]) - float(value)
                for key, value in sealed_expected.items()
            }
        cells[request.cell_id] = {
            "checkpoint_id": request.checkpoint_id,
            "checkpoint_path": str(request.checkpoint_path),
            "checkpoint_sha256": request.checkpoint_sha256,
            "eval_path": request.eval_path,
            "sealed_expected": sealed_expected,
            "sealed_delta": sealed_delta,
            "canvas_path": str(cell_dir / "canvas.npy"),
            **payload,
        }
    matrix = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "stitcher": STITCHER_PROVENANCE,
        "metric": METRIC_PROVENANCE,
        "gauge_handling": DECLARED_GAUGE_HANDLING,
        "eval_context_runner_delta": (
            dict(eval_context_delta) if eval_context_delta is not None else None
        ),
        "cells": cells,
    }
    (output_dir / "cross_eval_matrix.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return matrix


def _default_eval_contexts(
    spec: Any, rung: Any, args: Any
) -> dict[str, tuple[Any, Any, Any, int]]:
    """The two sealed evaluation contexts, import-linked (see module doc)."""
    from scripts.studies import grid_lines_torch_runner as runner_mod

    from .runtime_ladder_execution import build_runner_config, load_generic_test_dict
    from .runtime_reference import load_reference_spec

    reference_spec = load_reference_spec(
        spec.baseline.reference_spec, base_dir=spec.base_dir
    )
    arm = reference_spec.arm(spec.baseline.reference_id)
    reference_cfg = runner_mod.TorchRunnerConfig(
        train_npz=Path(args.dict_train),
        test_npz=Path(args.dict_test),
        output_dir=Path(args.work_dir) / "reference_ctx",
        **dict(arm.runner),
    )
    reference_data = runner_mod.load_cached_dataset_with_metadata(
        Path(args.dict_test)
    )
    mmap_cfg = build_runner_config(
        rung.resolved_config,
        train_npz=Path(args.generic_train),
        test_npz=Path(args.generic_test),
        output_dir=Path(args.work_dir) / "mmap_ctx",
    )
    recipe = spec.dataset(rung.dataset).recipe
    mmap_data = load_generic_test_dict(Path(args.generic_test), recipe)
    return {
        "reference_dictionary": (
            reference_cfg,
            *reference_data,
            int(reference_spec.recipe.offset),
        ),
        "mmap_generic": (mmap_cfg, *mmap_data, int(recipe.offset)),
    }


def main(argv: list[str] | None = None) -> int:
    """Committed CLI for the 2x2 matrix (task-21c review C-1)."""
    import argparse
    import tempfile

    from .runtime_ladder_evidence import parse_sealed_rung_evidence
    from .runtime_ladder_spec import load_ladder_spec

    parser = argparse.ArgumentParser(prog="bridge_ladder_cross_eval")
    parser.add_argument("--ladder-spec", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--rung", default="rung1e_sampler_plus_unit_norm")
    parser.add_argument("--rung-evidence", type=Path, required=True)
    parser.add_argument("--rung-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--dict-train", type=Path, required=True)
    parser.add_argument("--dict-test", type=Path, required=True)
    parser.add_argument("--generic-train", type=Path, required=True)
    parser.add_argument("--generic-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--work-dir", type=Path,
        default=Path(tempfile.mkdtemp(prefix="cross_eval_")),
    )
    args = parser.parse_args(argv)

    spec = load_ladder_spec(args.ladder_spec, base_dir=args.base_dir)
    rung = spec.rung(args.rung)
    rung_payload, _ = parse_sealed_rung_evidence(
        Path(args.rung_evidence).read_bytes()
    )
    import json as json_module

    reference_payload = json_module.loads(
        spec.baseline.evidence.read_text(encoding="utf-8")
    )
    contexts = _default_eval_contexts(spec, rung, args)
    models = {
        "rung0_reference": load_sealed_checkpoint(
            args.reference_checkpoint,
            expected_sha256=reference_payload["checkpoint_sha256"],
            device=args.device,
        ),
        args.rung: load_sealed_checkpoint(
            args.rung_checkpoint,
            expected_sha256=rung_payload["checkpoint_sha256"],
            device=args.device,
        ),
    }
    checkpoints = {
        "rung0_reference": (
            args.reference_checkpoint, reference_payload["checkpoint_sha256"]
        ),
        args.rung: (args.rung_checkpoint, rung_payload["checkpoint_sha256"]),
    }
    sealed = {
        "A": {
            "amp_ssim": float(reference_payload["fixture_amp_ssim"]),
            "phase_ssim": float(reference_payload["fixture_phase_ssim"]),
        },
        "B": {
            "amp_ssim": float(rung_payload["metrics"]["amp_ssim"]),
            "phase_ssim": float(rung_payload["metrics"]["phase_ssim"]),
        },
    }

    def run_cell(request: CellRequest) -> dict[str, Any]:
        from scripts.studies import grid_lines_torch_runner as runner_mod

        cfg, test_data, test_metadata, trim_offset = contexts[request.eval_path]
        runner_mod.setup_torch_configs(cfg)
        payload = evaluate_cell(
            model=models[request.checkpoint_id], runner_cfg=cfg,
            test_data=test_data, test_metadata=test_metadata,
            label=f"cross_eval_{request.cell_id}",
            trim_offset=trim_offset,
        )
        print(
            f"cell {request.cell_id} [{request.checkpoint_id} x "
            f"{request.eval_path}]: amp_ssim={payload['amp_ssim']:.4f} "
            f"phase_ssim={payload['phase_ssim']:.4f}", flush=True,
        )
        return payload

    requests = [
        CellRequest(
            cell_id=cell_id, checkpoint_id=ckpt_id,
            checkpoint_path=checkpoints[ckpt_id][0],
            checkpoint_sha256=checkpoints[ckpt_id][1],
            eval_path=eval_path, sealed_expected=sealed.get(cell_id),
        )
        for cell_id, ckpt_id, eval_path in (
            ("A", "rung0_reference", "reference_dictionary"),
            ("B", args.rung, "mmap_generic"),
            ("C", args.rung, "reference_dictionary"),
            ("D", "rung0_reference", "mmap_generic"),
        )
    ]
    matrix = assemble_matrix(
        requests, run_cell, output_dir=args.output_dir,
        eval_context_delta=runner_config_delta(
            contexts["reference_dictionary"][0], contexts["mmap_generic"][0]
        ),
    )
    for cell_id, cell in sorted(matrix["cells"].items()):
        print(
            f"  {cell_id}: amp={cell['amp_ssim']:.4f} "
            f"phase={cell['phase_ssim']:.4f} sealed_delta={cell['sealed_delta']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
