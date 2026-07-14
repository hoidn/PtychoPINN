"""Committed step-parity driver CLI (task-21c review P-4; the C-1 lesson).

The build/step-parity JSONs under
``.artifacts/bridge_ladder/diagnostics/step_parity/`` were produced by an
ad-hoc driver that was never committed. This module is the committed,
re-runnable replacement for the PROBE-layout legs: it loads the sealed rung0
reference checkpoint (identity-verified), reconstructs the deterministic
first raster batch from a sealed mmap work directory (read-only ``np.memmap``
views; nothing under ``.artifacts/`` is written except the caller-chosen
output JSON), and measures the training loss at fixed weights on CPU:

- ``documented_rank_gain1``  — the documented (B, C, P, N, N) probe layout at
  ``amplitude_physics_gain=1`` (the corrected physics; historically 0.936635
  on the rung1e first raster batch).
- ``documented_rank_gain16`` — the explicit gain replacement for the banned
  flat layout. With per-batch identical probes the flat broadcast was exactly
  a x16 amplitude gain, so this leg must reproduce the pre-fix flat-layout
  loss (historically 0.073175 on the same batch — the task-21a "0.9361 ->
  0.0755" reshape isolation, review-reproduced as 0.936635 -> 0.073175).
- ``scale_reshape_only``     — reshaping the batch[2] scale operand alone is a
  bit-no-op (amplitude-mode compute_loss never reads it).
- ``flat_probe``             — post-fix outcome: ``ProbeLayoutError``
  (PROBE-RANK-001; the layout is banned, so the pre-fix measurement is quoted
  with provenance instead of re-run).

The emitted JSON also records the cfg-delta between the checkpoint's
persisted configs and current dataclass defaults (review C-1: cfg-delta
belongs IN the artifact) and the P-1 label correction for the sealed
``step_parity.json``'s fourth leg.

Usage (defaults target the sealed rung1e work dir + rung0 reference)::

    python -m scripts.studies.ablation.runtime_ladder_step_parity_cli \
        --output .artifacts/bridge_ladder/diagnostics/step_parity/probe_layout_step_parity.json
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "REFERENCE_CHECKPOINT",
    "REFERENCE_EVIDENCE",
    "RUNG1E_MEMMAP_ROOT",
    "FirstRasterBatch",
    "load_first_raster_batch",
    "load_reference_model",
    "loss_with_gain",
    "flat_probe_leg",
    "scale_reshape_leg",
    "build_report",
    "main",
]

SCHEMA_VERSION = "bridge_ladder_probe_layout_step_parity_v1"

#: Sealed producers this driver re-measures against (paths relative to the
#: repository root; overridable via CLI arguments).
REFERENCE_EVIDENCE = Path(
    ".artifacts/reference_qualification/run1/grid_lines_hybrid_resnet_reference/"
    "reference_evidence.json"
)
REFERENCE_CHECKPOINT = Path(
    ".artifacts/reference_qualification/run1/grid_lines_hybrid_resnet_reference/"
    "work/checkpoints/epoch=epoch=04-mae_val=mae_val_loss=0.0823.ckpt"
)
RUNG1E_MEMMAP_ROOT = Path(
    ".artifacts/bridge_ladder/seed3_split/rung1e_sampler_plus_unit_norm/work/memmap"
)

_MEMMAP_DTYPES = {
    "torch.float32": np.float32,
    "torch.float64": np.float64,
    "torch.int64": np.int64,
    "torch.int32": np.int32,
    "torch.bool": np.bool_,
}

_BATCH_FIELDS = (
    "images",
    "coords_relative",
    "rms_scaling_constant",
    "physics_scaling_constant",
)


@dataclass
class FirstRasterBatch:
    """The deterministic first raster training batch of a sealed mmap dir."""

    fields: dict[str, Any]
    probe: Any  # (B, C=1, P, N, N) — the documented layout
    scale: Any  # (B, 1, 1, 1)
    provenance: dict[str, Any]

    def as_batch(self) -> tuple[dict[str, Any], Any, Any]:
        return self.fields, self.probe, self.scale


def load_first_raster_batch(
    memmap_root: Path, batch_size: int = 16
) -> FirstRasterBatch:
    """Reconstruct the first raster batch from a sealed mmap work directory.

    Read-only: fields come from ``np.memmap(..., mode='r')`` views of the
    ``train/*.memmap`` files (copied out), the probe from
    ``state_files.npz``'s ``data_dict`` exactly as
    ``PtychoDataset.__getitem__`` emits it (documented (B, C, P, N, N)
    layout: ``probes[get_idx].unsqueeze(1)`` with C=1). Raster order
    (indices ``0..batch_size-1``) makes the leg deterministic without the
    shuffled sampler.
    """
    import torch

    memmap_root = Path(memmap_root)
    train_dir = memmap_root / "train"
    meta = json.loads((train_dir / "meta.json").read_text(encoding="utf-8"))

    def field(name: str) -> Any:
        info = meta[name]
        view = np.memmap(
            train_dir / f"{name}.memmap",
            dtype=_MEMMAP_DTYPES[info["dtype"]],
            mode="r",
            shape=tuple(info["shape"]),
        )
        return torch.from_numpy(np.array(view[:batch_size]))

    fields = {name: field(name) for name in _BATCH_FIELDS}
    fields["experiment_id"] = field("experiment_id").to(torch.long)

    state = np.load(memmap_root / "state_files.npz", allow_pickle=True)
    data_dict = state["data_dict"].item()
    get_idx = fields["experiment_id"].reshape(-1)
    if int(get_idx.max()) >= int(data_dict["probes"].shape[0]):
        # Single-experiment work dirs store one probe row; mirror the
        # dataloader's n_files==1 zero-index convention.
        get_idx = torch.zeros_like(get_idx)
    probe = data_dict["probes"][get_idx].unsqueeze(1)  # (B, 1, P, N, N)
    scale = data_dict["probe_scaling"][get_idx].view(-1, 1, 1, 1)

    images_sha = hashlib.sha256(
        np.ascontiguousarray(fields["images"].numpy()).tobytes()
    ).hexdigest()
    provenance = {
        "memmap_root": str(memmap_root),
        "sampler": "raster",
        "indices": [0, batch_size],
        "batch_size": batch_size,
        "images_slice_sha256": images_sha,
        "probe_batch_shape": list(probe.shape),
    }
    return FirstRasterBatch(fields=fields, probe=probe, scale=scale, provenance=provenance)


def load_reference_model(checkpoint: Path, evidence: Path) -> Any:
    """Load the sealed rung0 reference checkpoint on CPU, identity-verified
    against the sealed evidence hash (delegates to the committed cross-eval
    loader)."""
    from .runtime_ladder_cross_eval import load_sealed_checkpoint

    payload = json.loads(Path(evidence).read_text(encoding="utf-8"))
    return load_sealed_checkpoint(
        Path(checkpoint),
        expected_sha256=payload["checkpoint_sha256"],
        device="cpu",
    )


def loss_with_gain(model: Any, batch: tuple, gain: float, *, seed: int = 3) -> float:
    """One deterministic amplitude-mode loss at fixed weights and explicit
    ``amplitude_physics_gain`` (the config object is shared across the
    Lightning module and its ForwardModel, which reads the gain live)."""
    import torch

    config = model.model_config
    original = getattr(config, "amplitude_physics_gain", 1.0)
    try:
        config.amplitude_physics_gain = float(gain)
        torch.manual_seed(int(seed))
        model.train()
        with torch.no_grad():
            loss = model.compute_loss(batch)
        return float(loss)
    finally:
        config.amplitude_physics_gain = original


def flat_probe_leg(model: Any, batch: tuple) -> dict[str, Any]:
    """Post-fix outcome of the banned flat (B, H, W) probe layout."""
    from ptycho_torch.model import ProbeLayoutError

    fields, probe, scale = batch
    flat = probe.reshape(probe.shape[0], probe.shape[-2], probe.shape[-1])
    try:
        loss_with_gain_result = loss_with_gain(model, (fields, flat, scale), 1.0)
    except ProbeLayoutError as error:
        return {
            "outcome": "ProbeLayoutError",
            "error": str(error).splitlines()[0],
        }
    return {"outcome": "loss", "loss": loss_with_gain_result}


def scale_reshape_leg(model: Any, batch: tuple) -> dict[str, Any]:
    """Reshaping only the scale operand must be a bit-no-op (amplitude-mode
    compute_loss never reads batch[2])."""
    fields, probe, scale = batch
    base = loss_with_gain(model, batch, 1.0)
    reshaped = loss_with_gain(
        model, (fields, probe, scale.reshape(scale.shape[0], 1)), 1.0
    )
    return {"loss": [base, reshaped], "bit_identical": base == reshaped}


def _config_delta_vs_defaults(model: Any) -> dict[str, Any]:
    """cfg-delta (C-1): checkpoint-persisted configs vs current defaults."""
    from ptycho_torch.config_params import (
        DataConfig,
        InferenceConfig,
        ModelConfig,
        TrainingConfig,
    )

    from .runtime_ladder_step_parity import hparams_config_delta

    class _Defaults:
        hparams = {
            "model_config": ModelConfig(),
            "data_config": DataConfig(),
            "training_config": TrainingConfig(),
            "inference_config": InferenceConfig(),
        }

    class _Loaded:
        hparams = {
            "model_config": model.model_config,
            "data_config": model.data_config,
            "training_config": model.training_config,
            "inference_config": model.inference_config,
        }

    delta = hparams_config_delta(_Loaded(), _Defaults())
    return {
        key: {name: [repr(a), repr(b)] for name, (a, b) in fields.items()}
        for key, fields in delta.items()
    }


#: P-1 fold: the sealed step_parity.json's fourth leg is mislabeled — its
#: content (losses [0.0714, 0.0714], identical=True) is a dictionary-batch
#: cross-model identity re-measurement, not a dict-vs-mmap own-batch
#: comparison. Sealed evidence is never rewritten; the correction rides here.
STEP_PARITY_LABEL_CORRECTION = {
    "artifact": "step_parity.json",
    "leg": "own_batches_dictmodel_dictbatch_vs_mmapmodel_mmapbatch",
    "issue": (
        "mislabeled: recorded losses [0.07137257, 0.07137257] with"
        " identical=True match a dictionary-batch cross-model identity leg"
        " (duplicate content of cross_model_same_dict_batch), not the"
        " dict-model/dict-batch vs mmap-model/mmap-batch comparison the name"
        " claims (that comparison is the 'own_batches' leg, losses"
        " [0.07137257, 0.93608087])"
    ),
    "source": ".superpowers/sdd/task-21c-review.md (2026-07-12 review, P-1)",
}

#: Pre-fix reshape-isolation measurement (P-1 persistence). The flat layout
#: is banned post-fix, so these numbers are quoted with provenance rather
#: than re-run; the documented_rank_gain16 leg reproduces the flat number
#: through the explicit-gain mechanism on the same batch and weights.
PRE_FIX_RESHAPE_ISOLATION = {
    "documented_rank_loss": 0.936635,
    "flat_rank_loss": 0.073175,
    "scale_reshape_only": "bit-identical no-op",
    "weights": "sealed rung0 reference checkpoint",
    "batch": "rung1e first raster train batch (B=16)",
    "code_state": "pre-fix (commit 2a9ee2ad9)",
    "source": (
        ".superpowers/sdd/task-21c-review.md independent reproduction"
        " (2026-07-12); original 0.9361 -> 0.0755 in"
        " .superpowers/sdd/task-21a-report.md step-parity section"
    ),
}


def build_report(
    model: Any,
    batch: FirstRasterBatch,
    *,
    checkpoint: Path,
    evidence: Path,
) -> dict[str, Any]:
    """Measure all probe-layout legs and assemble the artifact payload."""
    evidence_payload = json.loads(Path(evidence).read_text(encoding="utf-8"))
    gain1 = loss_with_gain(model, batch.as_batch(), 1.0)
    gain16 = loss_with_gain(model, batch.as_batch(), 16.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": 3,
        "device": "cpu",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": evidence_payload["checkpoint_sha256"],
        },
        "batch_provenance": batch.provenance,
        "legs": {
            "documented_rank_gain1": {"loss": gain1},
            "documented_rank_gain16": {
                "loss": gain16,
                "reproduces": (
                    "pre-fix flat-layout loss on the same batch/weights"
                    " (flat broadcast == exact x16 amplitude gain for"
                    " per-batch identical probes)"
                ),
            },
            "scale_reshape_only": scale_reshape_leg(model, batch.as_batch()),
            "flat_probe": flat_probe_leg(model, batch.as_batch()),
        },
        "pre_fix_reshape_isolation": PRE_FIX_RESHAPE_ISOLATION,
        "step_parity_label_correction": STEP_PARITY_LABEL_CORRECTION,
        "config_delta_vs_defaults": _config_delta_vs_defaults(model),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="bridge_ladder_probe_layout_step_parity")
    parser.add_argument("--reference-checkpoint", type=Path, default=REFERENCE_CHECKPOINT)
    parser.add_argument("--reference-evidence", type=Path, default=REFERENCE_EVIDENCE)
    parser.add_argument("--memmap-root", type=Path, default=RUNG1E_MEMMAP_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    model = load_reference_model(args.reference_checkpoint, args.reference_evidence)
    batch = load_first_raster_batch(args.memmap_root, batch_size=args.batch_size)
    report = build_report(
        model,
        batch,
        checkpoint=args.reference_checkpoint,
        evidence=args.reference_evidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    legs = report["legs"]
    print(
        "documented_rank gain1={:.6f} gain16={:.6f} flat_probe={} -> {}".format(
            legs["documented_rank_gain1"]["loss"],
            legs["documented_rank_gain16"]["loss"],
            legs["flat_probe"]["outcome"],
            args.output,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
