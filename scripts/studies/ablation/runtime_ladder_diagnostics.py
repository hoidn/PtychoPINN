"""V3b batch-consumption diagnostic: dictionary vs mmap training consumption.

Step-0 proved the mmap loader ingests bit-identical measurement tensors. Rung
1c restored unit normalization and recovered the rung0 quality band,
establishing normalization ownership as the mismatch. This module diffs what
the two training paths actually CONSUME: it captures
the exact dataloaders each real flow hands to ``lightning.pytorch.Trainer``
(by stubbing ``Trainer`` with a sentinel-raising recorder — the construction
code runs verbatim through ``run_torch_training`` respectively
``train_via_generic_loader``; nothing is reimplemented), iterates them
deterministically, and compares per-batch sample identities, per-epoch
multisets/orderings/step counts, and per-sample tensor bytes for aligned
samples, plus the validation loaders feeding checkpoint selection.

CPU only; no training step ever executes (the sentinel aborts inside
``Trainer.fit`` before any optimization).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .dataset_provenance import canonical_array_sha256
from .runtime_errors import RuntimeExecutionError
from .runtime_ladder_capture import (
    CapturedLoaders,
    capture_dictionary_training_loaders,
    capture_mmap_training_loaders,
)

__all__ = [
    "CapturedLoaders",
    "EpochRecord",
    "batch_sample_records",
    "build_image_identity_index",
    "capture_dictionary_training_loaders",
    "capture_mmap_training_loaders",
    "compare_consumption",
    "record_epochs",
    "run_batch_consumption_diff",
]

#: Mapping fields harvested per sample when present in a batch.
_SAMPLE_FIELDS = (
    "coords_relative",
    "coords_center",
    "rms_scaling_constant",
    "physics_scaling_constant",
)


def build_image_identity_index(images: np.ndarray) -> dict[str, int]:
    """Source-image hash -> scan index; collisions fail closed."""
    index: dict[str, int] = {}
    for position, image in enumerate(np.asarray(images)):
        digest = canonical_array_sha256(np.ascontiguousarray(image))
        if digest in index:
            raise RuntimeExecutionError(
                "identity_index",
                f"image hash collision between scans {index[digest]} and "
                f"{position}; identity mapping would be ambiguous (duplicate "
                "measurements)",
            )
        index[digest] = position
    return index


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _find_mapping(batch: Any) -> tuple[Mapping[str, Any], list[Any]]:
    """Locate the tensor mapping plus the remaining batch components."""
    if isinstance(batch, Mapping) or hasattr(batch, "keys"):
        return batch, []
    if isinstance(batch, (tuple, list)):
        mapping = None
        rest: list[Any] = []
        for component in batch:
            if mapping is None and (
                isinstance(component, Mapping) or hasattr(component, "keys")
            ):
                mapping = component
            else:
                rest.append(component)
        if mapping is not None:
            return mapping, rest
    raise RuntimeExecutionError(
        "batch_records", f"unrecognized batch structure {type(batch).__name__}"
    )


def batch_sample_records(batch: Any) -> list[dict[str, Any]]:
    """Normalize one consumed batch into per-sample field records.

    Handles both the dictionary path's ``(tensor_dict, probe, scaling)``
    triple and the mmap path's TensorDict-based batch. Every record carries
    the squeezed 2-D measurement image, the batch-level probe hash, the
    per-sample mapping fields that are present, and ``center_scan_id`` when
    the path provides one (mmap) or ``None`` (dictionary).
    """
    mapping, rest = _find_mapping(batch)
    images = _to_numpy(mapping["images"])
    count = images.shape[0]
    probe_sha = None
    if rest:
        probe = _to_numpy(rest[0])
        while probe.ndim > 2:
            probe = probe[0]
        probe_sha = canonical_array_sha256(np.ascontiguousarray(probe))
    ids = None
    keys = set(mapping.keys())
    if "center_scan_id" in keys:
        ids = _to_numpy(mapping["center_scan_id"]).reshape(count, -1)[:, 0]
    records: list[dict[str, Any]] = []
    for position in range(count):
        record: dict[str, Any] = {
            "image": np.ascontiguousarray(np.squeeze(images[position])),
            "probe_sha": probe_sha,
            "center_scan_id": None if ids is None else int(ids[position]),
        }
        for name in _SAMPLE_FIELDS:
            if name in keys:
                value = mapping[name]
                if value is not None:
                    record[name] = _to_numpy(value)[position]
        if len(rest) > 1 and rest[1] is not None:
            scaling = _to_numpy(rest[1])
            record["loader_scaling"] = (
                scaling[position] if scaling.ndim and scaling.shape[0] == count
                else scaling
            )
        records.append(record)
    return records


@dataclass
class EpochRecord:
    """One epoch of consumed batches for one path."""

    steps: int
    batch_sizes: list[int]
    ordered_ids: list[int]
    unmatched: int
    deep_samples: dict[int, dict[str, Any]] = field(default_factory=dict)
    first_unmatched_detail: dict[str, Any] | None = None


def record_epochs(
    loader: Any,
    identity_index: Mapping[str, int],
    *,
    epochs: int,
    deep_ids: frozenset[int] = frozenset(),
) -> list[EpochRecord]:
    """Iterate a captured loader and record what training would consume."""
    results: list[EpochRecord] = []
    for _ in range(epochs):
        ordered: list[int] = []
        batch_sizes: list[int] = []
        unmatched = 0
        first_unmatched: dict[str, Any] | None = None
        deep: dict[int, dict[str, Any]] = {}
        steps = 0
        for batch in loader:
            steps += 1
            records = batch_sample_records(batch)
            batch_sizes.append(len(records))
            for record in records:
                digest = canonical_array_sha256(record["image"])
                source_id = identity_index.get(digest, -1)
                if source_id < 0:
                    unmatched += 1
                    if first_unmatched is None:
                        image = record["image"]
                        first_unmatched = {
                            "dtype": str(image.dtype),
                            "shape": list(image.shape),
                            "center_scan_id": record["center_scan_id"],
                        }
                ordered.append(source_id)
                if source_id in deep_ids and source_id not in deep:
                    deep[source_id] = {
                        key: value
                        for key, value in record.items()
                        if key != "center_scan_id"
                    }
        results.append(
            EpochRecord(
                steps=steps,
                batch_sizes=batch_sizes,
                ordered_ids=ordered,
                unmatched=unmatched,
                deep_samples=deep,
                first_unmatched_detail=first_unmatched,
            )
        )
    return results


def _field_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    a_np, b_np = np.asarray(a), np.asarray(b)
    return bool(
        np.array_equal(np.squeeze(a_np), np.squeeze(b_np))
        and a_np.dtype == b_np.dtype
    )


def _field_detail(a: Any, b: Any) -> dict[str, Any]:
    def describe(value: Any) -> Any:
        if value is None or isinstance(value, str):
            return value
        array = np.asarray(value)
        return {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sample_values": np.asarray(array).reshape(-1)[:4].tolist(),
        }

    return {"a": describe(a), "b": describe(b)}


def compare_consumption(
    records_a: list[EpochRecord],
    records_b: list[EpochRecord],
    *,
    labels: tuple[str, str],
) -> dict[str, Any]:
    """Compare two paths' consumption records epoch by epoch.

    Field attestation is exhaustive (review V1): EVERY field of every aligned
    deep sample is byte-compared in every epoch — a divergence never
    short-circuits the remaining fields. Per-field divergence counts land in
    ``field_divergences``; fields byte-equal on every compared sample are
    listed in ``fields_attested_equal``. Order facts are measured, not
    derived from sampler class semantics (review V2): per epoch
    ``identity_raster_order`` (ordered ids == 0..n-1 per path) and top-level
    ``epoch_order_stable`` (per path, every epoch's order equals epoch 0's).
    """
    epochs: list[dict[str, Any]] = []
    field_stats: dict[str, dict[str, Any]] = {}
    for epoch_index, (a, b) in enumerate(zip(records_a, records_b)):
        order_equal = a.ordered_ids == b.ordered_ids
        first_order = None
        if not order_equal:
            for position, (left, right) in enumerate(
                zip(a.ordered_ids, b.ordered_ids)
            ):
                if left != right:
                    first_order = {"position": position, "ids": [left, right]}
                    break
            else:
                first_order = {
                    "position": min(len(a.ordered_ids), len(b.ordered_ids)),
                    "ids": [None, None],
                }
        epochs.append(
            {
                "step_counts": [a.steps, b.steps],
                "multiset_equal": Counter(a.ordered_ids) == Counter(b.ordered_ids),
                "order_equal": order_equal,
                "first_order_divergence": first_order,
                "unmatched": [a.unmatched, b.unmatched],
                "first_unmatched_detail": [
                    a.first_unmatched_detail,
                    b.first_unmatched_detail,
                ],
                "batch_sizes_head": [a.batch_sizes[:3], b.batch_sizes[:3]],
                "batch_sizes_tail": [a.batch_sizes[-1:], b.batch_sizes[-1:]],
                "identity_raster_order": [
                    record.ordered_ids == list(range(len(record.ordered_ids)))
                    for record in (a, b)
                ],
            }
        )
        shared = sorted(set(a.deep_samples) & set(b.deep_samples))
        for sample_id in shared:
            fields_a = a.deep_samples[sample_id]
            fields_b = b.deep_samples[sample_id]
            for name in sorted(set(fields_a) | set(fields_b)):
                stats = field_stats.setdefault(
                    name,
                    {"field": name, "compared_samples": 0,
                     "divergent_samples": 0, "first": None},
                )
                stats["compared_samples"] += 1
                if _field_equal(fields_a.get(name), fields_b.get(name)):
                    continue
                stats["divergent_samples"] += 1
                if stats["first"] is None:
                    stats["first"] = {
                        "epoch": epoch_index,
                        "sample_id": sample_id,
                        "detail": _field_detail(
                            fields_a.get(name), fields_b.get(name)
                        ),
                    }
    field_divergences = [
        stats for stats in field_stats.values() if stats["divergent_samples"]
    ]
    field_divergences.sort(
        key=lambda stats: (
            stats["first"]["epoch"],
            stats["first"]["sample_id"],
            stats["field"],
        )
    )
    first_field_divergence = None
    if field_divergences:
        head = field_divergences[0]
        first_field_divergence = {"field": head["field"], **head["first"]}
    return {
        "labels": list(labels),
        "epochs": epochs,
        "epoch_order_stable": [
            all(
                record.ordered_ids == records[0].ordered_ids
                for record in records
            )
            for records in (records_a, records_b)
        ],
        "field_divergences": field_divergences,
        "fields_attested_equal": sorted(
            name
            for name, stats in field_stats.items()
            if stats["compared_samples"] and not stats["divergent_samples"]
        ),
        "first_field_divergence": first_field_divergence,
    }


def _deep_id_set(total: int, count: int = 16) -> frozenset[int]:
    if total <= count:
        return frozenset(range(total))
    stride = max(1, total // count)
    return frozenset(list(range(4)) + list(range(0, total, stride)))


def run_batch_consumption_diff(
    captured_a: CapturedLoaders,
    captured_b: CapturedLoaders,
    *,
    train_source: Path,
    test_source: Path,
    epochs: int = 2,
    seed: int = 3,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Full train+val consumption diff between two captured flows."""
    import torch

    def source_images(path: Path) -> np.ndarray:
        with np.load(Path(path), allow_pickle=False) as archive:
            key = "diffraction" if "diffraction" in archive.files else "diff3d"
            images = np.asarray(archive[key], dtype=np.float32)
        if images.ndim == 4 and images.shape[-1] == 1:
            images = images[..., 0]
        return images

    train_index = build_image_identity_index(source_images(train_source))
    test_index = build_image_identity_index(source_images(test_source))
    deep_train = _deep_id_set(len(train_index))
    deep_test = _deep_id_set(len(test_index))

    def record(loader: Any, index: Mapping[str, int], deep: frozenset[int]):
        torch.manual_seed(seed)
        np.random.seed(seed)
        return record_epochs(loader, index, epochs=epochs, deep_ids=deep)

    labels = (captured_a.flow, captured_b.flow)
    train_summary = compare_consumption(
        record(captured_a.train_loader, train_index, deep_train),
        record(captured_b.train_loader, train_index, deep_train),
        labels=labels,
    )
    val_summary = compare_consumption(
        record(captured_a.val_loader, test_index, deep_test),
        record(captured_b.val_loader, test_index, deep_test),
        labels=labels,
    )
    summary = {
        "schema_version": "bridge_ladder_batch_consumption_diff_v2",
        # Review V2: epoch orders are produced under controlled seeding
        # OUTSIDE Lightning's fit loop — exact permutations are emulation
        # artifacts; only policy-level facts (sampler class, multisets,
        # step counts, measured order booleans, field bytes) are claims.
        "order_provenance": "emulated_outside_lightning_fit",
        "labels": list(labels),
        "seed": seed,
        "epochs": epochs,
        "train_source": str(train_source),
        "test_source": str(test_source),
        "samplers": {
            captured_a.flow: captured_a.sampler_info(),
            captured_b.flow: captured_b.sampler_info(),
        },
        "train": train_summary,
        "val": val_summary,
        "first_field_divergence": (
            train_summary["first_field_divergence"]
            or val_summary["first_field_divergence"]
        ),
        "field_divergences": {
            "train": train_summary["field_divergences"],
            "val": val_summary["field_divergences"],
        },
        "fields_attested_equal": {
            "train": train_summary["fields_attested_equal"],
            "val": val_summary["fields_attested_equal"],
        },
    }
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return summary


def render_verdict_table(summary: Mapping[str, Any]) -> str:
    """Short human-readable verdict table for the report."""
    lines = [f"paths: {summary['labels'][0]} vs {summary['labels'][1]}"]
    for section in ("train", "val"):
        for index, epoch in enumerate(summary[section]["epochs"]):
            lines.append(
                f"{section} epoch {index}: steps={epoch['step_counts']} "
                f"multiset_equal={epoch['multiset_equal']} "
                f"order_equal={epoch['order_equal']} "
                f"unmatched={epoch['unmatched']}"
            )
    divergence = summary["first_field_divergence"]
    lines.append(f"first_field_divergence: {divergence}")
    for section in ("train", "val"):
        for entry in summary[section]["field_divergences"]:
            lines.append(
                f"{section} field {entry['field']}: divergent "
                f"{entry['divergent_samples']}/{entry['compared_samples']} "
                f"first={entry['first']}"
            )
        attested = summary[section]["fields_attested_equal"]
        lines.append(f"{section} fields_attested_equal: {attested}")
        lines.append(
            f"{section} epoch_order_stable: "
            f"{summary[section]['epoch_order_stable']}"
        )
    samplers = summary["samplers"]
    for label, info in samplers.items():
        lines.append(f"sampler[{label}]: {info}")
    return "\n".join(lines)
