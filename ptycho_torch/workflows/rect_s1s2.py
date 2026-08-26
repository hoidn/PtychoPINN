"""Rectangular s1/s2 gauge initialization and maintained dose-closure loaders.

Owns the selected-row dataset/collation machinery, the row-zero channel
inspection, and the crash-safe training-summary publication used by the
Lightning service.
"""
from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
from typing import Any

from ptycho_torch.rect_s1s2_initialization import (
    RECT_S1S2_DOSE_CLOSURE_PATTERNS,
    RectS1S2InitializationRecord,
)
from ptycho_torch.rect_s1s2_sampling import (
    SelectedDoseClosureRow,
    _base_row_for_logical,
    build_dose_closure_sample_plan,
)
from ptycho_torch.train_utils import PrebuiltPtychoDataModule

def _move_batch_to_device(batch, device):
    """Move tensors in a nested Lightning batch structure to ``device``."""
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            key: _move_batch_to_device(value, device)
            for key, value in batch.items()
        }
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_batch_to_device(value, device) for value in batch)
    return batch


@dataclass(frozen=True, slots=True)
class _RectS1S2IndexedRows:
    value: Any
    access_rows: tuple[SelectedDoseClosureRow, ...]


@dataclass(frozen=True, slots=True)
class _RectS1S2SelectedBatch:
    value: Any
    access_rows: tuple[SelectedDoseClosureRow, ...]


_RECT_S1S2_IDENTITY_FIELD = "__rect_s1s2_logical_row_identity__"


def _rect_s1s2_attach_identities(value, access_rows, *, batched_indexing):
    import torch

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            "rect_s1s2 selected indexing must return a sequence whose first "
            "item is the batch field mapping"
        )
    fields = value[0]
    if _RECT_S1S2_IDENTITY_FIELD in fields:
        raise ValueError(
            "rect_s1s2 reserved identity field collides with dataset fields"
        )
    if isinstance(fields, dict):
        identified_fields = dict(fields)
    elif hasattr(fields, "batch_size") and callable(
        getattr(fields, "clone", None)
    ):
        identified_fields = fields.clone(recurse=False)
    else:
        raise ValueError(
            "rect_s1s2 selected indexing requires mutable mapping fields"
        )
    logical_rows = torch.tensor(
        [row.logical_row for row in access_rows],
        dtype=torch.int64,
    )
    identity = logical_rows if batched_indexing else logical_rows[0]
    identified_fields[_RECT_S1S2_IDENTITY_FIELD] = identity
    if isinstance(value, tuple):
        return (identified_fields, *value[1:])
    return [identified_fields, *value[1:]]


def _rect_s1s2_verify_collated_identities(batch, access_rows):
    import torch

    try:
        fields = batch[0]
        identity = fields[_RECT_S1S2_IDENTITY_FIELD]
        collated_logical_rows = tuple(
            int(value)
            for value in torch.as_tensor(identity).reshape(-1).tolist()
        )
    except Exception as error:
        raise ValueError(
            "rect_s1s2 maintained collation must preserve row identity"
        ) from error
    expected_logical_rows = tuple(row.logical_row for row in access_rows)
    if (
        len(collated_logical_rows) != len(expected_logical_rows)
        or set(collated_logical_rows) != set(expected_logical_rows)
    ):
        raise ValueError(
            "rect_s1s2 maintained collation has missing or extra identity "
            "coverage"
        )
    if collated_logical_rows != expected_logical_rows:
        raise ValueError(
            "rect_s1s2 maintained collation has reordered identity coverage"
        )
    try:
        del fields[_RECT_S1S2_IDENTITY_FIELD]
    except Exception as error:
        raise ValueError(
            "rect_s1s2 maintained collation must expose removable row identity"
        ) from error


class _RectS1S2SelectedDataset:
    """Index only the immutable logical rows selected for dose closure."""

    def __init__(self, dataset, access_rows, *, batched_indexing):
        self.dataset = dataset
        self.access_rows = tuple(access_rows)
        self.batched_indexing = bool(batched_indexing)
        self._ptycho_vectorized_batch = self.batched_indexing

    def __len__(self):
        return len(self.access_rows)

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError(
                "rect_s1s2 selected dataset requires an integer index"
            )
        row = self.access_rows[int(index)]
        try:
            value = self.dataset[row.logical_row]
        except Exception as error:
            raise ValueError(
                "rect_s1s2 selected dataset does not support logical-row "
                "indexing"
            ) from error
        value = _rect_s1s2_attach_identities(
            value,
            (row,),
            batched_indexing=False,
        )
        return _RectS1S2IndexedRows(value=value, access_rows=(row,))

    def __getitems__(self, indices):
        if not self.batched_indexing:
            return [self[index] for index in indices]
        rows = tuple(self.access_rows[int(index)] for index in indices)
        try:
            value = self.dataset.__getitems__(
                [row.logical_row for row in rows]
            )
        except Exception as error:
            raise ValueError(
                "rect_s1s2 selected dataset does not support maintained "
                "vectorized indexing"
            ) from error
        value = _rect_s1s2_attach_identities(
            value,
            rows,
            batched_indexing=True,
        )
        return _RectS1S2IndexedRows(value=value, access_rows=rows)


class _RectS1S2MaintainedCollation:
    def __init__(self, collate_fn, *, batched_indexing):
        self.collate_fn = collate_fn
        self.batched_indexing = bool(batched_indexing)

    def __call__(self, indexed):
        if self.batched_indexing:
            if not isinstance(indexed, _RectS1S2IndexedRows):
                raise ValueError(
                    "rect_s1s2 selected TensorDict indexing returned an "
                    "unsupported value"
                )
            values = indexed.value
            access_rows = indexed.access_rows
        else:
            if not isinstance(indexed, list) or not all(
                isinstance(value, _RectS1S2IndexedRows) for value in indexed
            ):
                raise ValueError(
                    "rect_s1s2 selected dataset indexing returned an "
                    "unsupported value"
                )
            values = [value.value for value in indexed]
            access_rows = tuple(
                row for value in indexed for row in value.access_rows
            )
        try:
            batch = self.collate_fn(values)
        except Exception as error:
            raise ValueError(
                "rect_s1s2 selected rows could not use the maintained "
                "training-loader collation"
            ) from error
        _rect_s1s2_verify_collated_identities(batch, access_rows)
        return _RectS1S2SelectedBatch(value=batch, access_rows=access_rows)


def _rect_s1s2_indexable_dataset(training_loader):
    dataset = getattr(training_loader, "dataset", None)
    if not callable(getattr(dataset, "__len__", None)) or not callable(
        getattr(dataset, "__getitem__", None)
    ):
        raise TypeError(
            "rect_s1s2 dose closure requires an indexable training-loader "
            "dataset"
        )
    try:
        len(dataset)
    except Exception as error:
        raise ValueError(
            "rect_s1s2 dose-closure dataset must have a valid length"
        ) from error
    return dataset


def _rebuild_rect_s1s2_loader(
    training_loader,
    *,
    access_rows,
    batch_size,
):
    """Rebuild one loader over selected logical rows without ambient state."""

    import torch

    dataset = _rect_s1s2_indexable_dataset(training_loader)
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("rect_s1s2 selected batch size must be a positive integer")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("rect_s1s2 selected batch size must be a positive integer")
    collate_fn = getattr(training_loader, "collate_fn", None)
    if not callable(collate_fn):
        raise ValueError(
            "rect_s1s2 dose closure requires maintained callable collation"
        )
    rows = tuple(access_rows)
    if not all(isinstance(row, SelectedDoseClosureRow) for row in rows):
        raise TypeError(
            "rect_s1s2 selected access rows must be immutable selection values"
        )
    if not isinstance(training_loader, torch.utils.data.DataLoader):
        raise TypeError(
            "rect_s1s2 dose closure supports PyTorch DataLoader instances"
        )

    capability_owner = dataset
    while isinstance(capability_owner, torch.utils.data.Subset):
        capability_owner = capability_owner.dataset
    batched_indexing = bool(
        getattr(capability_owner, "_ptycho_vectorized_batch", False)
        and callable(getattr(dataset, "__getitems__", None))
    )
    selected_dataset = _RectS1S2SelectedDataset(
        dataset,
        rows,
        batched_indexing=batched_indexing,
    )
    local_generator = torch.Generator()
    local_generator.manual_seed(0)
    return torch.utils.data.DataLoader(
        selected_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=_RectS1S2MaintainedCollation(
            collate_fn,
            batched_indexing=batched_indexing,
        ),
        generator=local_generator,
    )


def _rect_s1s2_batch_axes(batch, *, inspected_channels=None):
    try:
        fields = batch[0]
    except Exception as error:
        raise ValueError(
            "rect_s1s2 dose closure requires maintained batch collation"
        ) from error
    if "measured_intensity" not in fields:
        raise ValueError(
            "rect_s1s2 dose closure requires CI count-intensity batches "
            "with measured_intensity; legacy normalized-amplitude loaders "
            "are unsupported"
        )
    try:
        images = fields["images"]
        target = fields["measured_intensity"]
    except Exception as error:
        raise ValueError(
            "rect_s1s2 dose closure images and measured_intensity must share "
            "canonical (B, C, H, W) leading axes"
        ) from error
    if (
        images.ndim != 4
        or target.ndim != 4
        or tuple(target.shape[:2]) != tuple(images.shape[:2])
    ):
        raise ValueError(
            "rect_s1s2 dose closure images and measured_intensity must share "
            "canonical (B, C, H, W) leading axes"
        )
    batch_size = int(target.shape[0])
    channels = int(target.shape[1])
    if channels <= 0:
        raise ValueError(
            "rect_s1s2 dose closure requires a positive inspected channel count"
        )
    if inspected_channels is not None and channels != inspected_channels:
        raise ValueError(
            "rect_s1s2 selected row channel count "
            f"{channels} must match inspected channel count {inspected_channels}"
        )
    return fields, batch_size, channels


def _inspect_rect_s1s2_channels(training_loader):
    dataset = _rect_s1s2_indexable_dataset(training_loader)
    if len(dataset) == 0:
        raise ValueError("rect_s1s2 dose closure requires a non-empty dataset")
    access_row = SelectedDoseClosureRow(
        logical_row=0,
        base_row=_base_row_for_logical(dataset, 0),
        channels=(),
    )
    loader = _rebuild_rect_s1s2_loader(
        training_loader,
        access_rows=(access_row,),
        batch_size=1,
    )
    iterator = iter(loader)
    try:
        selected_batch = next(iterator)
    except StopIteration as error:
        raise ValueError(
            "rect_s1s2 row-zero inspection produced no batch"
        ) from error
    if not isinstance(selected_batch, _RectS1S2SelectedBatch):
        raise ValueError("rect_s1s2 row-zero inspection lost identity coverage")
    if selected_batch.access_rows != (access_row,):
        raise ValueError("rect_s1s2 row-zero inspection reordered identity coverage")
    _, batch_size, channels = _rect_s1s2_batch_axes(selected_batch.value)
    if batch_size != 1:
        raise ValueError(
            "rect_s1s2 row-zero inspection must collate exactly one logical row"
        )
    try:
        next(iterator)
    except StopIteration:
        return channels
    raise ValueError("rect_s1s2 row-zero inspection produced extra batches")


def _initialize_rect_s1s2_unmanaged(
    model,
    *,
    mode,
    training_loader=None,
):
    """Initialize the shared rectangular gauge from the fixed uniform sample."""

    import torch

    if mode not in {"ones", "dose_closure"}:
        raise ValueError(f"unsupported rect_s1s2 initialization mode {mode!r}")
    forward_model = getattr(getattr(model, "model", None), "forward_model", None)
    scaler = getattr(forward_model, "rect_scaler", None)
    if scaler is None:
        if mode == "ones":
            return RectS1S2InitializationRecord.ones().to_jsonable()
        raise ValueError(
            "rect_s1s2 dose closure requires a model with a rectangular "
            "physics scaler"
        )
    scaler.s1.data.fill_(1.0)
    scaler.s2.data.fill_(1.0)
    if mode == "ones":
        return RectS1S2InitializationRecord.ones().to_jsonable()
    if training_loader is None:
        raise ValueError("rect_s1s2 dose closure requires a CI training loader")
    dataset = _rect_s1s2_indexable_dataset(training_loader)
    channels = _inspect_rect_s1s2_channels(training_loader)
    available_patterns = len(dataset) * channels
    if available_patterns < RECT_S1S2_DOSE_CLOSURE_PATTERNS:
        raise ValueError(
            "rect_s1s2 dose closure has insufficient detector-pattern slots: "
            f"sampled {available_patterns}, required "
            f"{RECT_S1S2_DOSE_CLOSURE_PATTERNS}. Provide enough training "
            "patterns or use '--rect-s1s2-init ones'."
        )
    plan = build_dose_closure_sample_plan(dataset, channels=channels)
    selected_loader = _rebuild_rect_s1s2_loader(
        training_loader,
        access_rows=plan.access_rows,
        batch_size=getattr(training_loader, "batch_size", None),
    )
    selected_iterator = iter(selected_loader)
    expected_chunks = tuple(
        plan.access_rows[offset : offset + selected_loader.batch_size]
        for offset in range(0, len(plan.access_rows), selected_loader.batch_size)
    )
    observed_pattern_sums = []
    predicted_pattern_sums = []
    contributed_flat_slots = []
    for expected_rows in expected_chunks:
        try:
            selected_batch = next(selected_iterator)
        except StopIteration as error:
            raise ValueError(
                "rect_s1s2 selected loader has missing identity coverage"
            ) from error
        if not isinstance(selected_batch, _RectS1S2SelectedBatch):
            raise ValueError(
                "rect_s1s2 selected loader returned unsupported identity coverage"
            )
        if selected_batch.access_rows != expected_rows:
            raise ValueError(
                "rect_s1s2 selected loader has reordered identity coverage"
            )
        fields, batch_size, selected_channels = _rect_s1s2_batch_axes(
            selected_batch.value,
            inspected_channels=channels,
        )
        if batch_size != len(expected_rows):
            raise ValueError(
                "rect_s1s2 selected batch cardinality must match its exact "
                "identity chunk"
            )
        if selected_channels != channels:
            raise ValueError(
                "rect_s1s2 selected batch channel count changed unexpectedly"
            )
        batch = _move_batch_to_device(selected_batch.value, scaler.s1.device)
        fields = batch[0]
        positions = fields["coords_relative"]
        experiment_ids = fields["experiment_id"]
        target = fields["measured_intensity"]
        probe = fields["probe_training"]
        probe_normalization = fields["probe_normalization"]
        output_scale = probe_normalization.reshape(
            batch_size, 1, 1, 1
        ).reciprocal()
        unit_object = torch.ones_like(fields["images"], dtype=torch.complex64)
        with torch.no_grad():
            predicted = forward_model(
                unit_object,
                target,
                positions,
                probe,
                output_scale,
                experiment_ids,
            )
        if predicted.ndim != 4 or tuple(predicted.shape) != tuple(target.shape):
            raise ValueError(
                "rect_s1s2 dose closure predicted intensity must match "
                "measured_intensity shape (B, C, H, W)"
            )
        mask = torch.zeros(
            (batch_size, channels),
            dtype=torch.bool,
            device=target.device,
        )
        for row_index, access_row in enumerate(expected_rows):
            for channel in access_row.channels:
                flat_slot = access_row.logical_row * channels + channel
                contributed_flat_slots.append(flat_slot)
                mask[row_index, channel] = True
        selected_target = target.to(torch.float64)[mask]
        selected_predicted = predicted.to(torch.float64)[mask]
        expected_selected = sum(len(row.channels) for row in expected_rows)
        if int(mask.sum().item()) != expected_selected:
            raise ValueError(
                "rect_s1s2 selected channel masks have duplicate or missing "
                "flat-slot coverage"
            )
        if bool((selected_target < 0).any().item()):
            raise ValueError(
                "rect_s1s2 dose closure observed counts must be nonnegative"
            )
        observed_pattern_sums.append(
            selected_target.reshape(expected_selected, -1).sum(dim=1)
        )
        predicted_pattern_sums.append(
            selected_predicted.reshape(expected_selected, -1).sum(dim=1)
        )
    try:
        next(selected_iterator)
    except StopIteration:
        pass
    else:
        raise ValueError(
            "rect_s1s2 selected loader has extra identity coverage"
        )
    if (
        len(contributed_flat_slots) != RECT_S1S2_DOSE_CLOSURE_PATTERNS
        or len(set(contributed_flat_slots))
        != RECT_S1S2_DOSE_CLOSURE_PATTERNS
        or set(contributed_flat_slots) != set(plan.flat_slots)
    ):
        raise ValueError(
            "rect_s1s2 selected channel masks have missing, extra, or "
            "duplicate flat-slot coverage"
        )
    observed_sum = float(torch.cat(observed_pattern_sums).sum().item())
    predicted_sum = float(torch.cat(predicted_pattern_sums).sum().item())
    if not math.isfinite(observed_sum) or observed_sum <= 0.0:
        raise ValueError(
            "rect_s1s2 dose closure observed count sum must be positive and "
            f"finite; got {observed_sum!r}"
        )
    if not math.isfinite(predicted_sum) or predicted_sum <= 0.0:
        raise ValueError(
            "rect_s1s2 dose closure predicted intensity sum must be positive "
            f"and finite; got {predicted_sum!r}"
        )

    closure = observed_sum / predicted_sum
    if not math.isfinite(closure) or closure <= 0.0:
        raise ValueError(
            "rect_s1s2 dose closure c* must be positive and finite; "
            f"got {closure!r}"
        )
    gauge = math.sqrt(closure)
    if not math.isfinite(gauge) or gauge <= 0.0:
        raise ValueError(
            "rect_s1s2 dose closure gauge must be positive and finite; "
            f"got {gauge!r}"
        )
    scaler.s1.data.fill_(gauge)
    scaler.s2.data.fill_(gauge)
    return RectS1S2InitializationRecord.dose_closure(gauge).to_jsonable()


def _initialize_rect_s1s2(
    model,
    *,
    mode,
    training_loader=None,
):
    """Run initialization inference while preserving every module state."""

    if mode != "dose_closure":
        return _initialize_rect_s1s2_unmanaged(
            model,
            mode=mode,
            training_loader=training_loader,
        )
    training_states = tuple(
        (module, bool(module.training)) for module in model.modules()
    )
    model.eval()
    try:
        return _initialize_rect_s1s2_unmanaged(
            model,
            mode=mode,
            training_loader=training_loader,
        )
    finally:
        for module, training in training_states:
            module.training = training


def _write_training_summary_atomic(path, record):
    """Crash-safe JSON publication for the rank-zero training summary."""

    import json
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validated = RectS1S2InitializationRecord.from_mapping(record)
    encoded = (
        json.dumps(
            validated.to_jsonable(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_training_summary_and_barrier(trainer, path, record):
    """Publish on global zero, then release every rank from the live strategy."""

    if bool(getattr(trainer, "is_global_zero", False)):
        _write_training_summary_atomic(path, record)
    strategy = getattr(trainer, "strategy", None)
    barrier = getattr(strategy, "barrier", None)
    if not callable(barrier):
        raise RuntimeError(
            "Lightning strategy must expose barrier() while publishing the "
            "training summary"
        )
    barrier("rect_s1s2_training_summary")


def _rect_s1s2_training_loader(data_product, train_loader, mode):
    """Resolve the training source only when dose closure consumes it."""

    if mode == "ones":
        return None
    if isinstance(data_product, PrebuiltPtychoDataModule):
        data_product.setup("fit")
        return data_product.train_dataloader()
    return train_loader


def _effective_dataloader_settings(
    data_product,
    train_loader,
    execution_config,
):
    """Resolve the loader settings used by this Trainer invocation."""

    if isinstance(data_product, PrebuiltPtychoDataModule):
        return data_product._loader_settings()
    num_workers = int(getattr(train_loader, "num_workers", 0))
    return {
        "num_workers": num_workers,
        "pin_memory": bool(getattr(train_loader, "pin_memory", False)),
        "persistent_workers": (
            bool(getattr(train_loader, "persistent_workers", False))
            if num_workers > 0
            else False
        ),
        "prefetch_factor": (
            getattr(train_loader, "prefetch_factor", None)
            if num_workers > 0
            else None
        ),
    }


