"""Batch emission: canonicalization helpers and the shared row->tuple emitter.

Both the RAM ``_PtychoContainerDataset`` and the mmap ``PtychoDataset``
delegate selected rows to :func:`_emit_ptycho_batch`, which owns the single
conversion from selected fields to the ``(fields, probes, probe_scaling)``
Torch training tuple.
"""

from collections.abc import Mapping

import numpy as np
import torch
from tensordict import TensorDict


def _as_tensor(value):
    if value is None or isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(np.asarray(value))


def _canonical_probe_bank(probe, *, name="probe"):
    """Return probes as one explicit ``(E, P, H, W)`` bank."""

    probe = _as_tensor(probe)
    if probe is None:
        raise ValueError(f"{name} is required")
    probe = probe.to(torch.complex64)
    if probe.ndim == 2:
        return probe.unsqueeze(0).unsqueeze(0)
    if probe.ndim == 3:
        if probe.shape[-1] == 1 and probe.shape[0] == probe.shape[1]:
            return probe[..., 0].unsqueeze(0).unsqueeze(0)
        return probe.unsqueeze(0)
    if probe.ndim == 4:
        return probe
    raise ValueError(
        f"{name} must have shape (H,W), (P,H,W), or (E,P,H,W); "
        f"got {tuple(probe.shape)}."
    )


def _canonical_bank_scalars(value, experiments, *, name):
    value = _as_tensor(value)
    if value is None:
        return torch.ones(experiments, dtype=torch.float32)
    flat = value.to(torch.float32).reshape(-1)
    if flat.numel() == 1:
        return flat.expand(experiments)
    if flat.numel() == experiments:
        return flat
    if experiments == 1:
        # Old RAM batches sometimes carried one repeated per-sample value here.
        # The tuple contract is probe scaling, so only the single probe-bank
        # value is relevant.
        return flat[:1]
    raise ValueError(
        f"{name} must contain one value per experiment; got {flat.numel()} "
        f"values for {experiments} experiments."
    )


def _selected_scale(value, *, batch_size, scalar, name):
    value = _as_tensor(value)
    if value is None:
        value = torch.ones(1, dtype=torch.float32)
    value = value.to(torch.float32)
    if scalar:
        if value.numel() != 1:
            value = value.reshape(-1)[0]
        return value.reshape(1, 1, 1)
    if value.numel() == 1:
        return value.reshape(1, 1, 1, 1).expand(batch_size, -1, -1, -1)
    if value.shape[0] != batch_size:
        raise ValueError(
            f"{name} must be scalar or sample-aligned; got shape "
            f"{tuple(value.shape)} for batch size {batch_size}."
        )
    if value[0].numel() != 1:
        raise ValueError(f"{name} must contain one scalar per sample")
    return value.reshape(batch_size, 1, 1, 1)


def _select_experiment_scalars(values, indices, *, name):
    values = _as_tensor(values)
    if values is None:
        raise ValueError(f"{name} is required")
    values = values.to(torch.float32).reshape(-1)
    if values.numel() == 1:
        return values.expand(indices.numel())
    if indices.numel() and int(indices.max()) >= values.numel():
        raise ValueError(f"{name} has no value for experiment {int(indices.max())}")
    return values[indices]


def _copy_selected_fields(selected_fields):
    if isinstance(selected_fields, TensorDict):
        return TensorDict(
            {key: value for key, value in selected_fields.items()},
            batch_size=selected_fields.batch_size,
        )
    if isinstance(selected_fields, Mapping):
        return dict(selected_fields)
    raise TypeError("selected batch fields must be a mapping or TensorDict")


def _emit_ptycho_batch(
    selected_fields,
    *,
    probes,
    probe_scaling,
    probes_physical=None,
    ci_statistics=None,
    channel_last=False,
):
    """Build the one Torch training tuple from selected RAM or mmap rows."""

    fields = _copy_selected_fields(selected_fields)
    images = _as_tensor(fields.get("images"))
    if images is None:
        raise ValueError("batch fields require images")
    scalar = images.ndim == 3
    if images.ndim not in (3, 4):
        raise ValueError("images must have shape (C,H,W) or (B,C,H,W)")

    def channel_first(value):
        value = _as_tensor(value)
        if value is None or not channel_last:
            return value
        if value.ndim == 4:
            return value.permute(0, 3, 1, 2).clone(
                memory_format=torch.contiguous_format
            )
        if value.ndim == 3:
            return value.permute(2, 0, 1).clone(
                memory_format=torch.contiguous_format
            )
        raise ValueError("channel-last sample fields must have rank 3 or 4")

    for name in (
        "images",
        "observed_images",
        "measured_intensity",
        "label_amp",
        "label_phase",
    ):
        if fields.get(name) is not None:
            fields[name] = channel_first(fields[name])
    images = fields["images"]
    batch_size = 1 if scalar else int(images.shape[0])
    channels = int(images.shape[0] if scalar else images.shape[1])

    coords = _as_tensor(fields.get("coords_relative"))
    if coords is None:
        coords = torch.zeros(
            (channels, 1, 2) if scalar else (batch_size, channels, 1, 2),
            dtype=torch.float32,
        )
    elif channel_last:
        if coords.ndim == 4:
            coords = coords.permute(0, 3, 1, 2).clone(
                memory_format=torch.contiguous_format
            )
        elif coords.ndim == 3:
            coords = coords.permute(2, 0, 1).clone(
                memory_format=torch.contiguous_format
            )
        else:
            raise ValueError("coords_relative must have rank 3 or 4")
    fields["coords_relative"] = coords.to(torch.float32)

    experiment_id = _as_tensor(fields.get("experiment_id"))
    if experiment_id is None:
        experiment_id = torch.zeros(
            () if scalar else batch_size, dtype=torch.long
        )
    experiment_id = experiment_id.to(torch.long)
    if scalar:
        experiment_id = experiment_id.reshape(-1)[0]
    else:
        experiment_id = experiment_id.reshape(-1)
        if experiment_id.numel() == 1:
            experiment_id = experiment_id.expand(batch_size)
        if experiment_id.numel() != batch_size:
            raise ValueError("experiment_id must align with the selected rows")
    fields["experiment_id"] = experiment_id
    if fields.get("object_index") is not None:
        object_index = _as_tensor(fields["object_index"]).to(torch.long)
        fields["object_index"] = (
            object_index.reshape(-1)[0]
            if scalar
            else object_index.reshape(-1)
        )

    ids = experiment_id.reshape(-1)
    probe_bank = _canonical_probe_bank(probes)
    probe_ids = torch.zeros_like(ids) if probe_bank.shape[0] == 1 else ids
    if probe_ids.numel() and int(probe_ids.max()) >= probe_bank.shape[0]:
        raise ValueError(
            f"probe bank has no entry for experiment {int(probe_ids.max())}"
        )
    selected_probes = probe_bank[probe_ids].unsqueeze(1).expand(
        -1, channels, -1, -1, -1
    )
    scaling_bank = _canonical_bank_scalars(
        probe_scaling,
        int(probe_bank.shape[0]),
        name="probe_scaling",
    )
    selected_probe_scaling = scaling_bank[probe_ids].reshape(-1, 1, 1, 1)

    ci_active = probes_physical is not None or ci_statistics is not None
    if ci_active:
        if probes_physical is None or ci_statistics is None:
            raise ValueError(
                "CI batches require physical probes and frozen statistics"
            )
        physical_bank = _canonical_probe_bank(
            probes_physical, name="probe_physical"
        )
        physical_ids = torch.zeros_like(ids) if physical_bank.shape[0] == 1 else ids
        if physical_ids.numel() and int(physical_ids.max()) >= physical_bank.shape[0]:
            raise ValueError(
                "physical probe bank has no entry for selected experiment"
            )
        selected_physical = physical_bank[physical_ids].unsqueeze(1).expand(
            -1, channels, -1, -1, -1
        )
        rms = _select_experiment_scalars(
            ci_statistics.get("rms_input_scale"), ids, name="rms_input_scale"
        ).reshape(-1, 1, 1, 1)
        mean = _select_experiment_scalars(
            ci_statistics.get("mean_measured_intensity"),
            ids,
            name="mean_measured_intensity",
        ).reshape(-1, 1, 1, 1)
        normalization = selected_probe_scaling.unsqueeze(-1)
        fields["measured_intensity"] = fields.get(
            "measured_intensity", fields["images"]
        )
        fields["observed_images"] = fields["measured_intensity"]
        fields["probe_training"] = selected_probes
        fields["probe_physical"] = selected_physical
        fields["probe_normalization"] = normalization
        fields["rms_input_scale"] = rms
        fields["mean_measured_intensity"] = mean
    else:
        fields["observed_images"] = fields.get(
            "observed_images", fields["images"]
        )
        fields["rms_scaling_constant"] = _selected_scale(
            fields.get("rms_scaling_constant"),
            batch_size=batch_size,
            scalar=scalar,
            name="rms_scaling_constant",
        )
        fields["physics_scaling_constant"] = _selected_scale(
            fields.get("physics_scaling_constant"),
            batch_size=batch_size,
            scalar=scalar,
            name="physics_scaling_constant",
        )

    if scalar:
        selected_probes = selected_probes[0]
        selected_probe_scaling = selected_probe_scaling[0]
        if ci_active:
            # Named probe fields retain their explicit leading batch axis even
            # for scalar indexing; only the per-experiment statistics match
            # the scalar TensorDict convention.
            fields["rms_input_scale"] = fields["rms_input_scale"][0]
            fields["mean_measured_intensity"] = fields[
                "mean_measured_intensity"
            ][0]

    return fields, selected_probes, selected_probe_scaling
