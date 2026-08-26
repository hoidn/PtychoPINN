"""Named numerical adapters between acquisition storage and Torch batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DataAdapterName = Literal["dictionary_parity", "loader"]


@dataclass(frozen=True)
class DataAdapterPolicy:
    """Resolved preprocessing behavior shared by all Torch ingestion rails."""

    name: DataAdapterName
    normalize: Literal["Batch", "None"]
    probe_normalize: bool
    count_scale_mode: Literal["off", "loader"]
    explicit_unit_scales: bool


_POLICIES: dict[DataAdapterName, DataAdapterPolicy] = {
    "dictionary_parity": DataAdapterPolicy(
        name="dictionary_parity",
        normalize="None",
        probe_normalize=False,
        count_scale_mode="off",
        explicit_unit_scales=True,
    ),
    "loader": DataAdapterPolicy(
        name="loader",
        normalize="Batch",
        probe_normalize=True,
        count_scale_mode="loader",
        explicit_unit_scales=False,
    ),
}


def resolve_data_adapter(name: DataAdapterName) -> DataAdapterPolicy:
    """Return one immutable named adapter; reject implicit or unknown modes."""

    try:
        return _POLICIES[name]
    except (KeyError, TypeError) as error:
        expected = ", ".join(repr(item) for item in sorted(_POLICIES))
        raise ValueError(
            f"data adapter must be one of {expected}, got {name!r}"
        ) from error
