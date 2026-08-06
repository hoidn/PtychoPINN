"""Versioned record for rectangular s1/s2 gauge initialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any


RECT_S1S2_INITIALIZATION_SCHEMA = "rect-s1s2-initialization-v1"
RECT_S1S2_DOSE_CLOSURE_PATTERNS = 256
RECT_S1S2_INITIALIZATION_MODES = ("ones", "dose_closure")

_FIELDS = {
    "schema_version",
    "mode",
    "solved_gauge",
    "method",
    "sampled_patterns",
}
_MODE_METHODS = {
    "ones": "unit_default_no_solve",
    "dose_closure": "dose_closure_unit_object",
}


def validate_rect_s1s2_initialization_mode(mode: Any) -> str:
    """Return one supported initialization mode or reject it actionably."""

    if mode in RECT_S1S2_INITIALIZATION_MODES:
        return mode
    if mode == "data":
        raise ValueError(
            "rect_s1s2_init='data' is unsupported; rect_s1s2_init must be "
            "'ones' or 'dose_closure'; historical data artifacts require "
            "historical code or retraining."
        )
    raise ValueError(
        "rect_s1s2_init must be 'ones' or 'dose_closure', "
        f"got {mode!r}"
    )


def _validated_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("rect_s1s2 initialization record must be a mapping")
    if set(payload) != _FIELDS:
        raise ValueError(
            "rect_s1s2 initialization fields must be "
            f"{sorted(_FIELDS)!r}, got {sorted(str(key) for key in payload)!r}"
        )
    schema_version = payload["schema_version"]
    if schema_version != RECT_S1S2_INITIALIZATION_SCHEMA:
        raise ValueError(
            "rect_s1s2 initialization schema_version must be "
            f"{RECT_S1S2_INITIALIZATION_SCHEMA!r}"
        )
    mode = validate_rect_s1s2_initialization_mode(payload["mode"])
    solved_gauge = payload["solved_gauge"]
    if isinstance(solved_gauge, bool) or not isinstance(solved_gauge, Real):
        raise TypeError("rect_s1s2 initialization solved_gauge must be a number")
    solved_gauge = float(solved_gauge)
    if not math.isfinite(solved_gauge) or solved_gauge <= 0.0:
        raise ValueError(
            "rect_s1s2 initialization solved_gauge must be positive and finite"
        )
    method = payload["method"]
    expected_method = _MODE_METHODS[mode]
    if method != expected_method:
        raise ValueError(
            "rect_s1s2 initialization method must be "
            f"{expected_method!r} for mode {mode!r}"
        )
    sampled_patterns = payload["sampled_patterns"]
    if isinstance(sampled_patterns, bool) or not isinstance(
        sampled_patterns,
        Integral,
    ):
        raise TypeError(
            "rect_s1s2 initialization sampled_patterns must be an integer"
        )
    sampled_patterns = int(sampled_patterns)
    if mode == "ones":
        if solved_gauge != 1.0:
            raise ValueError(
                "rect_s1s2 initialization solved_gauge must be 1.0 for "
                "mode 'ones'"
            )
        if sampled_patterns != 0:
            raise ValueError(
                "rect_s1s2 initialization sampled_patterns must be 0 for "
                "mode 'ones'"
            )
    elif sampled_patterns < RECT_S1S2_DOSE_CLOSURE_PATTERNS:
        raise ValueError(
            "rect_s1s2 initialization sampled_patterns must be at least "
            f"{RECT_S1S2_DOSE_CLOSURE_PATTERNS} for mode 'dose_closure'"
        )
    return {
        "mode": mode,
        "solved_gauge": solved_gauge,
        "method": method,
        "sampled_patterns": sampled_patterns,
        "schema_version": schema_version,
    }


@dataclass(frozen=True, slots=True)
class RectS1S2InitializationRecord:
    """Validated initialization identity persisted by Torch training."""

    mode: str
    solved_gauge: float
    method: str
    sampled_patterns: int
    schema_version: str = RECT_S1S2_INITIALIZATION_SCHEMA

    def __post_init__(self) -> None:
        values = _validated_values(self.to_jsonable())
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | "RectS1S2InitializationRecord",
    ) -> "RectS1S2InitializationRecord":
        if isinstance(payload, cls):
            payload = payload.to_jsonable()
        return cls(**_validated_values(payload))

    @classmethod
    def ones(cls) -> "RectS1S2InitializationRecord":
        return cls(
            mode="ones",
            solved_gauge=1.0,
            method=_MODE_METHODS["ones"],
            sampled_patterns=0,
        )

    @classmethod
    def dose_closure(
        cls,
        solved_gauge: float,
        sampled_patterns: int = RECT_S1S2_DOSE_CLOSURE_PATTERNS,
    ) -> "RectS1S2InitializationRecord":
        return cls.from_mapping(
            {
                "schema_version": RECT_S1S2_INITIALIZATION_SCHEMA,
                "mode": "dose_closure",
                "solved_gauge": solved_gauge,
                "method": _MODE_METHODS["dose_closure"],
                "sampled_patterns": sampled_patterns,
            }
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "solved_gauge": self.solved_gauge,
            "method": self.method,
            "sampled_patterns": self.sampled_patterns,
        }


__all__ = [
    "RECT_S1S2_DOSE_CLOSURE_PATTERNS",
    "RECT_S1S2_INITIALIZATION_MODES",
    "RECT_S1S2_INITIALIZATION_SCHEMA",
    "RectS1S2InitializationRecord",
    "validate_rect_s1s2_initialization_mode",
]
