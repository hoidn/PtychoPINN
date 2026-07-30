"""Reusable exact-representation scalar types for configuration boundaries."""

from __future__ import annotations

import math
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
)


def _require_exact_int(value: Any) -> Any:
    if type(value) is not int:
        raise ValueError("must be an exact built-in integer")
    return value


def _require_exact_optional_int(value: Any) -> Any:
    if value is not None and type(value) is not int:
        raise ValueError("must be an exact built-in integer or None")
    return value


def _require_exact_bool(value: Any) -> Any:
    if type(value) is not bool:
        raise ValueError("must be an exact built-in boolean")
    return value


def _require_exact_finite_number(value: Any) -> Any:
    if type(value) not in {int, float}:
        raise ValueError("must be an exact built-in integer or float")
    return value


def _require_exact_str(value: Any) -> Any:
    if type(value) is not str:
        raise ValueError("must be an exact built-in string")
    return value


def _require_finite_number(value: int | float) -> int | float:
    if type(value) is float and not math.isfinite(value):
        raise ValueError("must be finite")
    return value


_StrictPositiveInt = Annotated[
    StrictInt,
    BeforeValidator(_require_exact_int),
    Field(gt=0),
]
_StrictNonNegativeInt = Annotated[
    StrictInt,
    BeforeValidator(_require_exact_int),
    Field(ge=0),
]
_StrictOptionalInt = Annotated[
    StrictInt | None,
    BeforeValidator(_require_exact_optional_int),
]
_StrictBool = Annotated[
    StrictBool,
    BeforeValidator(_require_exact_bool),
]
_StrictFiniteNumber = Annotated[
    StrictInt | StrictFloat,
    BeforeValidator(_require_exact_finite_number),
    AfterValidator(_require_finite_number),
]
_StrictFinitePositiveNumber = Annotated[_StrictFiniteNumber, Field(gt=0)]
_StrictFiniteNonNegativeNumber = Annotated[_StrictFiniteNumber, Field(ge=0)]
_StrictClosedUnitNumber = Annotated[_StrictFiniteNumber, Field(ge=0, le=1)]
_StrictHalfOpenUnitNumber = Annotated[_StrictFiniteNumber, Field(ge=0, lt=1)]
_StrictOpenUnitNumber = Annotated[_StrictFiniteNumber, Field(gt=0, lt=1)]
