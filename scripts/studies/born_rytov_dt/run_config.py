"""BRDT row-schema and per-row configuration.

Defines the first bounded row roster for the four-row preflight
(`classical_born_backprop`, `unet`, `fno_vanilla`, `hybrid_resnet`/
`sru_net`), preserves the historical `born_init_image` preflight contract,
and records explicit row metadata fields (`model`, `training`,
`input_mode`, `dataset_id`, `operator_version`, `row_status`) so the
later bounded preflight can aggregate rows under a shared contract.

Reviewer-binding constraints encoded here:

- direct sinogram input is a separate contract and must not be mixed with
  historical `born_init_image` rows;
- row labels separate model identity from training procedure;
- supervised-plus-Born-consistency rows are NOT relabeled `PINN-only`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

# ----------------------------------------------------------------------
# Row schema constants
# ----------------------------------------------------------------------
SUPPORTED_INPUT_MODES: Tuple[str, ...] = ("born_init_image", "sinogram")
"""Supported BRDT input contracts."""

REJECTED_INPUT_MODES: Tuple[str, ...] = ("direct_sinogram",)
"""Legacy aliases that are still rejected."""

DEFAULT_TRAINING_LABEL: str = "supervised + Born consistency"
"""Default training-procedure label for neural rows."""

CLASSICAL_TRAINING_LABEL: str = "none"
"""Training-procedure label for the classical reference row."""

RELATIVE_PHYSICS_ONLY_TRAINING_LABEL: str = "relative_physics_only"
"""Training-procedure label for the physics-only neural ablation rows."""

ROW_STATUS_VALUES: Tuple[str, ...] = (
    "ready",
    "blocked",
    "feasibility_only",
    "completed",
    "skipped",
)
"""Allowed row-status values. Distinct from ``model`` and ``training``."""

# Internal architecture IDs. The Hybrid-family row's underlying body is
# always ``hybrid_resnet``; ``sru_net`` is a visible paper label, NOT a
# distinct internal architecture. ``RowConfig.model`` is restricted to
# the internal IDs below so the visible row identity (``row_id`` /
# ``paper_label``) and the internal adapter body (``model``) stay
# explicitly distinct.
SUPPORTED_ARCHITECTURES: Tuple[str, ...] = (
    "classical_born_backprop",
    "unet",
    "fno_vanilla",
    "hybrid_resnet",
    "ffno",
)

# Visible row identifiers the bounded preflight may surface. ``sru_net``
# appears here (paper-label form of the Hybrid-family row) but NOT in
# ``SUPPORTED_ARCHITECTURES`` — the underlying adapter body remains
# ``hybrid_resnet``.
SUPPORTED_ROW_IDS: Tuple[str, ...] = (
    "classical_born_backprop",
    "unet",
    "fno_vanilla",
    "hybrid_resnet",
    "sru_net",
    "ffno",
)

HYBRID_FAMILY_ROW_IDS: Tuple[str, ...] = ("hybrid_resnet", "sru_net")
HYBRID_FAMILY_MODEL: str = "hybrid_resnet"


@dataclass(frozen=True)
class RowConfig:
    """Per-row metadata for the bounded BRDT preflight.

    The fields here are exactly the ones the four-row preflight is
    allowed to aggregate over. Adding fields is allowed; renaming or
    removing them is not without an approved follow-up.
    """

    row_id: str
    model: str  # internal architecture ID, one of SUPPORTED_ARCHITECTURES
    training: str  # human-readable training procedure
    input_mode: str  # must be in SUPPORTED_INPUT_MODES
    dataset_id: str  # canonical dataset name, e.g. "brdt128_sparse_fullview_preflight"
    operator_version: str  # operator git SHA or validation report path
    row_status: str = "ready"  # one of ROW_STATUS_VALUES
    paper_label: Optional[str] = None  # visible paper-table label, defaults to model
    blocker_reason: Optional[str] = None  # only populated when row_status == "blocked"
    blocker_message: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.input_mode not in SUPPORTED_INPUT_MODES:
            if self.input_mode in REJECTED_INPUT_MODES:
                raise ValueError(
                    f"BRDT rejects legacy direct-sinogram input_mode={self.input_mode!r}. "
                    "Use input_mode='sinogram' for measured-sinogram model input."
                )
            raise ValueError(
                f"unsupported input_mode={self.input_mode!r}; "
                f"allowed: {SUPPORTED_INPUT_MODES}"
            )
        if self.model not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"unsupported model={self.model!r}; "
                f"allowed: {SUPPORTED_ARCHITECTURES}"
            )
        if self.row_status not in ROW_STATUS_VALUES:
            raise ValueError(
                f"unsupported row_status={self.row_status!r}; "
                f"allowed: {ROW_STATUS_VALUES}"
            )
        # Reviewer-binding: do not call supervised+Born-consistency rows "PINN".
        normalized_training = self.training.lower()
        if "pinn" in normalized_training and "supervised" not in normalized_training:
            raise ValueError(
                "Rows that combine supervised image loss with Born consistency must NOT be "
                f"labeled as PINN-only. Got training={self.training!r}."
            )
        if self.row_status == "blocked" and not self.blocker_reason:
            raise ValueError("blocked rows must record blocker_reason")

    @property
    def visible_label(self) -> str:
        return self.paper_label or self.model

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "row_id": self.row_id,
            "model": self.model,
            "training": self.training,
            "input_mode": self.input_mode,
            "dataset_id": self.dataset_id,
            "operator_version": self.operator_version,
            "row_status": self.row_status,
            "paper_label": self.visible_label,
        }
        if self.blocker_reason:
            payload["blocker_reason"] = self.blocker_reason
        if self.blocker_message:
            payload["blocker_message"] = self.blocker_message
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


REQUIRED_ROW_FIELDS: Tuple[str, ...] = (
    "row_id",
    "model",
    "training",
    "input_mode",
    "dataset_id",
    "operator_version",
    "row_status",
)


def required_row_fields() -> Tuple[str, ...]:
    """Stable row-metadata fields the four-row preflight may rely on."""
    return REQUIRED_ROW_FIELDS


def default_row_roster(
    *,
    dataset_id: str,
    operator_version: str,
    hybrid_label: str = "hybrid_resnet",
    neural_training_label: str = DEFAULT_TRAINING_LABEL,
) -> List[RowConfig]:
    """Return the four-row roster for the bounded BRDT preflight.

    ``hybrid_label`` selects which label is surfaced for the Hybrid-family
    row. Use ``"sru_net"`` to present the row with the manuscript label;
    the internal architecture ID and adapter remain identical.

    ``neural_training_label`` overrides the visible training-procedure label
    for the three neural rows. Use ``DEFAULT_TRAINING_LABEL`` for the
    completed supervised+Born baseline; use
    ``RELATIVE_PHYSICS_ONLY_TRAINING_LABEL`` for the physics-only ablation.
    The classical row's training label remains ``CLASSICAL_TRAINING_LABEL``.
    """
    if hybrid_label not in ("hybrid_resnet", "sru_net"):
        raise ValueError(
            f"hybrid_label must be 'hybrid_resnet' or 'sru_net'; got {hybrid_label!r}"
        )
    rows: List[RowConfig] = [
        RowConfig(
            row_id="classical_born_backprop",
            model="classical_born_backprop",
            training=CLASSICAL_TRAINING_LABEL,
            input_mode="born_init_image",
            dataset_id=dataset_id,
            operator_version=operator_version,
            paper_label="Model-based Born inverse",
        ),
        RowConfig(
            row_id="unet",
            model="unet",
            training=neural_training_label,
            input_mode="born_init_image",
            dataset_id=dataset_id,
            operator_version=operator_version,
            paper_label="U-Net",
        ),
        RowConfig(
            row_id="fno_vanilla",
            model="fno_vanilla",
            training=neural_training_label,
            input_mode="born_init_image",
            dataset_id=dataset_id,
            operator_version=operator_version,
            paper_label="FNO vanilla",
        ),
        RowConfig(
            row_id=hybrid_label,
            model=HYBRID_FAMILY_MODEL,
            training=neural_training_label,
            input_mode="born_init_image",
            dataset_id=dataset_id,
            operator_version=operator_version,
            paper_label="SRU-Net" if hybrid_label == "sru_net" else "Hybrid ResNet",
        ),
    ]
    return rows


def sinogram_input_row_roster(
    *,
    dataset_id: str,
    operator_version: str,
    neural_training_label: str = DEFAULT_TRAINING_LABEL,
) -> List[RowConfig]:
    """Return the current BRDT manuscript roster for sinogram-input rows.

    The learned rows consume the measured complex sinogram. The classical row
    consumes the same measurement through the fixed Born inverse baseline and
    remains a non-learned reference.
    """
    return [
        RowConfig(
            row_id="classical_born_backprop",
            model="classical_born_backprop",
            training=CLASSICAL_TRAINING_LABEL,
            input_mode="sinogram",
            dataset_id=dataset_id,
            operator_version=operator_version,
            paper_label="Model-based Born inverse",
        ),
        RowConfig(
            row_id="ffno",
            model="ffno",
            training=neural_training_label,
            input_mode="sinogram",
            dataset_id=dataset_id,
            operator_version=operator_version,
            paper_label="FFNO",
        ),
        RowConfig(
            row_id="sru_net",
            model=HYBRID_FAMILY_MODEL,
            training=neural_training_label,
            input_mode="sinogram",
            dataset_id=dataset_id,
            operator_version=operator_version,
            paper_label="SRU-Net",
        ),
    ]


def make_blocked_row(
    row_id: str,
    *,
    model: str,
    training: str,
    dataset_id: str,
    operator_version: str,
    blocker_reason: str,
    blocker_message: str,
    paper_label: Optional[str] = None,
    input_mode: str = "born_init_image",
) -> RowConfig:
    """Construct a controlled row-level blocker for an optional dependency.

    ``blocker_reason`` is a short tag (e.g. ``odtbrain_unavailable``,
    ``neuralop_unavailable``, ``cuda_unavailable``) so downstream
    aggregation can reason about cause without parsing prose.
    """
    return RowConfig(
        row_id=row_id,
        model=model,
        training=training,
        input_mode=input_mode,
        dataset_id=dataset_id,
        operator_version=operator_version,
        row_status="blocked",
        paper_label=paper_label,
        blocker_reason=blocker_reason,
        blocker_message=blocker_message,
    )


@dataclass(frozen=True)
class LossWeights:
    """Loss weights for ``supervised + Born consistency`` neural rows."""

    image: float = 1.0
    physics: float = 0.1
    relative_physics: float = 0.1
    tv: float = 1e-5
    positivity: float = 1e-4

    def as_dict(self) -> Dict[str, float]:
        return {
            "image": float(self.image),
            "physics": float(self.physics),
            "relative_physics": float(self.relative_physics),
            "tv": float(self.tv),
            "positivity": float(self.positivity),
        }


# Objective presets. Each preset resolves to (loss_weights, training_label).
# The default preset is the supervised+Born-consistency contract used by the
# completed four-row preflight; ``relative_physics_only`` is the bounded
# physics-only neural ablation contract.
OBJECTIVE_PRESETS: Tuple[str, ...] = ("supervised_plus_born", "relative_physics_only")


def resolve_objective_preset(name: str) -> Tuple[LossWeights, str]:
    """Resolve an objective-preset label to (LossWeights, training_label).

    The preset captures the entire neural-row training objective so the
    preflight manifest, fingerprints, and per-row provenance never silently
    drift between the completed supervised+Born baseline and the physics-only
    ablation.
    """
    if name == "supervised_plus_born":
        return LossWeights(), DEFAULT_TRAINING_LABEL
    if name == "relative_physics_only":
        return (
            LossWeights(
                image=0.0,
                physics=0.0,
                relative_physics=1.0,
                tv=0.0,
                positivity=0.0,
            ),
            RELATIVE_PHYSICS_ONLY_TRAINING_LABEL,
        )
    raise ValueError(
        f"unknown objective preset {name!r}; allowed: {OBJECTIVE_PRESETS}"
    )


# Reasonable architecture defaults for sanity-only adapter construction.
# These are intentionally small so a fast-dev-run/single-batch step is
# cheap. The later bounded preflight may override them.
DEFAULT_ARCH_KWARGS: Dict[str, Dict[str, Any]] = {
    "unet": {"hidden_channels": 16},
    "fno_vanilla": {"hidden_channels": 16, "fno_modes": 8, "fno_blocks": 4},
    "hybrid_resnet": {
        "hidden_channels": 16,
        "fno_modes": 8,
        "fno_blocks": 2,
        "resnet_blocks": 2,
        "downsample_steps": 1,
    },
    # Task-local BRDT FFNO defaults. The body is the factorized Fourier
    # block stack from ``ptycho_torch.generators.ffno_bottleneck`` with
    # only a minimal BRDT output adapter; FFNO has its own internal
    # architecture identity distinct from ``fno_vanilla`` and must NOT
    # be aliased to it.
    "ffno": {
        "hidden_channels": 16,
        "fno_modes": 8,
        "fno_blocks": 4,
        "share_spectral_weights": False,
        "mlp_ratio": 2.0,
    },
}


def get_default_arch_kwargs(arch: str) -> Dict[str, Any]:
    """Return a copy of the default arch kwargs for an architecture."""
    return dict(DEFAULT_ARCH_KWARGS.get(arch, {}))
