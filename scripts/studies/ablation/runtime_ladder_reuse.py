"""Sealed-evidence reuse, byte-binding, and staging validation helpers.

Extracted from :mod:`runtime_ladder` (module-budget extraction planned in the
task-21c review, m5). Semantics:

- Reuse requires exact resolved-config equality, except for fields in the
  scoped :data:`runtime_ladder_config.MIGRATED_CONFIG_FIELDS` whitelist,
  where an ABSENT field in pre-field sealed evidence stands in for its
  assumed default only when the current config carries exactly that default
  (task-21c review I2, controller decision).
- Cross-rung dataset byte-binding fails closed on swapped materializations;
  diagnostic rungs bind against the sealed evidence of the nearest preceding
  chain rung sharing their dataset id (task-21c review I1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .runtime_ladder_config import MIGRATED_CONFIG_FIELDS
from .runtime_ladder_gating import LadderControl
from .runtime_ladder_evidence import parse_sealed_rung_evidence
from .runtime_ladder_spec import BridgeLadderSpec, LadderRung
from .runtime_errors import StudyRequestError

__all__ = [
    "assert_dataset_bytes_bound",
    "diagnostic_binding_reference",
    "reuse_existing_evidence",
    "rung_paths",
    "validate_staged_datasets",
]

_EVIDENCE_NAME = "rung_evidence.json"


def _expected_config(rung: LadderRung, seed: int | None) -> dict[str, Any]:
    config = dict(rung.resolved_config)
    if seed is not None:
        config["seed"] = int(seed)
    return config


def _assert_sealed_identity_and_config(
    rung: LadderRung, payload: Mapping[str, Any], evidence_path: Path,
    seed: int | None,
) -> None:
    """Rung-id + resolved-config (migration-aware) checks shared by evidence
    reuse and diagnostic control loading (task-21c review S-2)."""
    if payload["rung_id"] != rung.id:
        raise StudyRequestError(
            f"sealed evidence at {evidence_path} belongs to rung "
            f"{payload['rung_id']!r}, not {rung.id!r}"
        )
    expected = _expected_config(rung, seed)
    stored = payload.get("resolved_config")
    if stored != expected and isinstance(stored, dict):
        # Scoped migration (task-21c I2): an ABSENT whitelisted field in
        # pre-field sealed evidence stands in for its assumed default ONLY
        # when the current config carries exactly that default.
        stored = dict(stored)
        for field, assumed_default in MIGRATED_CONFIG_FIELDS.items():
            if field not in stored and expected.get(field) == assumed_default:
                stored[field] = assumed_default
    if stored != expected:
        raise StudyRequestError(
            f"sealed evidence at {evidence_path} was produced under a "
            "different resolved config; refusing to reuse it"
        )


def reuse_existing_evidence(
    rung: LadderRung,
    evidence_path: Path,
    control: LadderControl,
    seed: int | None,
) -> tuple[dict[str, Any], str]:
    payload, digest = parse_sealed_rung_evidence(evidence_path.read_bytes())
    _assert_sealed_identity_and_config(rung, payload, evidence_path, seed)
    stored_control = payload.get("control")
    if (
        not isinstance(stored_control, Mapping)
        or stored_control.get("rung_id") != control.rung_id
        or stored_control.get("evidence_sha256") != control.evidence_sha256
    ):
        raise StudyRequestError(
            f"sealed evidence at {evidence_path} is linked to control "
            f"{stored_control!r}, but the current passing control is "
            f"{control.rung_id!r} ({control.evidence_sha256}); control "
            "linkage drift fails closed"
        )
    return payload, digest


def rung_paths(rung: LadderRung, datasets_root: Path) -> tuple[Path, Path]:
    base = Path(datasets_root) / rung.dataset
    train, test = base / "train.npz", base / "test.npz"
    missing = [str(path) for path in (train, test) if not path.is_file()]
    if missing:
        raise StudyRequestError(
            f"rung {rung.id} dataset {rung.dataset!r} is not materialized: "
            f"missing {missing}"
        )
    return train, test


def assert_dataset_bytes_bound(
    rung: LadderRung,
    predecessor: tuple[LadderRung, Mapping[str, Any]] | None,
    train_sha256: str,
    test_sha256: str,
) -> None:
    """Cross-rung dataset-byte binding (review round 1, IMPORTANT 3).

    When the recipe step declares the dataset unchanged versus the
    predecessor rung, the consumed NPZ bytes must equal the predecessor's
    sealed evidence — otherwise a swapped materialization would silently turn
    the rung into a two-variable step.
    """
    if predecessor is None:
        return
    previous_rung, previous_payload = predecessor
    if rung.dataset != previous_rung.dataset:
        return
    previous_dataset = previous_payload["dataset"]
    if (train_sha256, test_sha256) != (
        previous_dataset["train_sha256"],
        previous_dataset["test_sha256"],
    ):
        raise StudyRequestError(
            f"rung {rung.id} consumes a different dataset realization than "
            f"predecessor {previous_rung.id}'s sealed evidence although the "
            "recipe step declares the dataset unchanged "
            f"(train {train_sha256} vs {previous_dataset['train_sha256']}, "
            f"test {test_sha256} vs {previous_dataset['test_sha256']}); "
            "swapped materializations fail closed"
        )


def control_from_sealed_evidence(
    control_rung: LadderRung, output_root: Path, seed: int | None
) -> LadderControl:
    """Gate control for a diagnostic declaring ``control_rung`` (rungs 1d/1e).

    Loads the named rung's sealed evidence from the output root and turns it
    into a :class:`LadderControl`, applying the same rung-id and
    (migration-aware) resolved-config checks the chain reuse path performs
    (S-2) — stale evidence from a drifted spec fails closed. The control's
    own control linkage is NOT re-verified here: standalone diagnostic
    selection skips the prior walk, so that linkage state is unavailable by
    design; it was verified when the control's evidence was sealed. A FAILED
    rung is a valid control (the gate measures the DELTA the diagnostic's
    group contributes on top of it), so no verdict re-adjudication applies.
    """
    evidence_path = output_root / control_rung.id / _EVIDENCE_NAME
    if not evidence_path.is_file():
        raise StudyRequestError(
            f"diagnostic control rung {control_rung.id!r} has no sealed "
            f"evidence at {evidence_path}; run the control rung first"
        )
    payload, digest = parse_sealed_rung_evidence(evidence_path.read_bytes())
    _assert_sealed_identity_and_config(
        control_rung, payload, evidence_path, seed
    )
    metrics = payload["metrics"]
    return LadderControl(
        rung_id=control_rung.id,
        amp_ssim=float(metrics["amp_ssim"]),
        phase_ssim=float(metrics["phase_ssim"]),
        evidence_sha256=digest,
    )


def diagnostic_binding_reference(
    spec: BridgeLadderSpec, rung: LadderRung, output_root: Path
) -> tuple[LadderRung, Mapping[str, Any]] | None:
    """Sealed chain evidence a diagnostic rung must byte-bind against (I1).

    The nearest preceding CHAIN rung sharing the diagnostic's dataset id,
    when its sealed evidence exists in the output root. Hash comparison
    only — the evidence is not adjudicated or reused here.
    """
    reference: LadderRung | None = None
    for candidate in spec.rungs:
        if candidate.id == rung.id:
            break
        if not candidate.diagnostic and candidate.dataset == rung.dataset:
            reference = candidate
    if reference is None:
        return None
    evidence_path = output_root / reference.id / _EVIDENCE_NAME
    if not evidence_path.is_file():
        return None
    payload, _ = parse_sealed_rung_evidence(evidence_path.read_bytes())
    return reference, payload


def validate_staged_datasets(
    spec: BridgeLadderSpec, datasets_root: Path
) -> list[str]:
    """Fail-closed validation of every rung dataset's staged pair (no torch).

    Each unique rung dataset must be materialized under
    ``<datasets_root>/<dataset_id>/{train,test}.npz`` and pass the
    recipe-pinned content validation; the returned plan lines carry the
    content fingerprints.
    """
    from .datasets import DatasetError, validate_ladder_npz_pair

    lines: list[str] = []
    seen: set[str] = set()
    for rung in spec.rungs:
        if rung.dataset in seen:
            continue
        seen.add(rung.dataset)
        train, test = rung_paths(rung, datasets_root)
        try:
            materialized = validate_ladder_npz_pair(
                spec.dataset(rung.dataset), train, test
            )
        except DatasetError as error:
            raise StudyRequestError(
                f"staged dataset {rung.dataset!r} is invalid: {error}"
            ) from error
        lines.append(
            f"dataset {rung.dataset} staged "
            f"train_sha256={materialized.train_sha256} "
            f"test_sha256={materialized.test_sha256} "
            f"probe_sha256={materialized.probe_sha256} "
            f"n_train={materialized.n_train} n_test={materialized.n_test}"
        )
    return lines
