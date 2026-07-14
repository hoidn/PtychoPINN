from __future__ import annotations

import math
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import numpy as np

from scripts.studies.ablation.manifest import (
    Comparison,
    FrozenDict,
    Gate,
    ResolvedComparison,
    ResolvedGate,
    RuleApplicability,
)


def _api():
    from scripts.studies.ablation import verdicts

    return verdicts


def _gate(operator: str, **values):
    defaults = Gate(
        "gate",
        FrozenDict(),
        operator,
        "truth_quality.amp_pearson" if operator != "status_count_ge" else None,
        "median" if operator in {"ge", "le"} else None,
        0.5 if operator in {"ge", "le"} else None,
        2 if operator in {"ge", "le", "finite"} else None,
        "success" if operator == "status_count_ge" else None,
        3 if operator == "status_count_ge" else None,
    )
    return ResolvedGate(
        replace(defaults, **values), "synthetic", "arm", RuleApplicability.ACTIVE
    )


def _row(
    seed: int, *, status="success", completion="terminal", value=0.6, arm_id="arm"
):
    api = _api()
    return api.AttemptRow(
        run_id=f"{arm_id}--seed-{seed}",
        arm_id=arm_id,
        dataset_id="synthetic",
        seed=seed,
        status=api.AttemptStatus(status),
        completion=api.CompletionState(completion),
        metrics={"truth_quality.amp_pearson": value},
    )


def test_status_gate_uses_terminal_requested_denominator_and_distinguishes_outcomes():
    api = _api()
    gate = _gate("status_count_ge", threshold=2, requested=3)

    assert (
        api.evaluate_gate(
            gate,
            (_row(1), _row(2), _row(3, status="failed")),
            requested_seeds=(1, 2, 3),
        ).verdict
        is api.Verdict.PASS
    )
    assert (
        api.evaluate_gate(
            gate,
            (_row(1), _row(2, status="failed"), _row(3, status="failed")),
            requested_seeds=(1, 2, 3),
        ).verdict
        is api.Verdict.FAIL
    )
    assert (
        api.evaluate_gate(
            gate,
            (_row(1), _row(2), _row(3, completion="incomplete", status="incomplete")),
            requested_seeds=(1, 2, 3),
        ).verdict
        is api.Verdict.INCONCLUSIVE
    )
    assert (
        api.evaluate_gate(gate, (_row(1), _row(2)), requested_seeds=(1, 2, 3)).verdict
        is api.Verdict.INCONCLUSIVE
    )


@pytest.mark.parametrize(
    ("status", "completion"),
    [
        ("success", "incomplete"),
        ("failed", "incomplete"),
        ("incomplete", "terminal"),
    ],
)
def test_attempt_row_rejects_inconsistent_status_and_completion(status, completion):
    api = _api()

    with pytest.raises(api.VerdictInputError, match="status/completion"):
        _row(1, status=status, completion=completion)


def test_status_gate_defensively_treats_incomplete_status_as_inconclusive():
    api = _api()
    gate = _gate("status_count_ge", threshold=2, requested=3)
    malformed = object.__new__(api.AttemptRow)
    object.__setattr__(malformed, "run_id", "arm--seed-3")
    object.__setattr__(malformed, "arm_id", "arm")
    object.__setattr__(malformed, "dataset_id", "synthetic")
    object.__setattr__(malformed, "seed", 3)
    object.__setattr__(malformed, "status", api.AttemptStatus.INCOMPLETE)
    object.__setattr__(malformed, "completion", api.CompletionState.TERMINAL)
    object.__setattr__(malformed, "metrics", {})

    result = api.evaluate_gate(
        gate, (_row(1), _row(2), malformed), requested_seeds=(1, 2, 3)
    )

    assert result.verdict is api.Verdict.INCONCLUSIVE
    assert result.reason == "missing_or_incomplete_attempt"


def test_status_gate_rejects_duplicate_and_unexpected_requested_rows():
    api = _api()
    gate = _gate("status_count_ge", threshold=2, requested=3)

    with pytest.raises(api.VerdictInputError, match="duplicate"):
        api.evaluate_gate(gate, (_row(1), _row(1), _row(2)), requested_seeds=(1, 2, 3))
    with pytest.raises(api.VerdictInputError, match="unexpected"):
        api.evaluate_gate(gate, (_row(1), _row(2), _row(4)), requested_seeds=(1, 2, 3))
    with pytest.raises(api.VerdictInputError, match="requested"):
        api.evaluate_gate(gate, (_row(1), _row(2), _row(3)), requested_seeds=(1, 2))


def test_numeric_gates_require_status_and_use_successful_only_median():
    api = _api()
    numeric = _gate("ge", threshold=0.55)
    status = _gate("status_count_ge", threshold=2, requested=3)
    rows = (_row(1, value=0.2), _row(2, value=0.8), _row(3, status="failed", value=0.0))
    status_result = api.evaluate_gate(status, rows, requested_seeds=(1, 2, 3))

    assert (
        api.evaluate_gate(numeric, rows, requested_seeds=(1, 2, 3)).verdict
        is api.Verdict.INCONCLUSIVE
    )
    assert (
        api.evaluate_gate(
            numeric, rows, requested_seeds=(1, 2, 3), status_result=status_result
        ).observed
        == 0.5
    )
    assert (
        api.evaluate_gate(
            numeric, rows, requested_seeds=(1, 2, 3), status_result=status_result
        ).verdict
        is api.Verdict.FAIL
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "inconclusive"), ("bad", "inconclusive"), (math.nan, "inconclusive")],
)
def test_numeric_missing_or_nonfinite_operand_is_inconclusive(value, expected):
    api = _api()
    numeric = _gate("ge", min_successful=1)
    status_result = api.GateResult.active("status", api.Verdict.PASS)

    result = api.evaluate_gate(
        numeric,
        (_row(1, value=value),),
        requested_seeds=(1,),
        status_result=status_result,
    )

    assert result.verdict.value == expected


def test_finite_gate_fails_for_explicit_nonfinite_and_requires_all_operands():
    api = _api()
    gate = _gate("finite", min_successful=2)
    status_result = api.GateResult.active("status", api.Verdict.PASS)

    assert (
        api.evaluate_gate(
            gate,
            (_row(1, value=1.0), _row(2, value=math.inf)),
            requested_seeds=(1, 2),
            status_result=status_result,
        ).verdict
        is api.Verdict.FAIL
    )
    assert (
        api.evaluate_gate(
            gate,
            (_row(1, value=1.0), _row(2, value=None)),
            requested_seeds=(1, 2),
            status_result=status_result,
        ).verdict
        is api.Verdict.INCONCLUSIVE
    )


@pytest.mark.parametrize("operator", ("finite", "ge", "le"))
def test_all_successful_fails_missing_or_invalid_operand_on_success(operator):
    api = _api()
    gate = _gate(
        operator,
        aggregation="all_successful",
        threshold=1.0 if operator in {"ge", "le"} else None,
        min_successful=2,
    )
    status_result = api.GateResult.active("status", api.Verdict.PASS)

    result = api.evaluate_gate(
        gate,
        (_row(1, value=1.0), _row(2, value=None)),
        requested_seeds=(1, 2),
        status_result=status_result,
    )

    assert result.verdict is api.Verdict.FAIL
    assert result.reason == "missing_or_invalid_operand"


def test_all_successful_reload_gate_requires_every_success_to_equal_one():
    api = _api()
    gate = _gate(
        "ge",
        metric="stability.reload_allclose",
        aggregation="all_successful",
        threshold=1.0,
        min_successful=2,
    )
    status_result = api.GateResult.active("status", api.Verdict.PASS)
    rows = (
        replace(_row(1, value=1.0), metrics={"stability.reload_allclose": 1.0}),
        replace(_row(2, value=1.0), metrics={"stability.reload_allclose": 0.0}),
    )

    result = api.evaluate_gate(
        gate,
        rows,
        requested_seeds=(1, 2),
        status_result=status_result,
    )

    assert result.verdict is api.Verdict.FAIL
    assert result.observed == 0.0


def test_diagnostic_comparisons_do_not_control_overall_verdict():
    api = _api()
    diagnostic = api.GateResult.active(
        "within_family", api.Verdict.FAIL, category="diagnostic_comparison"
    )

    assert (
        api.aggregate_verdict(
            (api.GateResult.active("mandatory", api.Verdict.PASS), diagnostic)
        )
        is api.Verdict.PASS
    )


def test_dose_cv_uses_sample_standard_deviation_and_rejects_zero_mean():
    api = _api()
    gate = _gate("le", aggregation="cv", threshold=0.3, min_successful=2)
    status_result = api.GateResult.active("status", api.Verdict.PASS)
    rows = (_row(1, value=1.0), _row(2, value=2.0))

    result = api.evaluate_gate(
        gate, rows, requested_seeds=(1, 2), status_result=status_result
    )
    assert result.observed == pytest.approx(math.sqrt(0.5) / 1.5)
    assert result.verdict is api.Verdict.FAIL
    zero = api.evaluate_gate(
        gate,
        (_row(1, value=-1.0), _row(2, value=1.0)),
        requested_seeds=(1, 2),
        status_result=status_result,
    )
    assert zero.verdict is api.Verdict.INCONCLUSIVE


def test_cv_uses_absolute_mean_for_negative_values():
    api = _api()
    gate = _gate("le", aggregation="cv", threshold=0.3, min_successful=2)
    status_result = api.GateResult.active("status", api.Verdict.PASS)

    result = api.evaluate_gate(
        gate,
        (_row(1, value=-1.0), _row(2, value=-2.0)),
        requested_seeds=(1, 2),
        status_result=status_result,
    )

    assert result.observed == pytest.approx(math.sqrt(0.5) / 1.5)
    assert result.verdict is api.Verdict.FAIL


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed", True),
        ("observed", "bad"),
        ("observed", math.nan),
        ("threshold", False),
        ("threshold", math.inf),
    ],
)
def test_gate_result_rejects_invalid_numeric_operands(field, value):
    api = _api()

    with pytest.raises(api.VerdictInputError, match=field):
        api.GateResult.active("gate", api.Verdict.PASS, **{field: value})


@pytest.mark.parametrize("run_ids", [("",), ("valid", "valid"), ("valid", 1)])
def test_gate_result_rejects_invalid_contributing_run_ids(run_ids):
    api = _api()

    with pytest.raises(api.VerdictInputError, match="contributing_run_ids"):
        api.GateResult.active("gate", api.Verdict.PASS, contributing_run_ids=run_ids)


def _comparison(
    threshold: float = 1.5, min_pairs: int = 2, *, diagnostic: bool = False
) -> ResolvedComparison:
    return ResolvedComparison(
        Comparison(
            "ratio",
            FrozenDict(),
            FrozenDict(),
            "paired_ratio_ge",
            "truth_quality.amp_pearson",
            "median",
            threshold,
            min_pairs,
            diagnostic=diagnostic,
        ),
        "synthetic",
        "synthetic",
        "left",
        "right",
        RuleApplicability.ACTIVE,
    )


def test_comparison_pairs_successes_by_seed_and_evaluates_clean_pairs():
    api = _api()
    rows = (
        _row(1, arm_id="left", value=0.8),
        _row(1, arm_id="right", value=0.4),
        _row(2, arm_id="left", value=0.6),
        _row(2, arm_id="right", value=0.3),
    )

    result = api.evaluate_comparison(_comparison(), rows, requested_seeds=(1, 2))

    assert result.verdict is api.Verdict.PASS
    assert result.observed == pytest.approx(2.0)
    assert api.aggregate_verdict((result,)) is api.Verdict.PASS


def test_mandatory_compatibility_floor_fails_low_paired_ratio():
    api = _api()
    rows = (
        _row(1, arm_id="left", value=0.30),
        _row(1, arm_id="right", value=0.60),
        _row(2, arm_id="left", value=0.36),
        _row(2, arm_id="right", value=0.60),
    )

    result = api.evaluate_comparison(
        _comparison(threshold=0.70), rows, requested_seeds=(1, 2)
    )

    assert result.verdict is api.Verdict.FAIL
    assert result.observed == pytest.approx(0.55)
    assert api.aggregate_verdict((result,)) is api.Verdict.FAIL


def test_mandatory_compatibility_floor_passes_paired_ratio_above_point_seven():
    api = _api()
    rows = (
        _row(1, arm_id="left", value=0.56),
        _row(1, arm_id="right", value=0.70),
        _row(2, arm_id="left", value=0.48),
        _row(2, arm_id="right", value=0.60),
    )

    result = api.evaluate_comparison(
        _comparison(threshold=0.70), rows, requested_seeds=(1, 2)
    )

    assert result.verdict is api.Verdict.PASS
    assert result.observed == pytest.approx(0.80)
    assert api.aggregate_verdict((result,)) is api.Verdict.PASS


def test_comparison_skips_terminal_failed_seed_and_scores_remaining_pairs():
    api = _api()
    rows = (
        _row(1, arm_id="left", value=0.8),
        _row(1, arm_id="right", value=0.4),
        _row(2, arm_id="left", value=0.6),
        _row(2, arm_id="right", value=0.3),
        _row(3, arm_id="left", value=0.9),
        _row(3, arm_id="right", status="failed", value=None),
    )

    result = api.evaluate_comparison(_comparison(), rows, requested_seeds=(1, 2, 3))

    assert result.verdict is api.Verdict.PASS
    assert result.observed == pytest.approx(2.0)
    assert set(result.contributing_run_ids) == {
        "left--seed-1",
        "right--seed-1",
        "left--seed-2",
        "right--seed-2",
    }


def test_comparison_missing_or_incomplete_attempt_is_inconclusive():
    api = _api()
    rows = (
        _row(1, arm_id="left", value=0.8),
        _row(1, arm_id="right", value=0.4),
        _row(2, arm_id="left", value=0.6),
        _row(2, arm_id="right", value=0.3),
        _row(3, arm_id="left", value=0.9),
    )

    missing = api.evaluate_comparison(_comparison(), rows, requested_seeds=(1, 2, 3))
    assert missing.verdict is api.Verdict.INCONCLUSIVE
    assert missing.reason == "missing_or_incomplete_attempt"
    assert api.aggregate_verdict((missing,)) is api.Verdict.INCONCLUSIVE

    incomplete = api.evaluate_comparison(
        _comparison(),
        rows + (_row(3, arm_id="right", status="incomplete", completion="incomplete"),),
        requested_seeds=(1, 2, 3),
    )
    assert incomplete.verdict is api.Verdict.INCONCLUSIVE
    assert incomplete.reason == "missing_or_incomplete_attempt"


def _aggregate_with_passing_mandatory(api, diagnostic):
    return api.aggregate_verdict(
        (api.GateResult.active("mandatory", api.Verdict.PASS), diagnostic)
    )


def test_missing_diagnostic_control_makes_aggregate_inconclusive():
    api = _api()
    rows = (
        _row(1, arm_id="left", value=0.8),
        _row(1, arm_id="right", value=0.4),
    )

    diagnostic = api.evaluate_comparison(
        _comparison(diagnostic=True), rows, requested_seeds=(1, 2)
    )

    assert diagnostic.verdict is api.Verdict.INCONCLUSIVE
    assert diagnostic.category == "required_diagnostic_evidence"
    assert _aggregate_with_passing_mandatory(api, diagnostic) is api.Verdict.INCONCLUSIVE


def test_failed_diagnostic_control_makes_aggregate_inconclusive():
    api = _api()
    rows = (
        _row(1, arm_id="left", value=0.8),
        _row(1, arm_id="right", value=0.4),
        _row(2, arm_id="left", value=0.6),
        _row(2, arm_id="right", status="failed", value=None),
    )

    diagnostic = api.evaluate_comparison(
        _comparison(min_pairs=1, diagnostic=True), rows, requested_seeds=(1, 2)
    )

    assert diagnostic.verdict is api.Verdict.INCONCLUSIVE
    assert diagnostic.reason == "failed_matched_control"
    assert diagnostic.category == "required_diagnostic_evidence"
    assert _aggregate_with_passing_mandatory(api, diagnostic) is api.Verdict.INCONCLUSIVE


def test_invalid_diagnostic_operand_makes_aggregate_inconclusive():
    api = _api()
    rows = (
        _row(1, arm_id="left", value=0.8),
        _row(1, arm_id="right", value=0.4),
        _row(2, arm_id="left", value=0.6),
        _row(2, arm_id="right", value=None),
    )

    diagnostic = api.evaluate_comparison(
        _comparison(diagnostic=True), rows, requested_seeds=(1, 2)
    )

    assert diagnostic.verdict is api.Verdict.INCONCLUSIVE
    assert diagnostic.reason == "missing_or_invalid_operand"
    assert diagnostic.category == "required_diagnostic_evidence"
    assert _aggregate_with_passing_mandatory(api, diagnostic) is api.Verdict.INCONCLUSIVE


def test_insufficient_diagnostic_pairs_make_aggregate_inconclusive():
    api = _api()
    rows = (
        _row(1, arm_id="left", value=0.8),
        _row(1, arm_id="right", value=0.4),
        _row(2, arm_id="left", value=0.6),
        _row(2, arm_id="right", value=0.3),
    )

    diagnostic = api.evaluate_comparison(
        _comparison(min_pairs=3, diagnostic=True), rows, requested_seeds=(1, 2)
    )

    assert diagnostic.verdict is api.Verdict.INCONCLUSIVE
    assert diagnostic.reason == "insufficient_pairs"
    assert diagnostic.category == "required_diagnostic_evidence"
    assert _aggregate_with_passing_mandatory(api, diagnostic) is api.Verdict.INCONCLUSIVE


def test_complete_diagnostic_ratio_outcome_does_not_control_aggregate():
    api = _api()
    rows = (
        _row(1, arm_id="left", value=0.3),
        _row(1, arm_id="right", value=0.6),
        _row(2, arm_id="left", value=0.3),
        _row(2, arm_id="right", value=0.6),
    )

    diagnostic = api.evaluate_comparison(
        _comparison(threshold=1.5, diagnostic=True), rows, requested_seeds=(1, 2)
    )

    assert diagnostic.verdict is api.Verdict.FAIL
    assert diagnostic.category == "diagnostic_comparison"
    assert _aggregate_with_passing_mandatory(api, diagnostic) is api.Verdict.PASS


def test_comparison_terminal_failures_below_min_pairs_are_inconclusive():
    api = _api()
    rows = (
        _row(1, arm_id="left", value=0.8),
        _row(1, arm_id="right", value=0.4),
        _row(2, arm_id="left", value=0.6),
        _row(2, arm_id="right", status="failed", value=None),
        _row(3, arm_id="left", status="failed", value=None),
        _row(3, arm_id="right", value=0.3),
    )

    result = api.evaluate_comparison(_comparison(), rows, requested_seeds=(1, 2, 3))

    assert result.verdict is api.Verdict.INCONCLUSIVE
    assert result.reason == "insufficient_pairs"


def test_comparison_negative_denominator_uses_clamped_verbatim_formula():
    api = _api()
    rows = (
        _row(1, arm_id="left", value=0.5),
        _row(1, arm_id="right", value=-0.3),
        _row(2, arm_id="left", value=0.6),
        _row(2, arm_id="right", value=0.3),
    )

    result = api.evaluate_comparison(_comparison(), rows, requested_seeds=(1, 2))

    assert result.verdict is api.Verdict.PASS
    assert result.observed == pytest.approx((0.5 / 1e-12 + 0.6 / 0.3) / 2.0)

    nonfinite = api.evaluate_comparison(
        _comparison(),
        (
            _row(1, arm_id="left", value=0.5),
            _row(1, arm_id="right", value=math.nan),
            _row(2, arm_id="left", value=0.6),
            _row(2, arm_id="right", value=0.3),
        ),
        requested_seeds=(1, 2),
    )
    assert nonfinite.verdict is api.Verdict.INCONCLUSIVE
    assert nonfinite.reason == "missing_or_invalid_operand"


def test_resolved_not_applicable_stays_visible_and_aggregate_ignores_it():
    api = _api()
    gate = _gate("manual_review")
    skipped = replace(
        gate, applicability=RuleApplicability.NOT_APPLICABLE, reason="dataset_kind"
    )
    result = api.evaluate_gate(skipped, (), requested_seeds=())

    assert result.applicability is RuleApplicability.NOT_APPLICABLE
    assert result.verdict is None
    assert result.reason == "dataset_kind"
    assert (
        api.aggregate_verdict(
            (result, api.GateResult.active("numeric", api.Verdict.PASS))
        )
        is api.Verdict.PASS
    )


def test_manual_review_schema_is_strict_and_pending_never_parses_as_completed():
    api = _api()
    payload = {
        "schema_version": api.REVIEW_SCHEMA_VERSION,
        "reviewer": "reviewer@example.test",
        "timestamp": "2026-07-10T12:00:00Z",
        "figure_sha256": "a" * 64,
        "families": {
            family: {
                "decision": "approve",
                "recognizable": True,
                "flat": False,
                "checkerboard": False,
                "mirrored": False,
                "saturation": False,
                "collapse": False,
                "notes": "looks structurally recognizable",
            }
            for family in ("deadleaves", "lines")
        },
    }
    review = api.parse_review(payload)
    assert set(review.families) == {"deadleaves", "lines"}
    assert review.families["deadleaves"].decision is api.ReviewDecision.APPROVE
    with pytest.raises(api.ReviewError, match="approval"):
        api.parse_review(
            {
                **payload,
                "families": {
                    **payload["families"],
                    "lines": {**payload["families"]["lines"], "flat": True},
                },
            }
        )
    with pytest.raises(api.ReviewError, match="unknown"):
        api.parse_review({**payload, "extra": "no"})
    with pytest.raises(api.ReviewError):
        api.parse_review({**payload, "recognizable": 1})
    with pytest.raises(api.ReviewError, match="RFC3339"):
        api.parse_review({**payload, "timestamp": "2026-07-10 12:00:00Z"})
    with pytest.raises(api.ReviewError, match="pending"):
        api.parse_review(
            api.pending_review_template("reconstruction_truth_error_grid.png")
        )


def test_family_manual_gates_require_both_family_reviews_to_pass():
    api = _api()
    payload = {
        "schema_version": api.REVIEW_SCHEMA_VERSION,
        "reviewer": "reviewer@example.test",
        "timestamp": "2026-07-10T12:00:00Z",
        "figure_sha256": "a" * 64,
        "families": {
            family: {
                "decision": "approve" if family == "deadleaves" else "reject",
                "recognizable": True,
                "flat": False,
                "checkerboard": False,
                "mirrored": False,
                "saturation": False,
                "collapse": False,
                "notes": "reviewed",
            }
            for family in ("deadleaves", "lines")
        },
    }
    review = api.parse_review(payload)
    results = tuple(
        api.evaluate_gate(
            _gate(
                "manual_review",
                id=f"{family}_manual",
                target=FrozenDict({"object_family": family}),
            ),
            (),
            requested_seeds=(),
            review=review,
        )
        for family in ("deadleaves", "lines")
    )

    assert [result.verdict for result in results] == [
        api.Verdict.PASS,
        api.Verdict.FAIL,
    ]
    assert api.aggregate_verdict(results) is api.Verdict.FAIL


def test_legacy_flat_approval_cannot_satisfy_two_family_manual_gates():
    api = _api()
    review = api.parse_review(
        {
            "schema_version": api.REVIEW_SCHEMA_VERSION,
            "reviewer": "legacy-reviewer@example.test",
            "timestamp": "2026-07-10T12:00:00Z",
            "figure_sha256": "c" * 64,
            "decision": "approve",
            "recognizable": True,
            "flat": False,
            "checkerboard": False,
            "mirrored": False,
            "saturation": False,
            "collapse": False,
            "notes": "legacy unscoped approval",
        }
    )

    results = tuple(
        api.evaluate_gate(
            _gate(
                "manual_review",
                id=f"{family}_manual",
                target=FrozenDict({"object_family": family}),
            ),
            (),
            requested_seeds=(),
            review=review,
        )
        for family in ("deadleaves", "lines")
    )

    assert review.families == {}
    assert [result.verdict for result in results] == [
        api.Verdict.INCONCLUSIVE,
        api.Verdict.INCONCLUSIVE,
    ]
    assert {result.reason for result in results} == {"missing_family_review"}
    assert api.aggregate_verdict(results) is api.Verdict.INCONCLUSIVE


def test_reject_with_all_failure_flags_false_is_a_valid_review():
    api = _api()
    payload = {
        "schema_version": api.REVIEW_SCHEMA_VERSION,
        "reviewer": "reviewer@example.test",
        "timestamp": "2026-07-10T12:00:00Z",
        "figure_sha256": "b" * 64,
        "families": {
            "deadleaves": {
                "decision": "reject",
                "recognizable": True,
                "flat": False,
                "checkerboard": False,
                "mirrored": False,
                "saturation": False,
                "collapse": False,
                "notes": "rejected for an unlisted defect; see notes",
            },
            "lines": {
                "decision": "approve",
                "recognizable": True,
                "flat": False,
                "checkerboard": False,
                "mirrored": False,
                "saturation": False,
                "collapse": False,
                "notes": "reviewed",
            },
        },
    }

    review = api.parse_review(payload)

    assert review.decision is api.ReviewDecision.REJECT
    assert (
        review.families["deadleaves"].notes
        == "rejected for an unlisted defect; see notes"
    )


def test_missing_capability_not_applicable_gate_is_na_and_excluded_from_aggregate():
    api = _api()
    gate = _gate("ge")
    skipped = replace(
        gate,
        applicability=RuleApplicability.NOT_APPLICABLE,
        reason="missing_capability:dose_sweep",
    )

    result = api.evaluate_gate(skipped, (), requested_seeds=())

    assert result.applicability is RuleApplicability.NOT_APPLICABLE
    assert result.verdict is None
    assert result.reason == "missing_capability:dose_sweep"
    assert (
        api.aggregate_verdict(
            (result, api.GateResult.active("numeric", api.Verdict.PASS))
        )
        is api.Verdict.PASS
    )


def test_verdict_import_does_not_load_training_or_runtime_frameworks():
    code = """
import sys
import scripts.studies.ablation.verdicts
blocked = [name for name in sys.modules if name == 'torch' or name.startswith('tensorflow') or 'train_lightning' in name or name.endswith('.runtime')]
assert not blocked, blocked
print('isolated')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "isolated"


def _integration_bridge_payload(**changes):
    payload = {
        "schema_version": "hybrid_resnet_integration_bridge_v4",
        "bridge_id": "grid_lines_hybrid_resnet_reference",
        "ssim_floor_source": "fixture",
        "ssim_floor_reference_artifact_sha256": None,
        "probe_source_kind": "run1084",
        "loader_kind": "dictionary",
        "raw_probe_archive_sha256": "9f82cb9eb2c5a853764b98c1657b778600c0e90425296a7d1fdc6e8fdb53c906",
        "raw_probe_array_sha256": "de564a3ed5e70118fde70d8b65214ddfb3f00364228ef8b7c61a3f31a56c309a",
        "transform_order": ["pad_extrapolate_complex", "smooth_complex"],
        "probe_smoothing_sigma": 0.5,
        "probe_target_n": 128,
        "transformed_probe_sha256": "eeccb1c92eae6dce36f4102bccda3f814b3eaa16e03e5c805f786edc628d4cd2",
        "probe_normalize": True,
        "probe_scale": 4.0,
        "dictionary_probe_normalization_policy": "legacy_passthrough_config_inactive",
        "dictionary_effective_probe_sha256": "eeccb1c92eae6dce36f4102bccda3f814b3eaa16e03e5c805f786edc628d4cd2",
        "mmap_probe_normalization_policy": "normalize_probe_like_tf_when_enabled",
        "mmap_same_config_equivalent": False,
        "probe_mask": False,
        "probe_mask_tensor_is_none": True,
        "resolved_probe_mask_kind": "identity",
        "resolved_probe_mask_sha256": "3513f7981d1ead42bde088485afbba413d675b83c061c6362634bcb044bf8613",
        "probe_mask_sigma": 1.0,
        "probe_mask_diameter": None,
        "model_edge_pad": 10,
        "grid_lines_size": 392,
        "grid_lines_n": 128,
        "grid_lines_gridsize": 1,
        "grid_lines_offset": 4,
        "outer_offset_train": 8,
        "outer_offset_test": 20,
        "nimgs_train": 2,
        "nimgs_test": 1,
        "historical_crop_border": 59,
        "historical_effective_patch_n": 10,
        "generic_position_crop_border": 59,
        "generic_effective_patch_n": 10,
        "crop_boundaries_equivalent": True,
        "architecture": "hybrid_resnet",
        "generator_output_mode": "real_imag",
        "hybrid_encoder_conv_hidden_scale": 2.0,
        "training_patch_weighting": "central_mask",
        "physics_forward_mode": "amplitude",
        "amplitude_physics_gain": 16.0,
        "torch_loss_mode": "mae",
        "seed": 3,
        "epochs": 5,
        "fixture_amp_mae_max": 0.09668590068817139,
        "fixture_phase_mae_max": 0.15318376669684494,
        "fixture_amp_ssim_min": 0.8508652644013688,
        "fixture_phase_ssim_min": 0.9468665959387648,
    }
    payload.update(changes)
    return payload


def _integration_bridge_evidence_payload(*, contract_changes=None, **changes):
    payload = {
        "schema_version": "hybrid_resnet_integration_bridge_evidence_v3",
        "contract": _integration_bridge_payload(**(contract_changes or {})),
        "checkpoint_sha256": "1" * 64,
        "selected_checkpoint": "artifacts/checkpoints/best.ckpt",
        "train_npz_sha256": "5" * 64,
        "test_npz_sha256": "6" * 64,
        "pre_stitch_patch_sha256": "2" * 64,
        "historical_canvas_sha256": "3" * 64,
        "ground_truth_sha256": "7" * 64,
        "generic_canvas_sha256": "3" * 64,
        "historical_mask_sha256": "4" * 64,
        "generic_mask_sha256": "4" * 64,
        "canvases_equivalent": True,
        "masks_equivalent": True,
        "no_resize_asserted": True,
        "gauge_handling": "declared_none",
        "recorded_differences": [],
        "fixture_amp_mae": 0.08,
        "fixture_phase_mae": 0.12,
        "fixture_amp_ssim": 0.88,
        "fixture_phase_ssim": 0.96,
        "architecture": "hybrid_resnet",
        "generator_output_mode": "real_imag",
        "hybrid_encoder_conv_hidden_scale": 2.0,
        "training_patch_weighting": "central_mask",
        "physics_forward_mode": "amplitude",
        "amplitude_physics_gain": 16.0,
        "torch_loss_mode": "mae",
        "seed": 3,
        "epochs": 5,
    }
    payload.update(changes)
    return payload


def _sealed_bridge_evidence(api, payload):
    return api.IntegrationBridgeEvidence.from_sealed_artifact_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )


def test_reference_performance_qualification_is_a_typed_claim_grade_prerequisite():
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )
    evidence = _sealed_bridge_evidence(api, _integration_bridge_evidence_payload())

    result = api.evaluate_integration_bridge(requirement, evidence)

    assert result.verdict is api.Verdict.PASS
    assert result.category == "claim_prerequisite"


@pytest.mark.parametrize(
    ("changes", "difference_field"),
    [
        ({"probe_source_kind": "fly"}, "probe_source_kind"),
        ({"loader_kind": "mmap"}, "loader_kind"),
        (
            {"transform_order": ["smooth_complex", "pad_extrapolate_complex"]},
            "transform_order",
        ),
        ({"transformed_probe_sha256": "a" * 64}, "transformed_probe_sha256"),
        ({"probe_mask": True}, "probe_mask"),
        ({"probe_mask_tensor_is_none": False}, "probe_mask_tensor_is_none"),
        ({"resolved_probe_mask_kind": "soft_disk"}, "resolved_probe_mask_kind"),
        ({"probe_mask_sigma": 2.0}, "probe_mask_sigma"),
        ({"probe_mask_diameter": 64.0}, "probe_mask_diameter"),
        (
            {
                "dictionary_probe_normalization_policy": (
                    "normalize_probe_like_tf_when_enabled"
                )
            },
            "dictionary_probe_normalization_policy",
        ),
        ({"dictionary_effective_probe_sha256": "a" * 64},
         "dictionary_effective_probe_sha256"),
        ({"mmap_same_config_equivalent": True}, "mmap_same_config_equivalent"),
        ({"model_edge_pad": 0}, "model_edge_pad"),
        ({"grid_lines_gridsize": 2}, "grid_lines_gridsize"),
        ({"outer_offset_test": 16}, "outer_offset_test"),
        ({"generic_position_crop_border": 32}, "generic_position_crop_border"),
        ({"generic_effective_patch_n": 64}, "generic_effective_patch_n"),
        ({"crop_boundaries_equivalent": False}, "crop_boundaries_equivalent"),
    ],
)
def test_unclassified_provenance_difference_fails_closed(changes, difference_field):
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )
    evidence = _sealed_bridge_evidence(
        api, _integration_bridge_evidence_payload(contract_changes=changes)
    )

    result = api.evaluate_integration_bridge(requirement, evidence)

    assert result.verdict is api.Verdict.FAIL
    assert result.reason == "integration_bridge_unclassified_difference"

    classified = _sealed_bridge_evidence(
        api,
        _integration_bridge_evidence_payload(
            contract_changes=changes,
            recorded_differences=[
                {
                    "field": difference_field,
                    "classification": "harmless",
                    "justification": "surfaced and reviewed as metric-neutral",
                }
            ],
        ),
    )

    assert (
        api.evaluate_integration_bridge(requirement, classified).verdict
        is api.Verdict.PASS
    )


def test_performance_relevant_difference_is_recorded_without_failing():
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )
    evidence = _sealed_bridge_evidence(
        api,
        _integration_bridge_evidence_payload(
            contract_changes={"loader_kind": "mmap"},
            recorded_differences=[
                {
                    "field": "loader_kind",
                    "classification": "performance_relevant",
                    "justification": "different loader; SSIM floors still decide",
                }
            ],
        ),
    )

    result = api.evaluate_integration_bridge(requirement, evidence)

    assert result.verdict is api.Verdict.PASS
    assert result.reason is None


def test_comparison_invalidating_classification_fails_despite_passing_ssim():
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )
    evidence = _sealed_bridge_evidence(
        api,
        _integration_bridge_evidence_payload(
            fixture_amp_ssim=0.99,
            fixture_phase_ssim=0.99,
            contract_changes={"grid_lines_size": 256},
            recorded_differences=[
                {
                    "field": "grid_lines_size",
                    "classification": "comparison_invalidating",
                    "justification": "different object extent breaks fair comparison",
                }
            ],
        ),
    )

    result = api.evaluate_integration_bridge(requirement, evidence)

    assert result.verdict is api.Verdict.FAIL
    assert result.reason == "integration_bridge_comparison_invalidating_difference"


@pytest.mark.parametrize(
    ("changes", "difference_field"),
    [
        (
            {"canvases_equivalent": False, "generic_canvas_sha256": "5" * 64},
            "canvas_equivalence",
        ),
        (
            {"masks_equivalent": False, "generic_mask_sha256": "6" * 64},
            "mask_equivalence",
        ),
    ],
)
def test_canvas_and_mask_divergence_downgrade_to_classified_diagnostics(
    changes, difference_field
):
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )
    unclassified = _sealed_bridge_evidence(
        api, _integration_bridge_evidence_payload(**changes)
    )

    result = api.evaluate_integration_bridge(requirement, unclassified)

    assert result.verdict is api.Verdict.FAIL
    assert result.reason == "integration_bridge_unclassified_difference"

    classified = _sealed_bridge_evidence(
        api,
        _integration_bridge_evidence_payload(
            recorded_differences=[
                {
                    "field": difference_field,
                    "classification": "harmless",
                    "justification": "stitcher bookkeeping differs; metrics agree",
                }
            ],
            **changes,
        ),
    )

    assert (
        api.evaluate_integration_bridge(requirement, classified).verdict
        is api.Verdict.PASS
    )


def test_spurious_difference_classification_fails():
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )
    evidence = _sealed_bridge_evidence(
        api,
        _integration_bridge_evidence_payload(
            recorded_differences=[
                {
                    "field": "loader_kind",
                    "classification": "harmless",
                    "justification": "no such difference was observed",
                }
            ],
        ),
    )

    result = api.evaluate_integration_bridge(requirement, evidence)

    assert result.verdict is api.Verdict.FAIL
    assert result.reason == "integration_bridge_spurious_difference_classification"


def test_command_operand_difference_cannot_be_classified_away():
    api = _api()

    with pytest.raises(api.VerdictInputError, match="seed"):
        api.IntegrationBridgeEvidence.from_mapping(
            _integration_bridge_evidence_payload(
                contract_changes={"seed": 17},
                recorded_differences=[
                    {
                        "field": "seed",
                        "classification": "harmless",
                        "justification": "classification must not excuse the condition",
                    }
                ],
            )
        )


def test_declared_command_condition_violation_fails_regardless_of_metrics():
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )
    evidence = _sealed_bridge_evidence(
        api,
        _integration_bridge_evidence_payload(
            contract_changes={"epochs": 20},
            fixture_amp_ssim=0.99,
            fixture_phase_ssim=0.99,
        ),
    )

    result = api.evaluate_integration_bridge(requirement, evidence)

    assert result.verdict is api.Verdict.FAIL
    assert result.reason == "integration_bridge_command_mismatch"


def test_gate_floor_contract_difference_cannot_be_classified_away():
    api = _api()

    with pytest.raises(api.VerdictInputError, match="fixture_amp_ssim_min"):
        api.IntegrationBridgeEvidence.from_mapping(
            _integration_bridge_evidence_payload(
                contract_changes={"fixture_amp_ssim_min": 0.5},
                recorded_differences=[
                    {
                        "field": "fixture_amp_ssim_min",
                        "classification": "harmless",
                        "justification": "classification must not weaken the gate",
                    }
                ],
            )
        )


def test_gate_floor_contract_mismatch_fails_without_classification():
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )
    evidence = _sealed_bridge_evidence(
        api,
        _integration_bridge_evidence_payload(
            contract_changes={"fixture_phase_ssim_min": 0.5},
        ),
    )

    result = api.evaluate_integration_bridge(requirement, evidence)

    assert result.verdict is api.Verdict.FAIL
    assert result.reason == "integration_bridge_gate_contract_mismatch"


@pytest.mark.parametrize(
    ("amp_ssim", "phase_ssim", "verdict"),
    [
        (0.8508652644013688, 0.9468665959387648, "pass"),
        (0.8508652644013687, 0.9468665959387648, "fail"),
        (0.8508652644013688, 0.9468665959387647, "fail"),
        (0.99, 0.93, "fail"),
        (0.80, 0.99, "fail"),
    ],
)
def test_locked_ssim_floors_are_the_primary_gate(amp_ssim, phase_ssim, verdict):
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )
    evidence = _sealed_bridge_evidence(
        api,
        _integration_bridge_evidence_payload(
            fixture_amp_ssim=amp_ssim, fixture_phase_ssim=phase_ssim
        ),
    )

    result = api.evaluate_integration_bridge(requirement, evidence)

    assert result.verdict.value == verdict
    if verdict == "fail":
        assert result.reason == "integration_bridge_ssim_floor_failed"


@pytest.mark.parametrize(
    ("changes", "verdict"),
    [
        ({"fixture_amp_mae": 0.09668590068817139}, "pass"),
        ({"fixture_amp_mae": 0.0966859006881714}, "fail"),
        ({"fixture_phase_mae": 0.15318376669684494}, "pass"),
        ({"fixture_phase_mae": 0.15318376669684497}, "fail"),
    ],
)
def test_mae_remains_a_supporting_guard(changes, verdict):
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )
    evidence = _sealed_bridge_evidence(
        api, _integration_bridge_evidence_payload(**changes)
    )

    result = api.evaluate_integration_bridge(requirement, evidence)

    assert result.verdict.value == verdict
    if verdict == "fail":
        assert result.reason == "integration_bridge_mae_guard_failed"


def _frozen_cnn_reference_payloads():
    frozen_artifact = json.dumps(
        {
            "amp_ssim_min": 0.886,
            "phase_ssim_min": 0.928,
            "source": "historical CNN N=128/C=1 grid-lines reference",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    requirement_payload = _integration_bridge_payload(
        bridge_id="grid_lines_cnn_reference",
        ssim_floor_source="frozen_reference_artifact",
        ssim_floor_reference_artifact_sha256=hashlib.sha256(
            frozen_artifact
        ).hexdigest(),
        # Approximate declarations; the frozen artifact locks the finals.
        fixture_amp_ssim_min=0.80,
        fixture_phase_ssim_min=0.85,
    )
    return requirement_payload, frozen_artifact


def test_frozen_artifact_reference_fails_closed_until_floors_are_locked():
    api = _api()
    requirement_payload, _ = _frozen_cnn_reference_payloads()
    requirement = api.IntegrationBridgeRequirement.from_mapping(requirement_payload)
    evidence = _sealed_bridge_evidence(
        api,
        _integration_bridge_evidence_payload(
            contract_changes={
                "bridge_id": requirement_payload["bridge_id"],
                "ssim_floor_source": requirement_payload["ssim_floor_source"],
                "ssim_floor_reference_artifact_sha256": requirement_payload[
                    "ssim_floor_reference_artifact_sha256"
                ],
                "fixture_amp_ssim_min": 0.80,
                "fixture_phase_ssim_min": 0.85,
            },
            fixture_amp_ssim=0.99,
            fixture_phase_ssim=0.99,
        ),
    )

    result = api.evaluate_integration_bridge(requirement, evidence)

    assert result.verdict is api.Verdict.FAIL
    assert result.reason == "integration_bridge_ssim_floors_unlocked"


def test_frozen_artifact_locks_final_floors_not_declared_approximations():
    api = _api()
    requirement_payload, frozen_artifact = _frozen_cnn_reference_payloads()
    requirement = api.IntegrationBridgeRequirement.from_mapping(requirement_payload)
    locked = api.LockedSsimFloors.from_frozen_reference_artifact_bytes(
        frozen_artifact
    )
    contract_changes = {
        "bridge_id": requirement_payload["bridge_id"],
        "ssim_floor_source": requirement_payload["ssim_floor_source"],
        "ssim_floor_reference_artifact_sha256": requirement_payload[
            "ssim_floor_reference_artifact_sha256"
        ],
        "fixture_amp_ssim_min": 0.80,
        "fixture_phase_ssim_min": 0.85,
    }
    # Above the declared approximations but below the locked floors: FAIL.
    between = _sealed_bridge_evidence(
        api,
        _integration_bridge_evidence_payload(
            contract_changes=contract_changes,
            fixture_amp_ssim=0.87,
            fixture_phase_ssim=0.93,
        ),
    )
    result = api.evaluate_integration_bridge(
        requirement, between, locked_ssim_floors=locked
    )
    assert result.verdict is api.Verdict.FAIL
    assert result.reason == "integration_bridge_ssim_floor_failed"

    # At or above the locked floors: PASS.
    passing = _sealed_bridge_evidence(
        api,
        _integration_bridge_evidence_payload(
            contract_changes=contract_changes,
            fixture_amp_ssim=0.886,
            fixture_phase_ssim=0.928,
        ),
    )
    result = api.evaluate_integration_bridge(
        requirement, passing, locked_ssim_floors=locked
    )
    assert result.verdict is api.Verdict.PASS


def test_locked_floors_must_come_from_the_declared_frozen_artifact():
    api = _api()
    requirement_payload, frozen_artifact = _frozen_cnn_reference_payloads()
    requirement = api.IntegrationBridgeRequirement.from_mapping(requirement_payload)
    other_locked = api.LockedSsimFloors.from_frozen_reference_artifact_bytes(
        frozen_artifact + b"\n"
    )
    evidence = _sealed_bridge_evidence(
        api,
        _integration_bridge_evidence_payload(
            contract_changes={
                "bridge_id": requirement_payload["bridge_id"],
                "ssim_floor_source": requirement_payload["ssim_floor_source"],
                "ssim_floor_reference_artifact_sha256": requirement_payload[
                    "ssim_floor_reference_artifact_sha256"
                ],
                "fixture_amp_ssim_min": 0.80,
                "fixture_phase_ssim_min": 0.85,
            },
            fixture_amp_ssim=0.99,
            fixture_phase_ssim=0.99,
        ),
    )

    result = api.evaluate_integration_bridge(
        requirement, evidence, locked_ssim_floors=other_locked
    )

    assert result.verdict is api.Verdict.FAIL
    assert result.reason == "integration_bridge_ssim_floor_artifact_mismatch"


def test_fixture_floor_source_rejects_external_locked_floors():
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )
    locked = api.LockedSsimFloors.from_frozen_reference_artifact_bytes(
        json.dumps({"amp_ssim_min": 0.1, "phase_ssim_min": 0.1}).encode()
    )

    with pytest.raises(api.VerdictInputError, match="fixture"):
        api.evaluate_integration_bridge(
            requirement,
            _sealed_bridge_evidence(api, _integration_bridge_evidence_payload()),
            locked_ssim_floors=locked,
        )


def test_frozen_floor_artifact_bytes_must_declare_valid_floors():
    api = _api()

    with pytest.raises(api.VerdictInputError):
        api.LockedSsimFloors.from_frozen_reference_artifact_bytes(b"not json")
    with pytest.raises(api.VerdictInputError):
        api.LockedSsimFloors.from_frozen_reference_artifact_bytes(
            json.dumps({"amp_ssim_min": 0.9}).encode()
        )
    with pytest.raises(api.VerdictInputError):
        api.LockedSsimFloors.from_frozen_reference_artifact_bytes(
            json.dumps({"amp_ssim_min": 1.5, "phase_ssim_min": 0.9}).encode()
        )


@pytest.mark.parametrize(
    "entry",
    [
        {"field": "loader_kind", "classification": "benign", "justification": "x"},
        {"field": "not_a_contract_field", "classification": "harmless",
         "justification": "x"},
        {"field": "loader_kind", "classification": "harmless", "justification": ""},
        {"field": "loader_kind", "classification": "harmless"},
        {"field": "loader_kind", "classification": "harmless",
         "justification": "x", "extra": True},
    ],
)
def test_recorded_difference_entries_use_a_closed_schema(entry):
    api = _api()

    with pytest.raises(api.VerdictInputError):
        api.IntegrationBridgeEvidence.from_mapping(
            _integration_bridge_evidence_payload(recorded_differences=[entry])
        )


def test_recorded_difference_entries_must_be_unique_per_field():
    api = _api()
    entry = {
        "field": "loader_kind",
        "classification": "harmless",
        "justification": "duplicate",
    }

    with pytest.raises(api.VerdictInputError, match="duplicate"):
        api.IntegrationBridgeEvidence.from_mapping(
            _integration_bridge_evidence_payload(
                recorded_differences=[entry, dict(entry)]
            )
        )


def test_bridge_declaration_cannot_be_reused_as_execution_evidence():
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )

    with pytest.raises(api.VerdictInputError):
        api.IntegrationBridgeEvidence.from_mapping(requirement.to_mapping())


def test_bridge_rejects_unknown_execution_schema_version():
    api = _api()
    payload = _integration_bridge_evidence_payload()
    payload["schema_version"] = "future_or_typo"

    with pytest.raises(api.VerdictInputError, match="schema_version"):
        api.IntegrationBridgeEvidence.from_mapping(payload)


def test_stale_v2_evidence_fails_early_as_superseded():
    api = _api()
    payload = _integration_bridge_evidence_payload()
    payload["schema_version"] = "hybrid_resnet_integration_bridge_evidence_v2"

    with pytest.raises(
        api.VerdictInputError,
        match=r"superseded.*hybrid_resnet_integration_bridge_evidence_v3",
    ):
        api.IntegrationBridgeEvidence.from_mapping(payload)


def test_stale_v3_contract_fails_early_as_superseded():
    api = _api()
    payload = _integration_bridge_payload(
        schema_version="hybrid_resnet_integration_bridge_v3"
    )

    with pytest.raises(
        api.VerdictInputError,
        match=r"superseded.*hybrid_resnet_integration_bridge_v4",
    ):
        api.IntegrationBridgeRequirement.from_mapping(payload)

    with pytest.raises(
        api.VerdictInputError,
        match=r"superseded.*hybrid_resnet_integration_bridge_v4",
    ):
        api.IntegrationBridgeEvidence.from_mapping(
            _integration_bridge_evidence_payload(
                contract_changes={
                    "schema_version": "hybrid_resnet_integration_bridge_v3"
                }
            )
        )


@pytest.mark.parametrize(
    "missing",
    ("transformed_probe_sha256", "raw_probe_archive_sha256", "ssim_floor_source"),
)
def test_missing_required_contract_provenance_field_fails(missing):
    api = _api()
    payload = _integration_bridge_payload()
    del payload[missing]

    with pytest.raises(api.VerdictInputError):
        api.IntegrationBridgeRequirement.from_mapping(payload)


@pytest.mark.parametrize(
    "missing",
    ("checkpoint_sha256", "gauge_handling", "recorded_differences",
     "no_resize_asserted"),
)
def test_missing_required_evidence_field_fails(missing):
    api = _api()
    payload = _integration_bridge_evidence_payload()
    del payload[missing]

    with pytest.raises(api.VerdictInputError):
        api.IntegrationBridgeEvidence.from_mapping(payload)


@pytest.mark.parametrize(
    "field",
    (
        "checkpoint_sha256",
        "pre_stitch_patch_sha256",
        "historical_canvas_sha256",
        "generic_canvas_sha256",
        "historical_mask_sha256",
        "generic_mask_sha256",
    ),
)
def test_bridge_rejects_sentinel_hash_for_every_execution_artifact(field):
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )
    evidence = _sealed_bridge_evidence(
        api, _integration_bridge_evidence_payload(**{field: "0" * 64})
    )

    result = api.evaluate_integration_bridge(requirement, evidence)

    assert result.verdict is api.Verdict.FAIL
    assert result.reason == "integration_bridge_artifact_identity_missing"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"checkpoint_sha256": "0" * 64}, "integration_bridge_artifact_identity_missing"),
        (
            {"pre_stitch_patch_sha256": "0" * 64},
            "integration_bridge_artifact_identity_missing",
        ),
        (
            {"generic_canvas_sha256": "5" * 64},
            "integration_bridge_unclassified_difference",
        ),
        (
            {"generic_mask_sha256": "6" * 64},
            "integration_bridge_unclassified_difference",
        ),
        (
            {"canvases_equivalent": False},
            "integration_bridge_unclassified_difference",
        ),
        ({"masks_equivalent": False}, "integration_bridge_unclassified_difference"),
        ({"no_resize_asserted": False}, "integration_bridge_resize_detected"),
        ({"fixture_amp_mae": 0.2}, "integration_bridge_mae_guard_failed"),
        ({"fixture_phase_mae": 0.2}, "integration_bridge_mae_guard_failed"),
        ({"fixture_amp_ssim": 0.5}, "integration_bridge_ssim_floor_failed"),
        ({"fixture_phase_ssim": 0.5}, "integration_bridge_ssim_floor_failed"),
        (
            {"hybrid_encoder_conv_hidden_scale": 1.0},
            "integration_bridge_command_mismatch",
        ),
        ({"architecture": "cnn"}, "integration_bridge_command_mismatch"),
        ({"generator_output_mode": "amp_phase"}, "integration_bridge_command_mismatch"),
        ({"training_patch_weighting": "probe"}, "integration_bridge_command_mismatch"),
        ({"physics_forward_mode": "rectangular_scaled"}, "integration_bridge_command_mismatch"),
        ({"amplitude_physics_gain": 4.0}, "integration_bridge_command_mismatch"),
        ({"torch_loss_mode": "poisson"}, "integration_bridge_command_mismatch"),
        ({"seed": 17}, "integration_bridge_command_mismatch"),
        ({"epochs": 20}, "integration_bridge_command_mismatch"),
    ],
)
def test_bridge_execution_evidence_must_prove_reference_run(changes, reason):
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )
    evidence = _sealed_bridge_evidence(
        api, _integration_bridge_evidence_payload(**changes)
    )

    result = api.evaluate_integration_bridge(requirement, evidence)

    assert result.verdict is api.Verdict.FAIL
    assert result.reason == reason


def test_missing_integration_bridge_keeps_claim_inconclusive():
    api = _api()
    requirement = api.IntegrationBridgeRequirement.from_mapping(
        _integration_bridge_payload()
    )

    result = api.evaluate_integration_bridge(requirement, None)

    assert result.verdict is api.Verdict.INCONCLUSIVE
    assert result.reason == "missing_integration_bridge_evidence"


def test_corrective_manifest_requires_reference_performance_and_bounded_gates():
    from scripts.studies.ablation import manifest
    from scripts.studies.ablation.configuration import resolve_torch_configs

    parsed = manifest.load_manifest(
        Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")
    )
    requirement = parsed.integration_bridge_requirement
    assert requirement is not None
    assert requirement.schema_version == "hybrid_resnet_integration_bridge_v4"
    assert requirement.amplitude_physics_gain == 16.0
    assert requirement.ssim_floor_source == "fixture"
    assert requirement.ssim_floor_reference_artifact_sha256 is None
    assert requirement.fixture_amp_mae_max == pytest.approx(0.09668590068817139)
    assert requirement.fixture_phase_mae_max == pytest.approx(0.15318376669684494)
    assert requirement.fixture_amp_ssim_min == pytest.approx(0.8508652644013688)
    assert requirement.fixture_phase_ssim_min == pytest.approx(0.9468665959387648)
    assert requirement.probe_source_kind == "run1084"
    assert requirement.transform_order == (
        "pad_extrapolate_complex",
        "smooth_complex",
    )
    assert requirement.loader_kind == "dictionary"
    assert requirement.dictionary_probe_normalization_policy == (
        "legacy_passthrough_config_inactive"
    )
    assert requirement.mmap_probe_normalization_policy == (
        "normalize_probe_like_tf_when_enabled"
    )
    assert requirement.mmap_same_config_equivalent is False
    assert requirement.probe_mask is False
    assert requirement.probe_mask_tensor_is_none is True
    assert requirement.resolved_probe_mask_kind == "identity"
    assert requirement.probe_mask_sigma == pytest.approx(1.0)
    assert requirement.probe_mask_diameter is None
    assert requirement.model_edge_pad == 10
    assert (requirement.outer_offset_train, requirement.outer_offset_test) == (8, 20)
    assert requirement.historical_crop_border == 59
    assert requirement.generic_position_crop_border == 59
    assert requirement.historical_effective_patch_n == 10
    assert requirement.generic_effective_patch_n == 10
    assert requirement.crop_boundaries_equivalent is True

    resolved = manifest.resolve_manifest(parsed)
    arms = {
        tuple(arm.dimensions[name] for name in ("object_family", "architecture", "physics_profile"))
        for arm in resolved.arms
    }
    for family in ("deadleaves", "lines"):
        assert (family, "cnn", "legacy_mae") in arms
    assert len(arms) == 12
    for arm in resolved.arms:
        configs = resolve_torch_configs(arm.overrides)
        if arm.dimensions["physics_profile"] == "ci_nll":
            assert arm.overrides["training.torch_loss_mode"] == "poisson"
            assert arm.overrides["inference.varpro_scaling"] is True
            assert configs.ci_scaling_active is True
        else:
            assert configs.ci_scaling_active is False
            assert configs.inference_config.varpro_scaling is False
        if arm.dimensions["physics_profile"] == "legacy_mae":
            assert configs.training_config.torch_loss_mode == "mae"
            assert configs.model_config.physics_forward_mode == "amplitude"

    gates = {gate.id: gate for gate in parsed.gates}
    assert gates["lines_ci_truth_amp_pearson"].threshold == pytest.approx(0.90)
    assert gates["lines_ci_truth_amp_ssim"].threshold == pytest.approx(0.75)
    assert gates["deadleaves_ci_truth_amp_ssim"].threshold >= 0.50
    assert gates["deadleaves_cnn_ci_post_varpro_amp_ssim"].threshold >= 0.50
    assert gates["lines_cnn_ci_post_varpro_amp_ssim"].threshold >= 0.50
    assert all("phase_ssim" not in (gate.metric or "") for gate in parsed.gates)

    comparisons = {comparison.id: comparison for comparison in parsed.comparisons}
    for family in ("deadleaves", "lines"):
        for architecture in ("hybrid", "cnn"):
            comparison = comparisons[
                f"{family}_{architecture}_ci_legacy_mae_amp_ssim_ratio"
            ]
            assert comparison.metric == "truth_quality.amp_ssim"
            assert comparison.operator == "paired_ratio_ge"
            assert comparison.threshold == pytest.approx(0.85)

    for family in ("deadleaves", "lines"):
        for component in ("real", "imag"):
            for rail in ("lower", "upper"):
                gate = gates[f"{family}_cnn_ci_{component}_head_{rail}_saturation"]
                assert gate.threshold == pytest.approx(0.05)
    assert all(
        gate.metric
        not in {
            "stability.real_head_saturation_fraction",
            "stability.imag_head_saturation_fraction",
        }
        for gate in parsed.gates
    )
    bounded_paths = {
        gate.metric
        for gate in gates.values()
        if gate.operator in {"ge", "le"}
    }
    assert {
        "truth_quality.post_varpro.amp_ssim",
        "truth_quality.absolute_amp_nrmse",
        "truth_quality.amp_mean_ratio",
        "truth_quality.amp_quantile_ratio_p05",
        "truth_quality.amp_quantile_ratio_p50",
        "truth_quality.amp_quantile_ratio_p95",
        "stability.scan_utilization_fraction",
        "stability.coverage_fraction",
        "measurement_consistency.model_to_poisson_oracle_error_ratio",
    } <= bounded_paths
    assert parsed.budget_threshold_contract_locked is False


def test_bridge_probe_audit_matches_reference_bytes_and_normalization_paths():
    from ptycho.workflows.grid_lines_workflow import (
        apply_probe_mask,
        apply_probe_transform_pipeline,
        load_probe_guess,
        normalize_probe_transform_pipeline,
    )
    from ptycho_torch import helper
    from ptycho_torch.probe_mask import resolve_probe_mask_np
    from scripts.studies.ablation import manifest

    requirement = manifest.load_manifest(
        Path("scripts/studies/specs/hybrid_resnet_ci_compatibility.toml")
    ).integration_bridge_requirement
    assert requirement is not None
    source = Path("datasets/Run1084_recon3_postPC_shrunk_3.npz")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        requirement.raw_probe_archive_sha256
    )
    raw = np.ascontiguousarray(load_probe_guess(source))
    assert hashlib.sha256(raw.tobytes()).hexdigest() == requirement.raw_probe_array_sha256
    _, steps = normalize_probe_transform_pipeline(
        target_N=requirement.probe_target_n,
        probe_shape=raw.shape,
        probe_scale_mode="pad_extrapolate",
        probe_smoothing_sigma=requirement.probe_smoothing_sigma,
        probe_transform_pipeline=None,
    )
    assert tuple(step["op"] for step in steps) == requirement.transform_order
    transformed = np.ascontiguousarray(apply_probe_transform_pipeline(raw, steps))
    transformed_hash = hashlib.sha256(transformed.tobytes()).hexdigest()
    assert transformed_hash == requirement.transformed_probe_sha256
    assert transformed_hash == requirement.dictionary_effective_probe_sha256
    assert np.array_equal(apply_probe_mask(transformed, None), transformed)

    identity_mask = np.ascontiguousarray(
        resolve_probe_mask_np(
            requirement.probe_target_n,
            probe_mask=requirement.probe_mask,
            probe_mask_tensor=None,
            probe_mask_sigma=requirement.probe_mask_sigma,
            probe_mask_diameter=requirement.probe_mask_diameter,
        )
    )
    assert np.array_equal(identity_mask, np.ones_like(identity_mask))
    assert hashlib.sha256(identity_mask.tobytes()).hexdigest() == (
        requirement.resolved_probe_mask_sha256
    )

    mmap_probe, mmap_scale = helper.normalize_probe_like_tf(
        transformed,
        probe_scale=requirement.probe_scale,
        probe_mask=requirement.probe_mask,
        probe_mask_tensor=None,
        probe_mask_sigma=requirement.probe_mask_sigma,
        probe_mask_diameter=requirement.probe_mask_diameter,
    )
    assert requirement.probe_normalize is True
    assert requirement.loader_kind == "dictionary"
    assert requirement.mmap_same_config_equivalent is False
    assert mmap_scale != pytest.approx(1.0)
    assert not np.array_equal(mmap_probe, transformed)
