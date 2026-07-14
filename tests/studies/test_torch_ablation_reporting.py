from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib.image as mpimg
from matplotlib.collections import PathCollection
from matplotlib.legend import Legend
from matplotlib.patches import ConnectionPatch
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Affine2D, Bbox
import numpy as np
import pytest

from scripts.studies.ablation.metrics import (
    build_metric_record,
    build_image_metric_record,
    build_measurement_metric_record,
)
from scripts.studies.ablation.verdicts import (
    AttemptRow,
    AttemptStatus,
    CompletionState,
    GateResult,
    Verdict,
)
from scripts.studies.ablation.visual_review import parse_review


def _api():
    from scripts.studies.ablation import reporting

    return reporting


def _trajectory_milestone(epoch: int, *, ci: bool = True):
    api = _api()
    values = {
        "stability.validation_loss_final": epoch / 100.0,
        "stability.learning_rate_final": epoch / 100_000.0,
        "truth_quality.post_varpro.amp_ssim": 0.5 + epoch / 1000.0,
        "truth_quality.post_varpro.phase_ssim": 0.4 + epoch / 1000.0,
        "stability.amp_variance": (epoch / 10.0) ** 2,
        "stability.phase_variance": epoch / 200.0,
        "stability.real_head_lower_saturation_fraction": 0.01,
        "stability.real_head_upper_saturation_fraction": 0.02,
        "stability.imag_head_lower_saturation_fraction": 0.03,
        "stability.imag_head_upper_saturation_fraction": 0.04,
    }
    if ci:
        values.update(
            {
                "measurement_consistency.mean_raw_poisson_nll": epoch * 1.5,
                "measurement_consistency.relative_l2_intensity_error": epoch / 50.0,
                "measurement_consistency.varpro.s1": 1.25,
                "measurement_consistency.varpro.s2": 0.5,
            }
        )
    records = tuple(
        SimpleNamespace(path=path, value=value) for path, value in values.items()
    )
    return api.MilestoneEvidence(
        epoch=epoch,
        checkpoint_sha256=f"{epoch:064x}",
        records=records,
        arrays={"reconstruction": np.full((3, 4), epoch, dtype=np.float64)},
    )


def test_trajectory_collation_writes_exact_compact_json_csv_pair(
    tmp_path: Path,
) -> None:
    api = _api()
    milestones = tuple(
        _trajectory_milestone(epoch) for epoch in (5, 20, 40, 80)
    )

    rows = api.collate_milestone_trajectory(milestones)
    json_path, csv_path = api.write_milestone_trajectory(tmp_path, milestones)

    assert [row["epoch"] for row in rows] == [5, 20, 40, 80]
    assert all(tuple(row) == api.MILESTONE_TRAJECTORY_COLUMNS for row in rows)
    assert rows[0]["stitched_amplitude_std"] == pytest.approx(0.5)
    assert rows[0]["cnn_rail_occupancy"] == {
        "real_lower": 0.01,
        "real_upper": 0.02,
        "imag_lower": 0.03,
        "imag_upper": 0.04,
    }
    assert rows[0]["ci_fitted_scales"] == {"s1": 1.25, "s2": 0.5}
    assert json.loads(json_path.read_text(encoding="utf-8")) == rows
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert tuple(csv_rows[0]) == api.MILESTONE_TRAJECTORY_COLUMNS
    assert [int(row["epoch"]) for row in csv_rows] == [5, 20, 40, 80]
    assert len(list(tmp_path.glob("milestone_trajectory.*"))) == 2


def test_trajectory_uses_blank_nonapplicable_ci_and_cnn_fields() -> None:
    api = _api()
    milestone = _trajectory_milestone(5, ci=False)
    records = tuple(
        record
        for record in milestone.records
        if "head_" not in record.path
    )

    row = api.collate_milestone_trajectory(
        (replace(milestone, records=records),)
    )[0]

    assert row["cnn_rail_occupancy"] is None
    assert row["ci_poisson_nll"] is None
    assert row["ci_relative_count_error"] is None
    assert row["ci_fitted_scales"] is None


def test_milestone_grid_renders_four_canonical_columns_and_concise_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api()
    from scripts.studies.ablation import reporting_figures

    milestones = tuple(
        _trajectory_milestone(epoch) for epoch in (5, 20, 40, 80)
    )
    captured: list[object] = []
    original_close = reporting_figures.plt.close
    monkeypatch.setattr(
        reporting_figures.plt,
        "close",
        lambda figure: captured.append(figure),
    )
    identity = SimpleNamespace(
        id="grid-lines--cnn--ci--seed-3",
        arm_id="grid-lines--cnn--ci",
        dataset_id="lines_ci_3p5m",
        seed=3,
    )

    names = api.write_milestone_visuals(tmp_path, identity, milestones)

    assert names == ("milestone_reconstruction_grid.png", "milestone_review.json")
    assert len(captured) == 1
    figure = captured[0]
    assert len(figure.axes) == 4
    assert [axis.get_title() for axis in figure.axes] == [
        "Epoch 5",
        "Epoch 20",
        "Epoch 40",
        "Epoch 80",
    ]
    assert figure.get_facecolor() == pytest.approx((1.0, 1.0, 1.0, 1.0))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    suptitle = figure._suptitle
    assert suptitle is not None
    assert len(suptitle.get_text().splitlines()) <= 2
    assert max(map(len, suptitle.get_text().splitlines())) <= 48
    suptitle_box = suptitle.get_window_extent(renderer)
    for axis in figure.axes:
        epoch_box = axis.title.get_window_extent(renderer)
        assert not suptitle_box.overlaps(epoch_box)
        label_box = axis.title.get_bbox_patch()
        assert label_box is not None
        assert axis.title.get_color() == "black"
        assert label_box.get_facecolor() == pytest.approx((1.0, 1.0, 1.0, 1.0))
    for axis, milestone in zip(figure.axes, milestones):
        np.testing.assert_array_equal(
            axis.images[0].get_array(),
            milestone.arrays["reconstruction"],
        )
    review = json.loads(
        (tmp_path / "milestone_review.json").read_text(encoding="utf-8")
    )
    assert review == {
        "arm_id": identity.arm_id,
        "dataset_id": identity.dataset_id,
        "milestone_epochs": [5, 20, 40, 80],
        "run_id": identity.id,
        "seed": 3,
        "recognizable": "pending",
        "collapsed": "pending",
        "saturated": "pending",
    }
    original_close(figure)


def test_milestone_grid_wraps_long_title_inside_reserved_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.studies.ablation import reporting_figures

    milestones = tuple(
        _trajectory_milestone(epoch) for epoch in (5, 20, 40, 80)
    )
    captured: list[object] = []
    original_close = reporting_figures.plt.close
    monkeypatch.setattr(reporting_figures.plt, "close", captured.append)

    reporting_figures.render_milestone_grid(
        milestones,
        tmp_path / "milestone_reconstruction_grid.png",
        title=(
            "grid-lines--hybrid_resnet--ci_nll--count_intensity--"
            "rectangular_components--seed-3"
        ),
    )

    figure = captured.pop()
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        suptitle = figure._suptitle
        assert suptitle is not None
        title_lines = suptitle.get_text().splitlines()
        assert len(title_lines) == 2
        assert max(map(len, title_lines)) <= 48
        assert suptitle.get_window_extent(renderer).y0 > max(
            axis.title.get_window_extent(renderer).y1 for axis in figure.axes
        )
    finally:
        original_close(figure)


def _family_review_records(notes: str = "human-reviewed") -> dict[str, object]:
    return {
        family: {
            "decision": "approve",
            "recognizable": True,
            "flat": False,
            "checkerboard": False,
            "mirrored": False,
            "saturation": False,
            "collapse": False,
            "notes": notes,
        }
        for family in ("deadleaves", "lines")
    }


def test_review_payload_round_trip_preserves_distinct_family_records() -> None:
    api = _api()
    payload = {
        "schema_version": "visual_review_v1",
        "reviewer": "reviewer@example.test",
        "timestamp": "2026-07-10T12:00:00Z",
        "figure_sha256": "a" * 64,
        "families": {
            "deadleaves": {
                **_family_review_records()["deadleaves"],
                "notes": "Dead Leaves approved",
            },
            "lines": {
                **_family_review_records()["lines"],
                "decision": "reject",
                "notes": "lines rejected",
            },
        },
    }

    review = parse_review(payload)
    serialized = api._review_payload(review)

    assert serialized == payload
    reparsed = parse_review(serialized)
    assert reparsed.families == review.families
    assert reparsed.families["deadleaves"].notes == "Dead Leaves approved"
    assert reparsed.families["lines"].decision.value == "reject"


def _row(run_id: str, arm_id: str, seed: int, *, role: str = "object_truth"):
    api = _api()
    reconstruction = np.asarray([[1.0, 2.0], [3.0, 4.0]]) + seed
    target = np.asarray([[1.5, 2.5], [3.5, 4.5]])
    attempt = AttemptRow(
        run_id=run_id,
        arm_id=arm_id,
        dataset_id="synthetic" if role == "object_truth" else "experimental",
        seed=seed,
        status=AttemptStatus.SUCCESS,
        completion=CompletionState.TERMINAL,
        metrics={"truth_quality.amp_pearson": 0.75},
    )
    image = (
        build_image_metric_record(
            "amp_pearson",
            0.75,
            truth_role=role,
            basis="raw_amplitude",
            alignment="centering",
        )
        if role != "none"
        else None
    )
    records = tuple(
        record
        for record in (
            image,
            build_measurement_metric_record(
                "varpro.s1", 1.2, basis="physical_counts", alignment="none"
            ),
        )
        if record is not None
    )
    return api.ReportRow(
        attempt=attempt,
        truth_role=role,
        reconstruction=reconstruction,
        target=target if role != "none" else None,
        error=reconstruction - target if role != "none" else None,
        training_loss=(3.0, 2.0, 1.0),
        gradient_norm=(0.5, 0.25, 0.125),
        metric_records=records,
        dose_points=((432.0, 1.1), (864.0, 1.2)),
        varpro_scales=(1.0, 1.1),
    )


def _identity(
    run_id: str,
    arm_id: str,
    dataset_id: str,
    seed: int,
    *,
    role: str = "object_truth",
    ci_scaling_active: bool = True,
    object_family: str | None = None,
):
    api = _api()
    capabilities = {"supports_count_metrics"} if ci_scaling_active else set()
    if role == "object_truth":
        capabilities.add("has_object_truth")
    elif role == "reference_reconstruction":
        capabilities.add("has_reference")
    return api.RunIdentity(
        run_id,
        arm_id,
        dataset_id,
        seed,
        truth_role=role,
        capabilities=frozenset(capabilities),
        ci_scaling_active=ci_scaling_active,
        contract_declared=True,
        object_family=object_family,
    )


def _status_record(namespace: str):
    if namespace == "measurement_consistency":
        return build_measurement_metric_record(
            "mean_raw_poisson_nll",
            0.2,
            basis="physical_count_space",
            alignment="none",
        )
    path = {
        "stability": "stability.finite",
        "runtime": "runtime.train_seconds",
    }[namespace]
    return build_metric_record(
        path,
        1.0,
        basis="canonical_reassembly" if namespace == "stability" else "wall_clock",
        alignment="none",
    )


def _namespace_text(study) -> str:
    api = _api()
    return "\n".join(api._namespace_disclosure(study))


def test_report_writes_required_stable_tables_figures_and_pending_review(
    tmp_path: Path,
):
    api = _api()
    rows = (
        _row("run-a-2", "arm-a", 2),
        _row("run-a-1", "arm-a", 1),
        api.ReportRow.failed(
            "run-b-1", "arm-b", "synthetic", 1, stage="training", error="OOM"
        ),
    )
    study = api.ReportInput(
        study_id="compatibility",
        rows=rows,
        requested_runs=(
            api.RunIdentity("run-a-1", "arm-a", "synthetic", 1),
            api.RunIdentity("run-a-2", "arm-a", "synthetic", 2),
            api.RunIdentity("run-b-1", "arm-b", "synthetic", 1),
            api.RunIdentity("run-b-2", "arm-b", "synthetic", 2),
        ),
        gate_results=(
            GateResult.active("numeric", Verdict.PASS),
            GateResult.active("visual", Verdict.INCONCLUSIVE, reason="pending_review"),
        ),
    )

    artifacts = api.write_report(study, tmp_path)

    expected = {
        "report.md",
        "aggregate_metrics.json",
        "aggregate_metrics.csv",
        "arm_seed_status.json",
        "arm_seed_status.csv",
        "verdicts.json",
        "verdicts.csv",
        "figure_row_mapping.json",
        "visual_review.json",
        "report_completion.json",
        "plot_metadata.json",
        "reconstruction_truth_error_grid.png",
        "structural_quality_grid.png",
        "training_gradient_curves.png",
        "seed_distribution.png",
        "varpro_scale.png",
        "absolute_scale_stability_dashboard.png",
        "source_manifest.toml",
        "source_config.json",
        "invocation.json",
        "expansion.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert artifacts.aggregate_verdict is Verdict.INCONCLUSIVE
    for name in sorted(item for item in expected if item.endswith(".png")):
        path = tmp_path / name
        assert path.stat().st_size > 100
        assert mpimg.imread(path).size > 0
    metrics = json.loads((tmp_path / "aggregate_metrics.json").read_text())
    assert [row["run_id"] for row in metrics["rows"]] == [
        "run-a-1",
        "run-a-1",
        "run-a-2",
        "run-a-2",
    ]
    plot_metadata = json.loads((tmp_path / "plot_metadata.json").read_text())
    assert plot_metadata["reconstruction_truth_error_grid.png"]["view"] == "absolute_scale"
    assert plot_metadata["structural_quality_grid.png"]["view"] == "gauge_normalized_structure"
    with (tmp_path / "aggregate_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert (
            list(csv.DictReader(handle))[0]["metric_path"]
            == "measurement_consistency.varpro.s1"
        )
    statuses = json.loads((tmp_path / "arm_seed_status.json").read_text())
    assert any(
        row["status"] == "missing" and row["run_id"] == "run-b-2"
        for row in statuses["rows"]
    )
    pending = json.loads((tmp_path / "visual_review.json").read_text())
    assert pending["state"] == "pending"
    grid = tmp_path / "reconstruction_truth_error_grid.png"
    assert pending["figure_path"] == grid.name
    assert pending["figure_sha256"] == hashlib.sha256(grid.read_bytes()).hexdigest()


def test_report_metadata_proves_shared_row_limits_and_run_mappings(tmp_path: Path):
    api = _api()
    study = api.ReportInput(
        "compatibility",
        (_row("run-a-1", "arm-a", 1),),
        (api.RunIdentity("run-a-1", "arm-a", "synthetic", 1),),
        (),
    )

    api.write_report(study, tmp_path)

    metadata = json.loads((tmp_path / "plot_metadata.json").read_text())
    panels = metadata["reconstruction_truth_error_grid.png"]["panels"]
    limits = {
        panel["panel"]: (panel["vmin"], panel["vmax"])
        for panel in panels
        if panel["run_id"] == "run-a-1"
    }
    assert limits["reconstruction"] == limits["target"]
    mapping = json.loads((tmp_path / "figure_row_mapping.json").read_text())
    assert mapping["reconstruction_truth_error_grid.png"] == ["run-a-1"]


def test_grid_compact_label_prettifies_current_and_future_arm_tokens():
    from scripts.studies.ablation import reporting_figures

    current = _identity(
        "hybrid-resnet-ci-compatibility--ci_3p5m--cnn--ci_nll--seed-3",
        "hybrid-resnet-ci-compatibility--ci_3p5m--cnn--ci_nll",
        "synthetic",
        3,
    )
    future = _identity(
        "study--dataset--spectral_transformer_xl--weighted_huber--seed-42",
        "study--dataset--spectral_transformer_xl--weighted_huber",
        "dataset",
        42,
    )

    assert reporting_figures._compact_grid_label(current) == "CNN | CI NLL | seed 3"
    assert (
        reporting_figures._compact_grid_label(future)
        == "Spectral Transformer XL | Weighted Huber | seed 42"
    )


def test_same_compact_arm_labels_keep_distinct_identity_styles_and_legends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.studies.ablation import reporting_figures

    base = _typed_physics_row()
    arm_ids = (
        "compatibility--variant_alpha--synthetic--cnn--ci_nll",
        "compatibility--variant_beta--synthetic--cnn--ci_nll",
    )
    identities = tuple(
        _identity(
            f"run-duplicate-{index}",
            arm_id,
            "synthetic",
            3,
            object_family="synthetic",
        )
        for index, arm_id in enumerate(arm_ids)
    )
    rows = tuple(
        replace(
            base,
            attempt=replace(
                base.attempt,
                run_id=identity.run_id,
                arm_id=identity.arm_id,
                seed=identity.seed,
            ),
        )
        for identity in identities
    )
    study = _api().ReportInput("duplicate-labels", rows, identities, ())
    registry = reporting_figures._typed_visual_role_registry(study)
    role_ids = {registry[arm_id].visual_role_id for arm_id in arm_ids}
    style_ids = {registry[arm_id].visual_style_id for arm_id in arm_ids}
    assert len(role_ids) == 2
    assert len(style_ids) == 2
    assert {registry[arm_id].display_label for arm_id in arm_ids} == {
        "CNN | CI NLL [Variant Alpha]",
        "CNN | CI NLL [Variant Beta]",
    }
    figures = []
    monkeypatch.setattr(reporting_figures.plt, "close", figures.append)

    reporting_figures.render_varpro(study, tmp_path / "varpro.png")

    figure = figures.pop()
    try:
        figure.canvas.draw()
        arm_legend = next(
            legend
            for legend in figure.findobj(Legend)
            if legend.get_title().get_text() == "CI arms"
        )
        assert _legend_text(arm_legend) == [
            "CNN | CI NLL [Variant Alpha]",
            "CNN | CI NLL [Variant Beta]",
        ]
        axis = next(item for item in figure.axes if item.axison)
        assert len(axis.collections) == 2
        assert (
            len(
                {
                    tuple(collection.get_facecolors()[0])
                    for collection in axis.collections
                }
            )
            == 2
        )
        style_colors, _ = reporting_figures._typed_figure_styles(study, registry)
        assert set(style_colors) == style_ids
    finally:
        reporting_figures.plt.close(figure)


def test_visual_roles_are_global_across_families_and_order_independent() -> None:
    from scripts.studies.ablation import reporting_figures

    study = _canonical_thirty_run_typed_study()
    reordered = replace(
        study,
        rows=tuple(reversed(study.rows)),
        requested_runs=tuple(reversed(study.requested_runs)),
    )

    registry = reporting_figures._typed_visual_role_registry(study)
    reordered_registry = reporting_figures._typed_visual_role_registry(reordered)
    assert registry == reordered_registry
    style_colors, seed_markers = reporting_figures._typed_figure_styles(study, registry)
    reordered_colors, reordered_markers = reporting_figures._typed_figure_styles(
        reordered, reordered_registry
    )
    assert style_colors == reordered_colors
    assert seed_markers == reordered_markers

    by_base_role: dict[str, list[object]] = {}
    for identity in study.requested_runs:
        base_role = reporting_figures._compact_arm_label(identity)
        by_base_role.setdefault(base_role, []).append(registry[identity.arm_id])
    for base_role, entries in by_base_role.items():
        assert {entry.display_label for entry in entries} == {base_role}
        assert len({entry.visual_role_id for entry in entries}) == 1
        assert len({entry.visual_style_id for entry in entries}) == 1


def test_grid_uses_compact_horizontal_left_margin_labels_and_preserves_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from scripts.studies.ablation import reporting_figures

    identities = (
        _identity(
            "hybrid-resnet-ci-compatibility--ci_3p5m--cnn--ci_nll--seed-3",
            "hybrid-resnet-ci-compatibility--ci_3p5m--cnn--ci_nll",
            "synthetic",
            3,
        ),
        _identity(
            "hybrid-resnet-ci-compatibility--ci_3p5m--hybrid_resnet--ci_nll--seed-3",
            "hybrid-resnet-ci-compatibility--ci_3p5m--hybrid_resnet--ci_nll",
            "synthetic",
            3,
        ),
        _identity(
            "hybrid-resnet-ci-compatibility--legacy_amp--hybrid_resnet--legacy_mae--seed-3",
            "hybrid-resnet-ci-compatibility--legacy_amp--hybrid_resnet--legacy_mae",
            "synthetic",
            3,
        ),
        _identity(
            "hybrid-resnet-ci-compatibility--legacy_amp--hybrid_resnet--legacy_nll--seed-3",
            "hybrid-resnet-ci-compatibility--legacy_amp--hybrid_resnet--legacy_nll",
            "synthetic",
            3,
        ),
    )
    rows = tuple(
        _row(identity.run_id, identity.arm_id, identity.seed)
        for identity in identities[:3]
    )
    study = _api().ReportInput("compatibility", rows, identities, ())
    figures = []
    original_close = reporting_figures.plt.close
    monkeypatch.setattr(reporting_figures.plt, "close", figures.append)

    panels, run_ids = reporting_figures.render_grid(
        study, tmp_path / "reconstruction_truth_error_grid.png"
    )
    figure = figures.pop()
    try:
        labels = [text for text in figure.texts if " | seed 3" in text.get_text()]
        assert [text.get_text() for text in labels] == [
            "CNN | CI NLL | seed 3",
            "Hybrid ResNet | CI NLL | seed 3",
            "Hybrid ResNet | Legacy MAE | seed 3",
            "Hybrid ResNet | Legacy NLL | seed 3",
        ]
        assert all(text.get_rotation() == 0.0 for text in labels)
        first_panel_left = min(axis.get_position().x0 for axis in figure.axes)
        assert all(text.get_position()[0] < first_panel_left for text in labels)
        assert all(axis.get_ylabel() == "" for axis in figure.axes)
        panel_text = "\n".join(
            text.get_text() for axis in figure.axes for text in axis.texts
        )
        assert all(identity.run_id not in panel_text for identity in identities)
        assert run_ids == [identity.run_id for identity in identities]
        assert [panel["run_id"] for panel in panels] == [
            identity.run_id for identity in identities for _ in range(3)
        ]
        assert [panel["panel"] for panel in panels] == [
            "reconstruction",
            "target",
            "error",
            "reconstruction",
            "target",
            "error",
            "reconstruction",
            "target",
            "error",
            "reconstruction",
            "target_reference",
            "absolute error",
        ]
    finally:
        original_close(figure)


def test_report_curve_mapping_includes_gradient_only_run(tmp_path: Path):
    api = _api()
    gradient_only = replace(_row("run-a-1", "arm-a", 1), training_loss=())
    both_curves = _row("run-a-2", "arm-a", 2)
    study = api.ReportInput(
        "compatibility",
        (gradient_only, both_curves),
        (
            api.RunIdentity("run-a-1", "arm-a", "synthetic", 1),
            api.RunIdentity("run-a-2", "arm-a", "synthetic", 2),
        ),
        (),
    )

    api.write_report(study, tmp_path)

    mapping = json.loads((tmp_path / "figure_row_mapping.json").read_text())
    assert mapping["training_gradient_curves.png"] == ["run-a-1", "run-a-2"]


def test_one_epoch_training_curve_has_a_visible_raster_mark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.studies.ablation import reporting_figures

    api = _api()
    row = replace(
        _row("run-one-point", "arm-a", 3),
        training_loss=(2.5,),
        gradient_norm=(),
    )
    study = api.ReportInput(
        "compatibility",
        (row,),
        (api.RunIdentity("run-one-point", "arm-a", "synthetic", 3),),
        (),
    )
    figures = []
    original_close = reporting_figures.plt.close
    monkeypatch.setattr(reporting_figures.plt, "close", figures.append)
    path = tmp_path / "training_gradient_curves.png"

    run_ids = reporting_figures.render_curves(study, path)
    figure = figures.pop()
    try:
        raster = mpimg.imread(path)[..., :3]
        axis = figure.axes[0]
        height, width = raster.shape[:2]
        display_x, display_y = axis.transData.transform((0.0, 2.5))
        raster_x = round(display_x * width / figure.bbox.width)
        raster_y = round(height - display_y * height / figure.bbox.height)
        radius = 5
        foreground = raster[
            raster_y - radius : raster_y + radius + 1,
            raster_x - radius : raster_x + radius + 1,
        ]
        background = raster[
            raster_y - radius : raster_y + radius + 1,
            raster_x + 25 - radius : raster_x + 25 + radius + 1,
        ]

        assert run_ids == ["run-one-point"]
        assert foreground.shape == background.shape
        assert np.max(np.abs(foreground - background)) > 0.05
    finally:
        original_close(figure)


def test_loss_only_curve_metadata_does_not_attribute_gradient_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.studies.ablation import reporting_figures

    api = _api()
    run_id = "run-loss-only"
    row = replace(
        _row(run_id, "arm-a", 3),
        training_loss=tuple(float(value) for value in range(20, 0, -1)),
        gradient_norm=(),
    )
    study = api.ReportInput(
        "compatibility",
        (row,),
        (api.RunIdentity(run_id, "arm-a", "synthetic", 3, object_family="lines"),),
        (),
    )
    figures = []
    original_close = reporting_figures.plt.close
    monkeypatch.setattr(reporting_figures.plt, "close", figures.append)

    api.write_report(study, tmp_path)
    try:
        curve_figure = next(
            figure
            for figure in figures
            if any(axis.get_title() == "Lines: Gradient norm" for axis in figure.axes)
        )
        gradient_axis = next(
            axis
            for axis in curve_figure.axes
            if axis.get_title() == "Lines: Gradient norm"
        )
        assert {text.get_text() for text in gradient_axis.texts} == {
            "Not applicable",
            "metric not logged",
        }

        metadata = json.loads((tmp_path / "plot_metadata.json").read_text())
        panels = {
            panel["panel"]: panel
            for panel in metadata["training_gradient_curves.png"]["panels"]
        }
        assert panels["training_loss"] == {
            "object_family": "lines",
            "panel": "training_loss",
            "run_ids": [run_id],
        }
        assert panels["gradient_norm"] == {
            "object_family": "lines",
            "panel": "gradient_norm",
            "run_ids": [],
            "not_applicable_reason": "metric_not_logged",
        }
    finally:
        for figure in figures:
            original_close(figure)


def test_report_mappings_include_only_rows_contributing_to_each_plot(tmp_path: Path):
    api = _api()
    amp_and_scale = _row("run-a-1", "arm-a", 1)
    gradient_only = replace(
        _row("run-a-2", "arm-a", 2), metric_records=(), varpro_scales=()
    )
    neither = replace(
        _row("run-a-3", "arm-a", 3),
        metric_records=(),
        varpro_scales=(),
        gradient_norm=(),
    )
    study = api.ReportInput(
        "compatibility",
        (amp_and_scale, gradient_only, neither),
        tuple(
            api.RunIdentity(f"run-a-{seed}", "arm-a", "synthetic", seed)
            for seed in (1, 2, 3)
        ),
        (),
    )

    api.write_report(study, tmp_path)

    mapping = json.loads((tmp_path / "figure_row_mapping.json").read_text())
    assert mapping["seed_distribution.png"] == ["run-a-1"]
    assert mapping["absolute_scale_stability_dashboard.png"] == []


def test_report_row_arrays_are_defensive_read_only_copies():
    api = _api()
    reconstruction = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    target = np.asarray([[1.5, 2.5], [3.5, 4.5]])
    error = reconstruction - target
    row = api.ReportRow(
        attempt=_row("run-a-1", "arm-a", 1).attempt,
        truth_role="object_truth",
        reconstruction=reconstruction,
        target=target,
        error=error,
    )

    reconstruction[0, 0] = 99.0
    target[0, 0] = 99.0
    error[0, 0] = 99.0

    assert row.reconstruction[0, 0] == 1.0
    assert row.target[0, 0] == 1.5
    assert row.error[0, 0] == -0.5
    with pytest.raises(ValueError):
        row.reconstruction[0, 0] = 0.0


def test_structural_grid_normalizes_and_crops_on_common_valid_mask():
    from scripts.studies.ablation import reporting_figures

    api = _api()
    reconstruction = np.asarray([[100.0, 100.0, 100.0], [100.0, 2.0, 4.0]])
    target = np.asarray([[1.0, 1.0, 1.0], [1.0, 4.0, 8.0]])
    mask = np.asarray([[False, False, False], [False, True, True]])
    row = api.ReportRow(
        attempt=_row("run-a-1", "arm-a", 1).attempt,
        truth_role="object_truth",
        reconstruction=reconstruction,
        target=target,
        error=np.abs(reconstruction - target),
        common_valid_mask=mask,
    )

    displayed_reconstruction, displayed_target, displayed_error = (
        reporting_figures._grid_display_arrays(row, gauge_normalized=True)
    )

    np.testing.assert_allclose(displayed_reconstruction, [[4.0, 8.0]])
    np.testing.assert_allclose(displayed_target, [[4.0, 8.0]])
    np.testing.assert_allclose(displayed_error, [[0.0, 0.0]])


def test_report_rejects_nonstandard_json_numbers(tmp_path: Path):
    api = _api()

    with pytest.raises(api.ReportingError, match="JSON"):
        api._stable_json(tmp_path / "invalid.json", {"value": math.nan})


def test_report_completion_hashes_every_published_artifact(tmp_path: Path):
    api = _api()

    api.write_report(_mini_study(), tmp_path)

    completion = json.loads((tmp_path / "report_completion.json").read_text())
    assert completion["schema_version"] == "ablation_report_completion_v1"
    artifacts = completion["artifacts"]
    assert {item["path"] for item in artifacts} >= {
        "reconstruction_truth_error_grid.png",
        "figure_row_mapping.json",
        "plot_metadata.json",
        "visual_review.json",
    }
    for artifact in artifacts:
        path = tmp_path / artifact["path"]
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_report_failure_leaves_previous_completed_bundle_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    api = _api()
    study = _mini_study()
    api.write_report(study, tmp_path)
    before = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("plot failure")

    monkeypatch.setattr(api, "render_all_figures", fail_render)

    with pytest.raises(api.ReportingError, match="render"):
        api.write_report(study, tmp_path)

    after = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("failure_number", (2, 5))
def test_report_publication_failure_restores_previous_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_number: int
):
    api = _api()
    root = tmp_path / "published"
    study = _mini_study()
    api.write_report(study, root)
    before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    original_replace = api.os.replace
    published = 0

    def fail_later_replace(source, destination):
        nonlocal published
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.parent.name.startswith(f".{root.name}.report-")
            and destination_path.parent == root
            and destination_path.name != "report_completion.json"
        ):
            published += 1
            if published == failure_number:
                raise OSError("injected publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(api.os, "replace", fail_later_replace)

    with pytest.raises(api.ReportingError, match="publication"):
        api.write_report(replace(study, study_id="changed"), root)

    after = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    assert after == before
    completion = json.loads((root / "report_completion.json").read_text())
    for artifact in completion["artifacts"]:
        assert (
            artifact["sha256"]
            == hashlib.sha256((root / artifact["path"]).read_bytes()).hexdigest()
        )
    assert not list(tmp_path.glob(f".{root.name}.report-*"))
    assert not list(tmp_path.glob(f".{root.name}.backup-*"))


def test_same_path_review_recovery_publication_failure_rolls_back_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api()
    root = tmp_path / "same-path-rollback"
    study = _mini_study()
    api.write_report(study, root)
    grid_sha256 = hashlib.sha256(
        (root / "reconstruction_truth_error_grid.png").read_bytes()
    ).hexdigest()
    review_payload = {
        "schema_version": "visual_review_v1",
        "reviewer": "rollback-reviewer",
        "timestamp": "2026-07-11T18:30:00Z",
        "figure_sha256": grid_sha256,
        "families": _family_review_records("rollback candidate"),
    }
    review_path = root / "visual_review.json"
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")
    review_sha256 = hashlib.sha256(review_path.read_bytes()).hexdigest()
    recovering = replace(
        study,
        review=parse_review(review_payload),
        in_place_visual_review_sha256=review_sha256,
    )
    before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    original_replace = api.os.replace
    published = 0

    def fail_second_publication(source, destination):
        nonlocal published
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.parent.name.startswith(f".{root.name}.report-")
            and destination_path.parent == root
            and destination_path.name != "report_completion.json"
        ):
            published += 1
            if published == 2:
                raise OSError("injected same-path publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(api.os, "replace", fail_second_publication)

    with pytest.raises(api.ReportingError, match="publication"):
        api.write_report(recovering, root)

    after = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    assert after == before
    assert not list(tmp_path.glob(f".{root.name}.report-*"))
    assert not list(tmp_path.glob(f".{root.name}.backup-*"))


def test_reporting_rejects_legacy_family_unaware_review_recovery_token(
    tmp_path: Path,
) -> None:
    api = _api()
    root = tmp_path / "legacy-recovery"
    study = _mini_study()
    api.write_report(study, root)
    grid_sha256 = hashlib.sha256(
        (root / "reconstruction_truth_error_grid.png").read_bytes()
    ).hexdigest()
    payload = {
        "schema_version": "visual_review_v1",
        "reviewer": "legacy-reviewer",
        "timestamp": "2026-07-11T19:00:00Z",
        "figure_sha256": grid_sha256,
        "decision": "approve",
        "recognizable": True,
        "flat": False,
        "checkerboard": False,
        "mirrored": False,
        "saturation": False,
        "collapse": False,
        "notes": "legacy family-unaware review",
    }
    review_path = root / "visual_review.json"
    review_path.write_text(json.dumps(payload), encoding="utf-8")
    recovering = replace(
        study,
        review=parse_review(payload),
        in_place_visual_review_sha256=hashlib.sha256(
            review_path.read_bytes()
        ).hexdigest(),
    )

    with pytest.raises(api.ReportingError, match="family-aware"):
        api.write_report(recovering, root)


def test_initial_report_publication_failure_leaves_no_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    api = _api()
    root = tmp_path / "initial"
    original_replace = api.os.replace
    published = 0

    def fail_second_replace(source, destination):
        nonlocal published
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.parent.name.startswith(f".{root.name}.report-")
            and destination_path.parent == root
            and destination_path.name != "report_completion.json"
        ):
            published += 1
            if published == 2:
                raise OSError("injected initial publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(api.os, "replace", fail_second_replace)

    with pytest.raises(api.ReportingError, match="publication"):
        api.write_report(_mini_study(), root)

    assert not (root / "report_completion.json").exists()
    assert not any((root / filename).exists() for filename in api._REPORT_FILENAMES)
    assert not list(tmp_path.glob(f".{root.name}.report-*"))
    assert not list(tmp_path.glob(f".{root.name}.backup-*"))


def test_report_rejects_invalid_staged_tables_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    api = _api()
    study = _mini_study()
    api.write_report(study, tmp_path)
    before = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }

    def write_empty_csv(path, *_args, **_kwargs):
        path.write_text("", encoding="utf-8")

    monkeypatch.setattr(api, "_write_csv", write_empty_csv)

    with pytest.raises(api.ReportingError, match="CSV"):
        api.write_report(study, tmp_path)

    after = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    assert after == before


def test_reviewed_report_rerenders_incomplete_semantic_sidecars(tmp_path: Path):
    api = _api()
    base = _mini_study()
    study = replace(
        base,
        rows=(replace(base.rows[0], source_fingerprint="a" * 64),),
    )
    api.write_report(study, tmp_path)
    grid_sha256 = hashlib.sha256(
        (tmp_path / "reconstruction_truth_error_grid.png").read_bytes()
    ).hexdigest()
    review = parse_review(
        {
            "schema_version": "visual_review_v1",
            "reviewer": "reviewer@example.test",
            "timestamp": "2026-07-10T12:00:00Z",
            "figure_sha256": grid_sha256,
            "families": _family_review_records(),
        }
    )
    (tmp_path / "figure_row_mapping.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "report_completion.json").unlink()

    api.write_report(replace(study, review=review), tmp_path)

    mapping = json.loads((tmp_path / "figure_row_mapping.json").read_text())
    assert (
        mapping["renderer_layout_schema_version"]
        == api.REPORT_RENDERER_LAYOUT_SCHEMA_VERSION
    )
    rerendered_review = json.loads((tmp_path / "visual_review.json").read_text())
    assert rerendered_review["state"] == "pending"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arm_id", "other-arm"),
        ("dataset_id", "other-dataset"),
        ("seed", 2),
    ],
)
def test_report_input_rejects_requested_run_identity_mismatch(field, value):
    api = _api()
    row = _row("run-a-1", "arm-a", 1)
    mismatched = replace(row, attempt=replace(row.attempt, **{field: value}))

    with pytest.raises(api.ReportingError, match=field):
        api.ReportInput(
            "compatibility",
            (mismatched,),
            (api.RunIdentity("run-a-1", "arm-a", "synthetic", 1),),
            (),
        )


def test_report_text_distinguishes_absolute_normalized_and_reference_labels(
    tmp_path: Path,
):
    api = _api()
    study = api.ReportInput(
        "compatibility",
        (
            _row("synthetic-1", "synthetic", 1),
            _row("reference-1", "reference", 1, role="reference_reconstruction"),
        ),
        (
            api.RunIdentity("synthetic-1", "synthetic", "synthetic", 1),
            api.RunIdentity("reference-1", "reference", "experimental", 1),
        ),
        (),
    )

    api.write_report(study, tmp_path)

    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "absolute quantities" in text
    assert "mean-normalized/recognizability quantities" in text
    assert "reference agreement (not truth)" in text
    assert "Failed or missing arms" in text


def test_report_discloses_namespaces_and_requested_seed_accounting(tmp_path: Path):
    api = _api()
    synthetic = replace(
        _row("synthetic-1", "synthetic", 1),
        metric_records=(
            build_image_metric_record(
                "amp_ssim",
                0.8,
                truth_role="object_truth",
                basis="mean_scaled_amplitude",
                alignment="anchor_common_mask_largest_valid_rectangle",
            ),
            build_metric_record(
                "stability.finite",
                1.0,
                basis="canonical_reassembly",
                alignment="none",
            ),
            build_metric_record(
                "runtime.train_seconds",
                3.0,
                basis="wall_clock",
                alignment="none",
            ),
        ),
    )
    experimental = replace(
        _row("experimental-1", "experimental", 1, role="none"),
        metric_records=(),
    )
    study = api.ReportInput(
        "compatibility",
        (
            synthetic,
            api.ReportRow.failed(
                "synthetic-2",
                "synthetic",
                "synthetic",
                2,
                stage="training",
                error="OOM",
            ),
            experimental,
        ),
        (
            _identity(
                "synthetic-1", "synthetic", "synthetic", 1, ci_scaling_active=False
            ),
            _identity(
                "synthetic-2", "synthetic", "synthetic", 2, ci_scaling_active=False
            ),
            _identity(
                "synthetic-3", "synthetic", "synthetic", 3, ci_scaling_active=False
            ),
            _identity(
                "experimental-1",
                "experimental",
                "experimental",
                1,
                role="none",
                ci_scaling_active=False,
            ),
            _identity(
                "experimental-2",
                "experimental",
                "experimental",
                2,
                role="none",
                ci_scaling_active=False,
            ),
        ),
        (),
    )

    api.write_report(study, tmp_path)

    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    for namespace in (
        "truth_quality",
        "reference_agreement",
        "measurement_consistency",
        "stability",
        "runtime",
    ):
        assert f"### {namespace}" in text
    assert "synthetic: AVAILABLE (truth_quality.amp_ssim)" in text
    assert "experimental: NOT_APPLICABLE" in text
    assert (
        "| synthetic | 3 | 1 | 1 | 1 | 3 requested / 1 successful / 1 failed / 1 missing |"
        in text
    )
    assert (
        "| experimental | 2 | 1 | 0 | 1 | 2 requested / 1 successful / 0 failed / 1 missing |"
        in text
    )


def test_failed_truth_arm_reports_applicable_namespaces_as_no_evidence():
    api = _api()
    study = api.ReportInput(
        "compatibility",
        (
            api.ReportRow.failed(
                "truth-1", "truth-arm", "synthetic", 1, stage="training", error="OOM"
            ),
        ),
        (_identity("truth-1", "truth-arm", "synthetic", 1),),
        (),
    )

    text = _namespace_text(study)

    assert "truth-arm: NO_EVIDENCE" in text
    assert "truth-arm: NOT_APPLICABLE" not in text.split("### reference_agreement")[0]
    assert text.count("truth-arm: NO_EVIDENCE") == 4


def test_undeclared_namespace_contract_never_implies_not_applicable():
    api = _api()
    study = api.ReportInput(
        "compatibility",
        (),
        (api.RunIdentity("unknown-1", "unknown-arm", "dataset", 1),),
        (),
    )

    text = _namespace_text(study)

    assert text.count("unknown-arm: NO_EVIDENCE") == 5
    assert "unknown-arm: NOT_APPLICABLE" not in text


def test_undeclared_namespace_contract_with_records_reports_available():
    api = _api()
    row = _row("unknown-1", "unknown-arm", 1)
    study = api.ReportInput(
        "compatibility",
        (row,),
        (api.RunIdentity("unknown-1", "unknown-arm", "synthetic", 1),),
        (),
    )

    text = _namespace_text(study)

    assert "unknown-arm: AVAILABLE (truth_quality.amp_pearson)" in text
    assert "unknown-arm: AVAILABLE (measurement_consistency.varpro.s1)" in text
    assert "unknown-arm: NOT_APPLICABLE" not in text


def test_missing_and_incomplete_arms_report_no_evidence_when_namespace_applies():
    api = _api()
    incomplete = api.ReportRow(
        attempt=AttemptRow(
            "incomplete-1",
            "incomplete-arm",
            "synthetic",
            1,
            AttemptStatus.INCOMPLETE,
            CompletionState.INCOMPLETE,
            {},
        ),
        truth_role="object_truth",
    )
    study = api.ReportInput(
        "compatibility",
        (incomplete,),
        (
            _identity("incomplete-1", "incomplete-arm", "synthetic", 1),
            _identity("missing-1", "missing-arm", "synthetic", 1),
        ),
        (),
    )

    text = _namespace_text(study)

    assert text.count("incomplete-arm: NO_EVIDENCE") == 4
    assert text.count("missing-arm: NO_EVIDENCE") == 4


def test_successful_blind_experimental_arm_routes_image_namespaces_to_na():
    api = _api()
    blind = replace(
        _row("blind-1", "blind-arm", 1, role="none"),
        metric_records=tuple(
            _status_record(namespace)
            for namespace in ("measurement_consistency", "stability", "runtime")
        ),
    )
    study = api.ReportInput(
        "compatibility",
        (blind,),
        (_identity("blind-1", "blind-arm", "experimental", 1, role="none"),),
        (),
    )

    text = _namespace_text(study)

    assert text.count("blind-arm: NOT_APPLICABLE") == 2
    assert text.count("blind-arm: AVAILABLE") == 3
    assert "blind-arm: NO_EVIDENCE" not in text


def test_successful_legacy_arm_marks_measurement_namespace_not_applicable():
    api = _api()
    legacy = replace(
        _row("legacy-1", "legacy-arm", 1),
        metric_records=(
            build_image_metric_record(
                "amp_ssim",
                0.8,
                truth_role="object_truth",
                basis="mean_scaled_amplitude",
                alignment="anchor_common_mask_largest_valid_rectangle",
            ),
            _status_record("stability"),
            _status_record("runtime"),
        ),
    )
    study = api.ReportInput(
        "compatibility",
        (legacy,),
        (
            _identity(
                "legacy-1",
                "legacy-arm",
                "synthetic",
                1,
                ci_scaling_active=False,
            ),
        ),
        (),
    )

    text = _namespace_text(study)

    assert "legacy-arm: AVAILABLE (truth_quality.amp_ssim)" in text
    assert "legacy-arm: NOT_APPLICABLE (legacy/non-CI run contract)" in text
    assert "legacy-arm: NO_EVIDENCE" not in text


def _contradictory_truth_study():
    api = _api()
    blind = replace(
        _row("blind-corrupt-1", "blind-corrupt", 1, role="none"),
        metric_records=(
            build_image_metric_record(
                "amp_ssim",
                0.8,
                truth_role="object_truth",
                basis="mean_scaled_amplitude",
                alignment="anchor_common_mask_largest_valid_rectangle",
            ),
        ),
    )
    return api.ReportInput(
        "compatibility",
        (blind,),
        (
            _identity(
                "blind-corrupt-1",
                "blind-corrupt",
                "experimental",
                1,
                role="none",
            ),
        ),
        (),
    )


def test_truth_record_contradicting_blind_contract_is_rejected():
    api = _api()

    with pytest.raises(
        api.ReportingError,
        match=(
            r"artifact consistency error.*blind-corrupt.*truth_quality\.amp_ssim.*"
            r"declared truth role is none"
        ),
    ):
        api._namespace_disclosure(_contradictory_truth_study())


def test_measurement_record_contradicting_legacy_contract_is_rejected():
    api = _api()
    legacy = replace(
        _row("legacy-corrupt-1", "legacy-corrupt", 1),
        metric_records=(_status_record("measurement_consistency"),),
    )
    study = api.ReportInput(
        "compatibility",
        (legacy,),
        (
            _identity(
                "legacy-corrupt-1",
                "legacy-corrupt",
                "synthetic",
                1,
                ci_scaling_active=False,
            ),
        ),
        (),
    )

    with pytest.raises(
        api.ReportingError,
        match=(
            r"artifact consistency error.*legacy-corrupt.*"
            r"measurement_consistency\.mean_raw_poisson_nll.*"
            r"legacy/non-CI run contract"
        ),
    ):
        api._namespace_disclosure(study)


def test_contradictory_evidence_leaves_no_initial_report_completion_marker(
    tmp_path: Path,
):
    api = _api()
    root = tmp_path / "invalid-report"

    with pytest.raises(api.ReportingError, match="artifact consistency error"):
        api.write_report(_contradictory_truth_study(), root)

    assert not (root / "report_completion.json").exists()
    assert not any((root / filename).exists() for filename in api._REPORT_FILENAMES)
    assert not list(tmp_path.glob(f".{root.name}.report-*"))


def test_contradictory_evidence_preserves_previous_completed_report(
    tmp_path: Path,
):
    api = _api()
    root = tmp_path / "published"
    api.write_report(_mini_study(), root)
    before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}

    with pytest.raises(api.ReportingError, match="artifact consistency error"):
        api.write_report(_contradictory_truth_study(), root)

    after = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    assert after == before
    assert (root / "report_completion.json").is_file()
    assert not list(tmp_path.glob(f".{root.name}.report-*"))


def test_namespace_disclosure_keeps_mixed_arm_states_distinct():
    api = _api()
    available = replace(
        _row("available-1", "available-arm", 1),
        metric_records=(
            build_image_metric_record(
                "amp_ssim",
                0.9,
                truth_role="object_truth",
                basis="mean_scaled_amplitude",
                alignment="anchor_common_mask_largest_valid_rectangle",
            ),
        ),
    )
    study = api.ReportInput(
        "compatibility",
        (available,),
        (
            _identity("available-1", "available-arm", "synthetic", 1),
            _identity("unavailable-1", "unavailable-arm", "synthetic", 1),
            _identity(
                "not-applicable-1",
                "not-applicable-arm",
                "experimental",
                1,
                role="none",
            ),
        ),
        (),
    )

    truth_section = _namespace_text(study).split("### reference_agreement", 1)[0]

    assert "available-arm: AVAILABLE (truth_quality.amp_ssim)" in truth_section
    assert "unavailable-arm: NO_EVIDENCE" in truth_section
    assert "not-applicable-arm: NOT_APPLICABLE" in truth_section


def test_report_arm_accounting_discloses_all_requested_seeds_as_successful(
    tmp_path: Path,
):
    api = _api()
    rows = tuple(
        _row(f"{arm_id}-{seed}", arm_id, seed)
        for arm_id in ("arm-a", "arm-b")
        for seed in (1, 2, 3)
    )
    requested = tuple(
        api.RunIdentity(f"{arm_id}-{seed}", arm_id, "synthetic", seed)
        for arm_id in ("arm-a", "arm-b")
        for seed in (1, 2, 3)
    )

    api.write_report(api.ReportInput("compatibility", rows, requested, ()), tmp_path)

    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert (
        "| arm-a | 3 | 3 | 0 | 0 | 3 requested / 3 successful / 0 failed / 0 missing |"
        in text
    )
    assert (
        "| arm-b | 3 | 3 | 0 | 0 | 3 requested / 3 successful / 0 failed / 0 missing |"
        in text
    )


def test_report_separates_manual_review_using_typed_category_not_rule_id(
    tmp_path: Path,
):
    api = _api()
    study = api.ReportInput(
        "compatibility",
        (_row("run-a-1", "arm-a", 1),),
        (api.RunIdentity("run-a-1", "arm-a", "synthetic", 1),),
        (
            GateResult.active("quality_threshold", Verdict.PASS),
            GateResult.active(
                "human_adjudication", Verdict.INCONCLUSIVE, category="manual_review"
            ),
        ),
    )

    api.write_report(study, tmp_path)

    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    numeric, manual = text.split("## Manual review", maxsplit=1)
    assert "quality_threshold" in numeric
    assert "human_adjudication" not in numeric
    assert "human_adjudication" in manual


def _mini_study():
    api = _api()
    return api.ReportInput(
        "compatibility",
        (_row("run-a-1", "arm-a", 1),),
        (api.RunIdentity("run-a-1", "arm-a", "synthetic", 1),),
        (GateResult.active("numeric", Verdict.PASS),),
    )


def _typed_physics_row(*, gradient_norm: tuple[float, ...] = ()):
    row = _row("run-ci-3", "arm-ci", 3)
    records = (
        build_measurement_metric_record(
            "varpro.s1", 1.2, basis="physical_counts", alignment="none"
        ),
        build_measurement_metric_record(
            "varpro.s2", 0.5, basis="physical_counts", alignment="none"
        ),
        build_measurement_metric_record(
            "varpro.unit_objective", 4.0, basis="physical_counts", alignment="none"
        ),
        build_measurement_metric_record(
            "varpro.fitted_objective", 1.0, basis="physical_counts", alignment="none"
        ),
        build_image_metric_record(
            "amp_mean_ratio",
            0.95,
            truth_role="object_truth",
            basis="raw_amplitude",
            alignment="none",
        ),
        build_image_metric_record(
            "absolute_amp_nrmse",
            0.1,
            truth_role="object_truth",
            basis="raw_amplitude",
            alignment="none",
        ),
        build_metric_record(
            "stability.reload_max_abs_error",
            0.0,
            basis="checkpoint_reload",
            alignment="none",
        ),
    )
    return replace(
        row,
        metric_records=records,
        gradient_norm=gradient_norm,
        varpro_scales=(),
    )


def test_typed_varpro_and_dashboard_evidence_do_not_depend_on_gradient_logging(
    tmp_path: Path,
) -> None:
    from scripts.studies.ablation import reporting_figures

    api = _api()
    row = _typed_physics_row()
    study = api.ReportInput(
        "compatibility",
        (row,),
        (_identity("run-ci-3", "arm-ci", "synthetic", 3),),
        (),
    )
    visual_role = reporting_figures._typed_visual_role_registry(study)["arm-ci"]

    api.write_report(study, tmp_path)

    metadata = json.loads((tmp_path / "plot_metadata.json").read_text())
    varpro = metadata["varpro_scale.png"]
    dashboard = metadata["absolute_scale_stability_dashboard.png"]
    assert varpro["run_ids"] == ["run-ci-3"]
    assert varpro["series"]["run-ci-3"] == pytest.approx(
        {"s1": 1.2, "s2": 0.5, "c_A": 1.3, "c_phi": math.atan2(0.5, 1.2)}
    )
    assert dashboard["run_ids"] == ["run-ci-3"]
    assert dashboard["series"]["run-ci-3"] == pytest.approx(
        {
            "amp_mean_ratio": 0.95,
            "absolute_amp_nrmse": 0.1,
            "varpro_objective_ratio": 0.25,
            "reload_max_abs_error": 0.0,
        }
    )
    assert dashboard["zero_annotations"] == [
        {
            "metric": "reload_max_abs_error",
            "label": "0",
            "object_family": "synthetic",
            "run_id": "run-ci-3",
            "arm_id": "arm-ci",
            "visual_role_id": visual_role.visual_role_id,
            "visual_style_id": visual_role.visual_style_id,
            "arm_display_label": "Arm CI",
            "seed": 3,
            "panel": "physics_reload",
            "renderer_layout_schema_version": "ablation_report_renderer_layout_v2",
            "exact_anchor": {"x": 0.25, "y": 0.0},
            "marker_display_offset_points": {"x": 0.0, "y": 0.0},
            "annotation_slot": 0,
            "annotation_display_offset_points": {"x": -30.0, "y": 12.0},
            "marker_artist_id": "scatter-marker:physics_reload:run-ci-3",
            "connector_id": None,
            "annotation_artist_id": "dashboard-zero-label:run-ci-3",
            "connectors": {
                "anchor_to_marker": None,
                "marker_to_annotation": "dashboard-marker-zero:run-ci-3",
            },
        }
    ]
    mappings = json.loads((tmp_path / "figure_row_mapping.json").read_text())
    assert mappings["varpro_scale.png"] == ["run-ci-3"]
    assert mappings["absolute_scale_stability_dashboard.png"] == ["run-ci-3"]


def test_typed_figures_group_run_ids_by_object_family(tmp_path: Path) -> None:
    api = _api()
    base = _typed_physics_row()
    deadleaves = replace(
        base,
        attempt=replace(
            base.attempt,
            run_id="run-deadleaves-3",
            arm_id="arm-deadleaves-ci",
            dataset_id="deadleaves_ci_3p5m",
        ),
    )
    lines = replace(
        base,
        attempt=replace(
            base.attempt,
            run_id="run-lines-3",
            arm_id="arm-lines-ci",
            dataset_id="lines_ci_3p5m",
        ),
    )
    study = api.ReportInput(
        "compatibility",
        (deadleaves, lines),
        (
            _identity(
                "run-deadleaves-3",
                "arm-deadleaves-ci",
                "deadleaves_ci_3p5m",
                3,
                object_family="deadleaves",
            ),
            _identity(
                "run-lines-3",
                "arm-lines-ci",
                "lines_ci_3p5m",
                3,
                object_family="lines",
            ),
        ),
        (),
    )

    api.write_report(study, tmp_path)

    metadata = json.loads((tmp_path / "plot_metadata.json").read_text())
    expected = {
        "deadleaves": ["run-deadleaves-3"],
        "lines": ["run-lines-3"],
    }
    assert metadata["varpro_scale.png"]["object_family_groups"] == expected
    assert metadata["training_gradient_curves.png"]["object_family_groups"] == expected
    assert (
        metadata["absolute_scale_stability_dashboard.png"]["object_family_groups"]
        == expected
    )
    assert {
        panel["object_family"] for panel in metadata["varpro_scale.png"]["panels"]
    } == {"deadleaves", "lines"}


def test_typed_figures_keep_failed_object_family_visible(tmp_path: Path) -> None:
    api = _api()
    base = _typed_physics_row()
    deadleaves = replace(
        base,
        attempt=replace(
            base.attempt,
            arm_id="arm-deadleaves-ci",
            dataset_id="deadleaves_ci_3p5m",
        ),
    )
    lines = api.ReportRow.failed(
        "run-lines-3",
        "arm-lines-ci",
        "lines_ci_3p5m",
        3,
        stage="training",
        error="failed",
    )
    study = api.ReportInput(
        "compatibility",
        (deadleaves, lines),
        (
            _identity(
                "run-ci-3",
                "arm-deadleaves-ci",
                "deadleaves_ci_3p5m",
                3,
                object_family="deadleaves",
            ),
            _identity(
                "run-lines-3",
                "arm-lines-ci",
                "lines_ci_3p5m",
                3,
                object_family="lines",
            ),
        ),
        (GateResult.active("numeric", Verdict.FAIL),),
    )

    api.write_report(study, tmp_path)

    metadata = json.loads((tmp_path / "plot_metadata.json").read_text())
    lines_panels = [
        panel
        for panel in metadata["varpro_scale.png"]["panels"]
        if panel["object_family"] == "lines"
    ]
    assert lines_panels
    assert {panel["not_applicable_reason"] for panel in lines_panels} == {
        "no_successful_ci_evidence"
    }


def _canonical_thirty_run_typed_study():
    api = _api()
    valid_arms = (
        ("hybrid_resnet", "ci_nll"),
        ("cnn", "ci_nll"),
        ("hybrid_resnet", "legacy_nll"),
        ("hybrid_resnet", "legacy_mae"),
        ("cnn", "legacy_nll"),
    )
    rows = []
    identities = []
    for family in ("deadleaves", "lines"):
        for architecture, profile in valid_arms:
            ci_active = profile == "ci_nll"
            for seed in (3, 17, 29):
                run_id = (
                    "hybrid-resnet-ci-compatibility"
                    f"--{family}_{profile}--{family}--{architecture}--{profile}"
                    f"--seed-{seed}"
                )
                arm_id = (
                    "hybrid-resnet-ci-compatibility"
                    f"--{family}_{profile}--{family}--{architecture}--{profile}"
                )
                base = _typed_physics_row()
                records = (
                    base.metric_records
                    if ci_active
                    else tuple(
                        record
                        for record in base.metric_records
                        if record.path.startswith("truth_quality.")
                    )
                )
                records = (
                    *records,
                    build_image_metric_record(
                        "amp_pearson",
                        0.75,
                        truth_role="object_truth",
                        basis="mean_normalized_amplitude",
                        alignment="centering",
                    ),
                )
                if family == "lines" and architecture == "cnn" and ci_active:
                    real_ratio = {
                        3: 0.19865972746176933,
                        17: 0.1982306835639577,
                        29: 0.19876926598871533,
                    }[seed]
                    records = tuple(
                        replace(record, value=4.0 * real_ratio)
                        if record.path
                        == "measurement_consistency.varpro.fitted_objective"
                        else record
                        for record in records
                    )
                rows.append(
                    replace(
                        base,
                        attempt=replace(
                            base.attempt,
                            run_id=run_id,
                            arm_id=arm_id,
                            dataset_id=f"{family}-{profile}",
                            seed=seed,
                        ),
                        metric_records=records,
                    )
                )
                identities.append(
                    _identity(
                        run_id,
                        arm_id,
                        f"{family}-{profile}",
                        seed,
                        ci_scaling_active=ci_active,
                        object_family=family,
                    )
                )
    return api.ReportInput("canonical", tuple(rows), tuple(identities), ())


def _legend_text(legend: Legend) -> list[str]:
    return [text.get_text() for text in legend.get_texts()]


def _path_signature(collection: object) -> tuple[tuple[float, float], ...]:
    vertices = collection.get_paths()[0].vertices
    return tuple((round(float(x), 5), round(float(y), 5)) for x, y in vertices)


def _marker_display_box(collection: PathCollection) -> Bbox:
    center = collection.get_offset_transform().transform(collection.get_offsets())[0]
    path = collection.get_paths()[0].transformed(
        Affine2D(collection.get_transforms()[0])
    )
    local = path.get_extents()
    return Bbox.from_extents(
        local.x0 + center[0],
        local.y0 + center[1],
        local.x1 + center[0],
        local.y1 + center[1],
    )


def test_pure_scatter_layout_is_collision_selective_and_dpi_independent() -> None:
    from scripts.studies.ablation.reporting_scatter_layout import (
        PanelBounds,
        ScatterLayoutPoint,
        layout_scatter_points,
    )

    def points(dpi: int):
        scale = 72.0 / dpi
        raw_pixels = ((100.0, 100.0), (100.2, 100.1), (100.4, 100.2), (180.0, 140.0))
        return tuple(
            ScatterLayoutPoint(
                run_id=f"run-{index}",
                arm_id=f"arm-{index % 2}",
                visual_role_id=f"role-{index % 2}",
                visual_style_id=f"style-{index % 2}",
                arm_display_label=f"Arm {index % 2}",
                object_family="synthetic",
                panel="s1_s2",
                seed=(3, 17, 29, 3)[index],
                exact_anchor=(float(index), float(index + 1)),
                display_anchor_points=(x * scale, y * scale),
                marker_artist_id=f"marker-{index}",
            )
            for index, (x, y) in enumerate(raw_pixels)
        )

    at_100 = layout_scatter_points(points(100), PanelBounds(0.0, 0.0, 200.0, 160.0))
    at_200 = layout_scatter_points(points(200), PanelBounds(0.0, 0.0, 100.0, 80.0))

    assert [record.marker_display_offset_points for record in at_100[:3]] == [
        record.marker_display_offset_points for record in at_200[:3]
    ]
    assert len({record.display_marker_center_points for record in at_100[:3]}) == 3
    assert all(record.connector_id is not None for record in at_100[:3])
    assert at_100[3].marker_display_offset_points == (0.0, 0.0)
    assert at_100[3].connector_id is None
    assert [record.exact_anchor for record in at_100] == [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 4.0),
    ]
    assert all(
        record.renderer_layout_schema_version == "ablation_report_renderer_layout_v2"
        for record in at_100
    )


def _layout_test_point(run_id: str, center: tuple[float, float]):
    from scripts.studies.ablation.reporting_scatter_layout import ScatterLayoutPoint

    return ScatterLayoutPoint(
        run_id=run_id,
        arm_id=f"arm-{run_id}",
        visual_role_id=f"role-{run_id}",
        visual_style_id=f"style-{run_id}",
        arm_display_label=f"Arm {run_id}",
        object_family="synthetic",
        panel="s1_s2",
        seed=3,
        exact_anchor=center,
        display_anchor_points=center,
        marker_artist_id=f"marker-{run_id}",
    )


def test_scatter_layout_reserves_singleton_before_collision_for_every_order() -> None:
    from scripts.studies.ablation.reporting_scatter_layout import (
        PanelBounds,
        layout_scatter_points,
    )

    points = (
        _layout_test_point("collision-a", (100.0, 100.0)),
        _layout_test_point("collision-b", (100.0, 100.0)),
        _layout_test_point("singleton", (88.0, 88.0)),
    )
    layouts = []
    for permutation in itertools.permutations(points):
        records = layout_scatter_points(
            permutation,
            PanelBounds(0.0, 0.0, 200.0, 200.0),
            marker_half_size_points=5.0,
        )
        by_run = {record.run_id: record for record in records}
        assert by_run["singleton"].marker_display_offset_points == (0.0, 0.0)
        assert by_run["singleton"].display_marker_center_points == (88.0, 88.0)
        assert by_run["singleton"].connector_id is None
        assert len({record.display_marker_center_points for record in records}) == 3
        layouts.append(
            {
                run_id: record.marker_display_offset_points
                for run_id, record in sorted(by_run.items())
            }
        )
    assert all(layout == layouts[0] for layout in layouts)


def test_scatter_layout_avoids_multiple_reserved_singletons() -> None:
    from scripts.studies.ablation.reporting_scatter_layout import (
        PanelBounds,
        layout_scatter_points,
    )

    singletons = tuple(
        _layout_test_point(f"singleton-{index}", center)
        for index, center in enumerate(((88.0, 88.0), (100.0, 88.0), (112.0, 88.0)))
    )
    points = (
        _layout_test_point("collision-a", (100.0, 100.0)),
        _layout_test_point("collision-b", (100.0, 100.0)),
        *singletons,
    )

    records = layout_scatter_points(
        points,
        PanelBounds(0.0, 0.0, 200.0, 200.0),
        marker_half_size_points=5.0,
    )

    by_run = {record.run_id: record for record in records}
    assert all(
        by_run[point.run_id].marker_display_offset_points == (0.0, 0.0)
        for point in singletons
    )
    assert len({record.display_marker_center_points for record in records}) == 5


def test_scatter_layout_reports_exhausted_finite_candidate_space() -> None:
    from scripts.studies.ablation.reporting_scatter_layout import (
        PanelBounds,
        ScatterLayoutError,
        layout_scatter_points,
    )

    points = (
        _layout_test_point("collision-a", (10.0, 10.0)),
        _layout_test_point("collision-b", (10.0, 10.0)),
    )

    with pytest.raises(
        ScatterLayoutError,
        match=(
            "^no collision-free scatter layout exists within panel bounds "
            "and finite candidate offsets$"
        ),
    ):
        layout_scatter_points(
            points,
            PanelBounds(0.0, 0.0, 20.0, 20.0),
            marker_half_size_points=5.0,
        )


@pytest.mark.parametrize(
    ("renderer_name", "expected_arm_counts"),
    (
        ("render_varpro", {"CI arms": 2}),
        (
            "render_dashboard",
            {"Absolute scale arms": 5, "Physics/reload arms": 2},
        ),
    ),
)
def test_typed_figure_legends_are_compact_stable_and_outside_plot_regions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    renderer_name: str,
    expected_arm_counts: dict[str, int],
) -> None:
    from scripts.studies.ablation import reporting_figures

    study = _canonical_thirty_run_typed_study()
    figures = []
    monkeypatch.setattr(reporting_figures.plt, "close", figures.append)
    renderer = getattr(reporting_figures, renderer_name)
    renderer(study, tmp_path / f"{renderer_name}.png")
    figure = figures.pop()
    try:
        figure.canvas.draw()
        legends = figure.findobj(Legend)
        titles = [legend.get_title().get_text() for legend in legends]
        assert titles.count("Seeds") == 2
        for title, count in expected_arm_counts.items():
            matching = [
                legend for legend in legends if legend.get_title().get_text() == title
            ]
            assert len(matching) == 2
            assert all(len(_legend_text(legend)) == count for legend in matching)
        visible_text = [text.get_text() for text in figure.findobj(Text)]
        assert not any(
            identity.run_id in label
            for identity in study.requested_runs
            for label in visible_text
        )
        assert {label for legend in legends for label in _legend_text(legend)} >= {
            "Hybrid ResNet | CI NLL",
            "CNN | CI NLL",
            "seed 3",
            "seed 17",
            "seed 29",
        }

        data_axes = [axis for axis in figure.axes if axis.axison]
        assert len(data_axes) == 4
        renderer_backend = figure.canvas.get_renderer()
        figure_box = figure.bbox
        legend_boxes = [
            legend.get_window_extent(renderer_backend) for legend in legends
        ]
        assert all(
            figure_box.contains(box.x0, box.y0) and figure_box.contains(box.x1, box.y1)
            for box in legend_boxes
        )
        for box in legend_boxes:
            for axis in data_axes:
                protected = (
                    axis.get_window_extent(renderer_backend),
                    axis.title.get_window_extent(renderer_backend),
                    axis.xaxis.label.get_window_extent(renderer_backend),
                    axis.yaxis.label.get_window_extent(renderer_backend),
                )
                assert not any(box.overlaps(region) for region in protected)
        assert not any(
            first.overlaps(second)
            for index, first in enumerate(legend_boxes)
            for second in legend_boxes[index + 1 :]
        )

        for axis in data_axes:
            collections = axis.collections
            panel_box = axis.get_window_extent(renderer_backend)
            for collection in collections:
                center = collection.get_offset_transform().transform(
                    collection.get_offsets()
                )[0]
                assert panel_box.contains(*center)
            by_arm = [
                collections[index : index + 3]
                for index in range(0, len(collections), 3)
            ]
            for arm_points in by_arm:
                assert (
                    len({tuple(point.get_facecolors()[0]) for point in arm_points}) == 1
                )
                assert len({_path_signature(point) for point in arm_points}) == 3
            if len(by_arm) > 1:
                assert len(
                    {tuple(points[0].get_facecolors()[0]) for points in by_arm}
                ) == len(by_arm)
        assert all(
            (collection.get_gid() or "").startswith("scatter-marker:")
            for axis in data_axes
            for collection in axis.collections
        )
    finally:
        reporting_figures.plt.close(figure)


def test_typed_renderers_share_actual_arm_colors_and_seed_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.studies.ablation import reporting_figures

    study = _canonical_thirty_run_typed_study()
    figures = []
    monkeypatch.setattr(reporting_figures.plt, "close", figures.append)
    reporting_figures.render_varpro(study, tmp_path / "varpro.png")
    reporting_figures.render_dashboard(study, tmp_path / "dashboard.png")
    varpro_figure, dashboard_figure = figures
    try:
        family = "deadleaves"
        ci_identities = sorted(
            (
                identity
                for identity in study.requested_runs
                if identity.object_family == family and identity.ci_scaling_active
            ),
            key=lambda identity: identity.run_id,
        )
        all_identities = sorted(
            (
                identity
                for identity in study.requested_runs
                if identity.object_family == family
            ),
            key=lambda identity: identity.run_id,
        )
        varpro_axis = next(axis for axis in varpro_figure.axes if axis.axison)
        dashboard_axis = next(axis for axis in dashboard_figure.axes if axis.axison)

        def styles(axis: object, identities: list[object]):
            colors = {}
            markers = {}
            assert len(axis.collections) == len(identities)
            for collection, identity in zip(axis.collections, identities, strict=True):
                arm_label = reporting_figures._compact_arm_label(identity)
                colors.setdefault(arm_label, tuple(collection.get_facecolors()[0]))
                markers.setdefault(identity.seed, _path_signature(collection))
            return colors, markers

        varpro_colors, varpro_markers = styles(varpro_axis, ci_identities)
        dashboard_colors, dashboard_markers = styles(dashboard_axis, all_identities)
        assert varpro_colors == {
            label: dashboard_colors[label] for label in varpro_colors
        }
        assert varpro_markers == dashboard_markers

        lines_ci = sorted(
            (
                identity
                for identity in study.requested_runs
                if identity.object_family == "lines" and identity.ci_scaling_active
            ),
            key=lambda identity: identity.run_id,
        )
        lines_all = sorted(
            (
                identity
                for identity in study.requested_runs
                if identity.object_family == "lines"
            ),
            key=lambda identity: identity.run_id,
        )
        lines_varpro_colors, _ = styles(varpro_figure.axes[3], lines_ci)
        lines_dashboard_colors, _ = styles(dashboard_figure.axes[3], lines_all)
        assert lines_varpro_colors == varpro_colors
        assert lines_dashboard_colors == dashboard_colors

        legends = [
            legend
            for legend in varpro_figure.findobj(Legend)
            if legend.get_title().get_text() == "CI arms"
        ]
        assert len(legends) == 2
        assert _legend_text(legends[0]) == _legend_text(legends[1])
    finally:
        reporting_figures.plt.close(varpro_figure)
        reporting_figures.plt.close(dashboard_figure)


def test_dashboard_zero_annotations_are_collision_safe_and_marker_associated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.studies.ablation import reporting_figures

    study = _canonical_thirty_run_typed_study()
    figures = []
    monkeypatch.setattr(reporting_figures.plt, "close", figures.append)
    reporting_figures.render_dashboard(study, tmp_path / "dashboard.png")
    figure = figures.pop()
    try:
        figure.canvas.draw()
        backend = figure.canvas.get_renderer()
        data_axes = [axis for axis in figure.axes if axis.axison]
        physics_axes = data_axes[1::2]
        legend_boxes = [
            legend.get_window_extent(backend) for legend in figure.findobj(Legend)
        ]
        assert len(physics_axes) == 2
        for family, axis in zip(("deadleaves", "lines"), physics_axes, strict=True):
            expected_identities = [
                identity
                for identity in study.requested_runs
                if identity.object_family == family and identity.ci_scaling_active
            ]
            expected_identities.sort(
                key=lambda identity: (
                    next(
                        index
                        for index, requested in enumerate(study.requested_runs)
                        if requested.arm_id == identity.arm_id
                    ),
                    identity.seed,
                    identity.run_id,
                )
            )
            markers = [
                collection
                for collection in axis.collections
                if (collection.get_gid() or "").startswith(
                    "scatter-marker:physics_reload:"
                )
            ]
            assert [item.get_gid() for item in markers] == [
                f"scatter-marker:physics_reload:{identity.run_id}"
                for identity in expected_identities
            ]
            marker_boxes = [_marker_display_box(item) for item in markers]
            marker_centers = [
                item.get_offset_transform().transform(item.get_offsets())[0]
                for item in markers
            ]
            assert not any(
                first.overlaps(second)
                for index, first in enumerate(marker_boxes)
                for second in marker_boxes[index + 1 :]
            )
            assert len({tuple(center) for center in marker_centers}) == 6
            for marker, identity in zip(markers, expected_identities, strict=True):
                values = reporting_figures._dashboard_physics_series(study)[
                    identity.run_id
                ]
                assert marker.get_offsets().tolist() == [
                    [values["varpro_objective_ratio"], 0.0]
                ]

            anchor_connectors = [
                patch
                for patch in axis.patches
                if isinstance(patch, ConnectionPatch)
                and (patch.get_gid() or "").startswith(
                    "scatter-anchor-marker:physics_reload:"
                )
            ]
            assert [item.get_gid() for item in anchor_connectors] == [
                f"scatter-anchor-marker:physics_reload:{identity.run_id}"
                for identity in expected_identities
            ]
            annotations = [
                text
                for text in axis.texts
                if isinstance(text, Annotation) and text.get_text() == "0"
            ]
            assert len(annotations) == 6
            assert [item.get_gid() for item in annotations] == [
                f"dashboard-zero-label:{identity.run_id}"
                for identity in expected_identities
            ]
            assert [item.get_position() for item in annotations] == [
                (-30, 12),
                (-18, 26),
                (-8, 40),
                (8, 54),
                (18, 68),
                (30, 82),
            ]
            boxes = [Text.get_window_extent(item, backend) for item in annotations]
            assert not any(
                first.overlaps(second)
                for index, first in enumerate(boxes)
                for second in boxes[index + 1 :]
            )
            panel_box = axis.get_window_extent(backend)
            protected = (
                axis.title.get_window_extent(backend),
                axis.xaxis.label.get_window_extent(backend),
                axis.yaxis.label.get_window_extent(backend),
                *legend_boxes,
            )
            for annotation, box in zip(annotations, boxes, strict=True):
                assert panel_box.contains(box.x0, box.y0)
                assert panel_box.contains(box.x1, box.y1)
                assert not any(box.overlaps(region) for region in protected)
                marker_center = axis.transData.transform(annotation.xy)
                assert not box.contains(*marker_center)
                assert annotation.arrow_patch is not None
                assert annotation.arrow_patch.get_visible()
                assert (annotation.arrow_patch.get_gid() or "").startswith(
                    "dashboard-marker-zero:"
                )
            for marker, annotation in zip(markers, annotations, strict=True):
                marker_center = marker.get_offset_transform().transform(
                    marker.get_offsets()
                )[0]
                annotation_target = annotation.xycoords.transform(annotation.xy)
                assert annotation_target == pytest.approx(marker_center)
    finally:
        reporting_figures.plt.close(figure)


def test_compact_typed_legends_preserve_full_run_ids_in_semantic_sidecars(
    tmp_path: Path,
) -> None:
    study = _canonical_thirty_run_typed_study()

    _api().write_report(study, tmp_path)

    full_ids = {identity.run_id for identity in study.requested_runs}
    ci_ids = {
        identity.run_id
        for identity in study.requested_runs
        if identity.ci_scaling_active
    }
    metadata = json.loads((tmp_path / "plot_metadata.json").read_text())
    mappings = json.loads((tmp_path / "figure_row_mapping.json").read_text())
    assert set(metadata["varpro_scale.png"]["series"]) == ci_ids
    assert set(metadata["absolute_scale_stability_dashboard.png"]["series"]) == full_ids
    zero_annotations = metadata["absolute_scale_stability_dashboard.png"][
        "zero_annotations"
    ]
    expected_zero_ids = [
        identity.run_id
        for family in ("deadleaves", "lines")
        for identity in study.requested_runs
        if identity.object_family == family and identity.ci_scaling_active
    ]
    assert [item["run_id"] for item in zero_annotations] == expected_zero_ids
    assert {item["run_id"] for item in zero_annotations} == ci_ids
    assert {item["label"] for item in zero_annotations} == {"0"}
    assert {item["metric"] for item in zero_annotations} == {"reload_max_abs_error"}
    dashboard_series = metadata["absolute_scale_stability_dashboard.png"]["series"]
    physics_layout = {
        item["run_id"]: item
        for item in metadata["absolute_scale_stability_dashboard.png"]["scatter_layout"]
        if item["panel"] == "physics_reload"
    }
    for index, item in enumerate(zero_annotations):
        family_slot = index % 6
        assert item["object_family"] in {"deadleaves", "lines"}
        assert item["arm_id"]
        assert item["arm_display_label"] in {
            "Hybrid ResNet | CI NLL",
            "CNN | CI NLL",
        }
        assert item["seed"] in {3, 17, 29}
        assert item["exact_anchor"] == pytest.approx(
            {
                "x": dashboard_series[item["run_id"]]["varpro_objective_ratio"],
                "y": 0.0,
            }
        )
        layout = physics_layout[item["run_id"]]
        assert (
            item["marker_display_offset_points"]
            == layout["marker_display_offset_points"]
        )
        assert item["marker_artist_id"] == layout["marker_artist_id"]
        assert item["annotation_slot"] == family_slot
        assert item["annotation_display_offset_points"] == {
            "x": (-30, -18, -8, 8, 18, 30)[family_slot],
            "y": 12 + 14 * family_slot,
        }
        assert item["annotation_artist_id"] == (
            f"dashboard-zero-label:{item['run_id']}"
        )
        assert item["connectors"] == {
            "anchor_to_marker": layout["connector_id"],
            "marker_to_annotation": f"dashboard-marker-zero:{item['run_id']}",
        }
    assert set(mappings["varpro_scale.png"]) == ci_ids
    assert set(mappings["absolute_scale_stability_dashboard.png"]) == full_ids


def test_all_typed_scatter_panels_emit_versioned_renderer_order_layout_records(
    tmp_path: Path,
) -> None:
    study = _canonical_thirty_run_typed_study()

    _api().write_report(study, tmp_path)

    metadata = json.loads((tmp_path / "plot_metadata.json").read_text())
    from scripts.studies.ablation import reporting_figures

    registry = reporting_figures._typed_visual_role_registry(study)
    version = "ablation_report_renderer_layout_v2"
    expected_panels = {
        "varpro_scale.png": {"s1_s2": 12, "c_A_c_phi": 12},
        "absolute_scale_stability_dashboard.png": {
            "absolute_scale": 30,
            "physics_reload": 12,
        },
    }
    for filename, panel_counts in expected_panels.items():
        records = metadata[filename]["scatter_layout"]
        assert all(
            record["renderer_layout_schema_version"] == version for record in records
        )
        assert {
            panel: sum(record["panel"] == panel for record in records)
            for panel in panel_counts
        } == panel_counts
        assert len({record["marker_artist_id"] for record in records}) == len(records)
        for record in records:
            assert record["run_id"]
            assert record["arm_id"]
            assert record["visual_role_id"]
            assert record["visual_style_id"]
            role = registry[record["arm_id"]]
            assert record["visual_role_id"] == role.visual_role_id
            assert record["visual_style_id"] == role.visual_style_id
            assert record["arm_display_label"] == role.display_label
            assert record["arm_display_label"]
            assert record["object_family"] in {"deadleaves", "lines"}
            assert record["seed"] in {3, 17, 29}
            assert set(record["exact_anchor"]) == {"x", "y"}
            assert set(record["marker_display_offset_points"]) == {"x", "y"}

    varpro = metadata["varpro_scale.png"]
    dashboard = metadata["absolute_scale_stability_dashboard.png"]
    varpro_series = varpro["series"]
    dashboard_series = dashboard["series"]
    for record in varpro["scatter_layout"]:
        values = varpro_series[record["run_id"]]
        expected = (
            {"x": values["s1"], "y": values["s2"]}
            if record["panel"] == "s1_s2"
            else {"x": values["c_A"], "y": values["c_phi"]}
        )
        assert record["exact_anchor"] == pytest.approx(expected)
    for record in dashboard["scatter_layout"]:
        values = dashboard_series[record["run_id"]]
        expected = (
            {"x": values["amp_mean_ratio"], "y": values["absolute_amp_nrmse"]}
            if record["panel"] == "absolute_scale"
            else {
                "x": values["varpro_objective_ratio"],
                "y": values["reload_max_abs_error"],
            }
        )
        assert record["exact_anchor"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("renderer_name", "panels"),
    (
        ("render_varpro", ("s1_s2", "c_A_c_phi")),
        ("render_dashboard", ("absolute_scale", "physics_reload")),
    ),
)
def test_all_scatter_panel_artists_match_layout_and_do_not_hide_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    renderer_name: str,
    panels: tuple[str, str],
) -> None:
    from scripts.studies.ablation import reporting_figures

    study = _canonical_thirty_run_typed_study()
    figures = []
    monkeypatch.setattr(reporting_figures.plt, "close", figures.append)
    result = getattr(reporting_figures, renderer_name)(
        study, tmp_path / f"{renderer_name}.png"
    )
    layout = result[2]
    figure = figures.pop()
    try:
        figure.canvas.draw()
        backend = figure.canvas.get_renderer()
        data_axes = [axis for axis in figure.axes if axis.axison]
        expected_axes = [
            (family, panel) for family in ("deadleaves", "lines") for panel in panels
        ]
        for axis, (family, panel) in zip(data_axes, expected_axes, strict=True):
            records = [
                record
                for record in layout
                if record["object_family"] == family and record["panel"] == panel
            ]
            markers = [
                collection
                for collection in axis.collections
                if (collection.get_gid() or "").startswith(f"scatter-marker:{panel}:")
            ]
            assert [marker.get_gid() for marker in markers] == [
                record["marker_artist_id"] for record in records
            ]
            boxes = [_marker_display_box(marker) for marker in markers]
            assert not any(
                first.overlaps(second)
                for index, first in enumerate(boxes)
                for second in boxes[index + 1 :]
            )
            panel_box = axis.get_window_extent(backend)
            connectors = {
                patch.get_gid(): patch
                for patch in axis.patches
                if isinstance(patch, ConnectionPatch) and patch.get_gid()
            }
            for marker, record, box in zip(markers, records, boxes, strict=True):
                assert marker.get_offsets().tolist() == [
                    [record["exact_anchor"]["x"], record["exact_anchor"]["y"]]
                ]
                assert panel_box.contains(box.x0, box.y0)
                assert panel_box.contains(box.x1, box.y1)
                if record["marker_display_offset_points"] == {"x": 0.0, "y": 0.0}:
                    assert record["connector_id"] is None
                else:
                    assert record["connector_id"] in connectors
    finally:
        reporting_figures.plt.close(figure)


def test_canonical_thirty_run_typed_figure_layout_is_family_legible(
    tmp_path: Path,
) -> None:
    from scripts.studies.ablation import reporting_figures

    study = _canonical_thirty_run_typed_study()
    grid_path = tmp_path / "grid.png"
    curves_path = tmp_path / "curves.png"
    seed_path = tmp_path / "seed.png"
    varpro_path = tmp_path / "varpro.png"
    dashboard_path = tmp_path / "dashboard.png"

    grid_panels, grid_ids = reporting_figures.render_grid(study, grid_path)
    curve_ids = reporting_figures.render_curves(study, curves_path)
    seed_ids = reporting_figures.render_seed_distribution(study, seed_path)
    varpro_ids, varpro_panels, _ = reporting_figures.render_varpro(study, varpro_path)
    dashboard_ids, dashboard_panels, _, _ = reporting_figures.render_dashboard(
        study, dashboard_path
    )

    assert len(study.requested_runs) == 30
    assert len(grid_ids) == len(curve_ids) == 30
    assert len(seed_ids) == 30
    assert {panel["object_family"] for panel in grid_panels} == {
        "deadleaves",
        "lines",
    }
    assert len(varpro_ids) == 12
    assert len(dashboard_ids) == 30
    assert {panel["object_family"] for panel in varpro_panels} == {
        "deadleaves",
        "lines",
    }
    assert {panel["object_family"] for panel in dashboard_panels} == {
        "deadleaves",
        "lines",
    }
    for path in (varpro_path, dashboard_path):
        height, width = mpimg.imread(path).shape[:2]
        assert width >= 1000
        assert height >= 800
    grid_height, grid_width = mpimg.imread(grid_path).shape[:2]
    assert grid_width >= 1800
    assert grid_height < 4000
    curves_height, curves_width = mpimg.imread(curves_path).shape[:2]
    assert curves_width >= 1000
    assert curves_height >= 700
    seed_height, seed_width = mpimg.imread(seed_path).shape[:2]
    assert seed_width >= 600
    assert seed_height >= 400


def _complete_visual_review(root: Path, notes: str = "reviewed") -> bytes:
    grid_sha256 = hashlib.sha256(
        (root / "reconstruction_truth_error_grid.png").read_bytes()
    ).hexdigest()
    completed = json.dumps(
        {
            "schema_version": "visual_review_v1",
            "reviewer": "reviewer@example.test",
            "timestamp": "2026-07-10T12:00:00Z",
            "figure_sha256": grid_sha256,
            "families": _family_review_records(notes),
        },
        indent=2,
    ).encode()
    (root / "visual_review.json").write_bytes(completed)
    (root / "report_completion.json").unlink()
    return completed


def _fingerprinted_typed_study(fingerprint: str):
    api = _api()
    row = replace(_typed_physics_row(), source_fingerprint=fingerprint)
    return api.ReportInput(
        "compatibility",
        (row,),
        (_identity("run-ci-3", "arm-ci", "synthetic", 3),),
        (),
    )


def test_resume_preserves_review_only_for_identical_visual_evidence(
    tmp_path: Path,
) -> None:
    api = _api()
    study = _fingerprinted_typed_study("a" * 64)
    api.write_report(study, tmp_path)
    completed = _complete_visual_review(tmp_path)

    api.write_report(study, tmp_path)

    assert (tmp_path / "visual_review.json").read_bytes() == completed


@pytest.mark.parametrize("change", ("metrics", "fingerprint", "eligible_ids"))
def test_resume_invalidates_review_when_visual_evidence_changes(
    tmp_path: Path, change: str
) -> None:
    api = _api()
    study = _fingerprinted_typed_study("a" * 64)
    api.write_report(study, tmp_path)
    _complete_visual_review(tmp_path, notes="stale")

    if change == "metrics":
        row = study.rows[0]
        changed_records = tuple(
            build_image_metric_record(
                "amp_mean_ratio",
                0.8,
                truth_role="object_truth",
                basis="raw_amplitude",
                alignment="none",
            )
            if record.path == "truth_quality.amp_mean_ratio"
            else record
            for record in row.metric_records
        )
        changed = replace(study, rows=(replace(row, metric_records=changed_records),))
    elif change == "fingerprint":
        changed = replace(
            study,
            rows=(replace(study.rows[0], source_fingerprint="b" * 64),),
        )
    else:
        second = replace(
            study.rows[0],
            attempt=replace(study.rows[0].attempt, run_id="run-ci-17", seed=17),
            source_fingerprint="c" * 64,
        )
        changed = replace(
            study,
            rows=(*study.rows, second),
            requested_runs=(
                *study.requested_runs,
                _identity("run-ci-17", "arm-ci", "synthetic", 17),
            ),
        )

    api.write_report(changed, tmp_path)

    review = json.loads((tmp_path / "visual_review.json").read_text())
    assert review["state"] == "pending"
    assert review.get("notes") != "stale"


def test_resume_preservation_cannot_bypass_eligible_axis_validation(
    tmp_path: Path,
) -> None:
    api = _api()
    study = _fingerprinted_typed_study("a" * 64)
    api.write_report(study, tmp_path)
    _complete_visual_review(tmp_path)
    empty = replace(study.rows[0], metric_records=())

    with pytest.raises(api.ReportingError, match="eligible.*varpro_scale"):
        api.write_report(replace(study, rows=(empty,)), tmp_path)


def test_required_typed_figure_rejects_eligible_rows_without_marks(
    tmp_path: Path,
) -> None:
    api = _api()
    row = replace(_row("run-ci-3", "arm-ci", 3), metric_records=(), varpro_scales=())
    study = api.ReportInput(
        "compatibility",
        (row,),
        (_identity("run-ci-3", "arm-ci", "synthetic", 3),),
        (),
    )

    with pytest.raises(api.ReportingError, match="eligible.*varpro_scale"):
        api.write_report(study, tmp_path)


def test_failed_ci_attempt_is_not_eligible_for_typed_figure_marks(
    tmp_path: Path,
) -> None:
    api = _api()
    study = api.ReportInput(
        "compatibility",
        (
            api.ReportRow.failed(
                "run-ci-3",
                "arm-ci",
                "synthetic",
                3,
                stage="training",
                error="failed",
            ),
        ),
        (_identity("run-ci-3", "arm-ci", "synthetic", 3),),
        (GateResult.active("numeric", Verdict.FAIL),),
    )

    api.write_report(study, tmp_path)

    metadata = json.loads((tmp_path / "plot_metadata.json").read_text())
    assert metadata["varpro_scale.png"]["run_ids"] == []
    assert metadata["absolute_scale_stability_dashboard.png"]["run_ids"] == []
    assert metadata["varpro_scale.png"]["not_applicable_reason"] == (
        "no_successful_ci_evidence"
    )
    assert (
        metadata["absolute_scale_stability_dashboard.png"]["not_applicable_reason"]
        == "no_successful_ci_evidence"
    )


def test_inapplicable_typed_figure_has_visible_machine_reason(tmp_path: Path) -> None:
    api = _api()
    row = replace(_row("run-legacy-3", "arm-legacy", 3), metric_records=())
    study = api.ReportInput(
        "compatibility",
        (row,),
        (
            _identity(
                "run-legacy-3",
                "arm-legacy",
                "synthetic",
                3,
                ci_scaling_active=False,
            ),
        ),
        (),
    )

    api.write_report(study, tmp_path)

    metadata = json.loads((tmp_path / "plot_metadata.json").read_text())
    assert metadata["varpro_scale.png"]["not_applicable_reason"] == (
        "legacy_normalized_amplitude"
    )
    assert metadata["varpro_scale.png"]["visible_label"] == "Not applicable"


def test_public_required_report_set_and_verifier_are_exact(tmp_path: Path) -> None:
    api = _api()
    api.write_report(_mini_study(), tmp_path)

    assert "dose_response.png" not in api.REQUIRED_REPORT_ARTIFACTS
    assert {
        "source_manifest.toml",
        "source_config.json",
        "invocation.json",
        "expansion.json",
    } <= api.REQUIRED_REPORT_ARTIFACTS
    completion = api.verify_completed_report(tmp_path)
    assert {item["path"] for item in completion["artifacts"]} == (
        api.REQUIRED_REPORT_ARTIFACTS
    )

    payload = json.loads((tmp_path / "report_completion.json").read_text())
    payload["artifacts"].append(payload["artifacts"][0])
    (tmp_path / "report_completion.json").write_text(json.dumps(payload))
    with pytest.raises(api.ReportingError, match="artifact paths"):
        api.verify_completed_report(tmp_path)


def test_report_verifier_rejects_hash_mismatch(tmp_path: Path) -> None:
    api = _api()
    api.write_report(_mini_study(), tmp_path)
    (tmp_path / "report.md").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(api.ReportingError, match="report bytes"):
        api.verify_completed_report(tmp_path)


def test_report_verifier_rejects_symlinked_completion_anchor(tmp_path: Path) -> None:
    api = _api()
    root = tmp_path / "report"
    api.write_report(_mini_study(), root)
    outside = tmp_path / "outside-completion.json"
    (root / "report_completion.json").replace(outside)
    (root / "report_completion.json").symlink_to(outside)

    with pytest.raises(api.ReportingError, match="trust anchor"):
        api.verify_completed_report(root)


@pytest.mark.parametrize(
    ("actual", "expected"),
    (
        (None, None),
        ("a" * 64, None),
        (None, "a" * 64),
        ("a" * 64, "b" * 64),
    ),
)
def test_report_input_rejects_claim_grade_without_equal_protocol_hashes(
    actual: str | None,
    expected: str | None,
) -> None:
    api = _api()

    with pytest.raises(api.ReportingError, match="claim-grade protocol"):
        api.ReportInput(
            "claim",
            (),
            (),
            (),
            claim_grade_eligible=True,
            claim_grade_disqualifying_reasons=(),
            actual_protocol_sha256=actual,
            expected_protocol_sha256=expected,
        )


@pytest.mark.parametrize(
    ("actual", "expected"),
    (
        (None, None),
        ("a" * 64, None),
        (None, "a" * 64),
        ("a" * 64, "b" * 64),
    ),
)
def test_report_verifier_rejects_claim_grade_without_equal_protocol_hashes(
    tmp_path: Path,
    actual: str | None,
    expected: str | None,
) -> None:
    api = _api()
    api.write_report(_mini_study(), tmp_path)
    completion = tmp_path / "report_completion.json"
    payload = json.loads(completion.read_text(encoding="utf-8"))
    payload.update(
        {
            "claim_grade_eligible": True,
            "claim_grade_disqualifying_reasons": [],
            "actual_protocol_sha256": actual,
            "expected_protocol_sha256": expected,
        }
    )
    completion.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(api.ReportingError, match="claim-grade protocol"):
        api.verify_completed_report(tmp_path)


def test_experimental_ci_dashboard_keeps_physics_when_truth_is_inapplicable(
    tmp_path: Path,
) -> None:
    api = _api()
    base = _row("run-exp-3", "arm-exp-ci", 3, role="reference_reconstruction")
    records = (
        build_measurement_metric_record(
            "varpro.s1", 1.2, basis="physical_counts", alignment="none"
        ),
        build_measurement_metric_record(
            "varpro.s2", 0.5, basis="physical_counts", alignment="none"
        ),
        build_measurement_metric_record(
            "varpro.unit_objective", 4.0, basis="physical_counts", alignment="none"
        ),
        build_measurement_metric_record(
            "varpro.fitted_objective", 1.0, basis="physical_counts", alignment="none"
        ),
        build_metric_record(
            "stability.reload_max_abs_error",
            0.0,
            basis="checkpoint_reload",
            alignment="none",
        ),
    )
    row = replace(base, metric_records=records, varpro_scales=())
    study = api.ReportInput(
        "experimental-smoke",
        (row,),
        (
            _identity(
                "run-exp-3",
                "arm-exp-ci",
                "experimental",
                3,
                role="reference_reconstruction",
                object_family="experimental",
            ),
        ),
        (),
    )

    api.write_report(study, tmp_path)

    metadata = json.loads((tmp_path / "plot_metadata.json").read_text())
    dashboard = metadata["absolute_scale_stability_dashboard.png"]
    assert dashboard["panel_run_ids"]["absolute_truth"] == []
    assert dashboard["panel_run_ids"]["physics_reload"] == ["run-exp-3"]
    absolute = next(
        panel for panel in dashboard["panels"] if panel["panel"] == "absolute_scale"
    )
    assert absolute["not_applicable_reason"] == "no_object_truth"
    assert dashboard["run_ids"] == ["run-exp-3"]
    mappings = json.loads((tmp_path / "figure_row_mapping.json").read_text())
    assert mappings["absolute_scale_stability_dashboard.png"] == ["run-exp-3"]


def test_study_publication_freezes_supplied_sources_and_provenance(
    tmp_path: Path,
) -> None:
    api = _api()
    manifest_bytes = b'[study]\nid = "frozen"\n'
    source_config = {"training": {"epochs": 10}}
    invocation = {"command": ["driver", "--resume"]}
    expansion = {"selected_runs": ["run-a-1"]}
    study = replace(
        _mini_study(),
        source_manifest=manifest_bytes,
        source_config=source_config,
        invocation=invocation,
        expansion=expansion,
    )

    api.write_report(study, tmp_path)

    assert (tmp_path / "source_manifest.toml").read_bytes() == manifest_bytes
    assert json.loads((tmp_path / "source_config.json").read_text()) == source_config
    assert json.loads((tmp_path / "expansion.json").read_text()) == expansion
    stored_invocation = json.loads((tmp_path / "invocation.json").read_text())
    assert stored_invocation["command"] == invocation["command"]
    api.verify_completed_report(tmp_path)


def test_each_root_replacement_failure_rolls_back_complete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _api()
    study = _mini_study()
    original_replace = api.os.replace
    probe = tmp_path / "probe"
    api.write_report(study, probe)
    publication_order = (
        *sorted(api.REQUIRED_REPORT_ARTIFACTS),
        "report_completion.json",
    )

    for failure_name in publication_order:
        root = tmp_path / failure_name.replace(".", "-")
        api.write_report(study, root)
        before = {
            path.name: path.read_bytes() for path in root.iterdir() if path.is_file()
        }

        def fail_named_replace(source, destination, *, _failure_name=failure_name):
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                source_path.parent.name.startswith(f".{root.name}.report-")
                and destination_path.parent == root
                and destination_path.name == _failure_name
            ):
                raise OSError(f"injected failure for {_failure_name}")
            original_replace(source, destination)

        with monkeypatch.context() as context:
            context.setattr(api.os, "replace", fail_named_replace)
            with pytest.raises(api.ReportingError, match="publication"):
                api.write_report(replace(study, study_id="changed"), root)

        after = {
            path.name: path.read_bytes() for path in root.iterdir() if path.is_file()
        }
        assert after == before, failure_name
        api.verify_completed_report(root)


def test_rerun_invalidates_completed_visual_evidence(tmp_path: Path) -> None:
    api = _api()
    study = _mini_study()
    api.write_report(study, tmp_path)
    old_grid = (tmp_path / "reconstruction_truth_error_grid.png").read_bytes()
    grid_sha256 = hashlib.sha256(old_grid).hexdigest()
    completed = {
        "schema_version": "visual_review_v1",
        "reviewer": "reviewer@example.test",
        "timestamp": "2026-07-10T12:00:00Z",
        "figure_sha256": grid_sha256,
        "families": _family_review_records("old evidence"),
    }
    (tmp_path / "visual_review.json").write_text(json.dumps(completed))
    (tmp_path / "report_completion.json").unlink()

    api.write_report(replace(study, preserve_visual_evidence=False), tmp_path)

    review = json.loads((tmp_path / "visual_review.json").read_text())
    assert review["state"] == "pending"
    assert review.get("notes") != "old evidence"
    assert json.loads((tmp_path / "figure_row_mapping.json").read_text())


def test_rerender_preserves_completed_visual_review_byte_for_byte(tmp_path: Path):
    api = _api()
    base = _mini_study()
    study = replace(
        base,
        rows=(replace(base.rows[0], source_fingerprint="a" * 64),),
    )
    api.write_report(study, tmp_path)
    grid_sha256 = hashlib.sha256(
        (tmp_path / "reconstruction_truth_error_grid.png").read_bytes()
    ).hexdigest()
    completed = json.dumps(
        {
            "schema_version": "visual_review_v1",
            "reviewer": "reviewer@example.test",
            "timestamp": "2026-07-10T12:00:00Z",
            "figure_sha256": grid_sha256,
            "families": _family_review_records(),
        },
        indent=4,
    )
    review_path = tmp_path / "visual_review.json"
    review_path.write_text(completed, encoding="utf-8")
    (tmp_path / "report_completion.json").unlink()

    api.write_report(study, tmp_path)

    assert review_path.read_text(encoding="utf-8") == completed


def test_stale_renderer_sidecars_cannot_preserve_reviewed_pngs(tmp_path: Path) -> None:
    api = _api()
    base = _mini_study()
    study = replace(
        base,
        rows=(replace(base.rows[0], source_fingerprint="a" * 64),),
    )
    api.write_report(study, tmp_path)
    grid_sha256 = hashlib.sha256(
        (tmp_path / "reconstruction_truth_error_grid.png").read_bytes()
    ).hexdigest()
    completed = {
        "schema_version": "visual_review_v1",
        "reviewer": "reviewer@example.test",
        "timestamp": "2026-07-10T12:00:00Z",
        "figure_sha256": grid_sha256,
        "families": _family_review_records("stale renderer"),
    }
    (tmp_path / "visual_review.json").write_text(json.dumps(completed))
    stale_png = b"stale-reviewed-png"
    (tmp_path / "varpro_scale.png").write_bytes(stale_png)
    metadata = json.loads((tmp_path / "plot_metadata.json").read_text())
    metadata.pop("renderer_layout_schema_version", None)
    metadata["absolute_scale_stability_dashboard.png"]["zero_annotations"] = [
        {"metric": "reload_max_abs_error", "run_id": "run-a-1", "label": "0"}
    ]
    (tmp_path / "plot_metadata.json").write_text(json.dumps(metadata))
    mapping = json.loads((tmp_path / "figure_row_mapping.json").read_text())
    mapping.pop("renderer_layout_schema_version", None)
    (tmp_path / "figure_row_mapping.json").write_text(json.dumps(mapping))
    (tmp_path / "report_completion.json").unlink()

    api.write_report(study, tmp_path)

    assert (tmp_path / "varpro_scale.png").read_bytes() != stale_png
    current_metadata = json.loads((tmp_path / "plot_metadata.json").read_text())
    current_mapping = json.loads((tmp_path / "figure_row_mapping.json").read_text())
    expected = api.REPORT_RENDERER_LAYOUT_SCHEMA_VERSION
    assert current_metadata["renderer_layout_schema_version"] == expected
    assert current_mapping["renderer_layout_schema_version"] == expected
    assert (
        current_metadata["_visual_evidence_identity"]["renderer_layout_schema_version"]
        == expected
    )
    review = json.loads((tmp_path / "visual_review.json").read_text())
    assert review["state"] == "pending"


def test_legacy_zero_annotation_records_cannot_preserve_reviewed_pngs(
    tmp_path: Path,
) -> None:
    api = _api()
    base = _mini_study()
    study = replace(
        base,
        rows=(replace(base.rows[0], source_fingerprint="a" * 64),),
    )
    api.write_report(study, tmp_path)
    grid_sha256 = hashlib.sha256(
        (tmp_path / "reconstruction_truth_error_grid.png").read_bytes()
    ).hexdigest()
    (tmp_path / "visual_review.json").write_text(
        json.dumps(
            {
                "schema_version": "visual_review_v1",
                "reviewer": "reviewer@example.test",
                "timestamp": "2026-07-10T12:00:00Z",
                "figure_sha256": grid_sha256,
                "families": _family_review_records("legacy zero layout"),
            }
        )
    )
    stale_png = b"stale-zero-layout-png"
    (tmp_path / "absolute_scale_stability_dashboard.png").write_bytes(stale_png)
    metadata_path = tmp_path / "plot_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["absolute_scale_stability_dashboard.png"]["zero_annotations"] = [
        {"metric": "reload_max_abs_error", "run_id": "run-a-1", "label": "0"}
    ]
    metadata_path.write_text(json.dumps(metadata))
    (tmp_path / "report_completion.json").unlink()

    api.write_report(study, tmp_path)

    assert (
        tmp_path / "absolute_scale_stability_dashboard.png"
    ).read_bytes() != stale_png
    review = json.loads((tmp_path / "visual_review.json").read_text())
    assert review["state"] == "pending"


@pytest.mark.parametrize("filename", ("plot_metadata.json", "figure_row_mapping.json"))
def test_completed_report_rejects_missing_renderer_sidecar_version(
    tmp_path: Path, filename: str
) -> None:
    api = _api()
    api.write_report(_mini_study(), tmp_path)
    sidecar_path = tmp_path / filename
    sidecar = json.loads(sidecar_path.read_text())
    sidecar.pop("renderer_layout_schema_version", None)
    sidecar_path.write_text(json.dumps(sidecar))
    completion_path = tmp_path / "report_completion.json"
    completion = json.loads(completion_path.read_text())
    for artifact in completion["artifacts"]:
        if artifact["path"] == filename:
            artifact["sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    completion_path.write_text(json.dumps(completion))

    with pytest.raises(api.ReportingError, match="renderer.*schema"):
        api.verify_completed_report(tmp_path)


def test_write_report_creates_pending_review_when_absent(tmp_path: Path):
    api = _api()

    api.write_report(_mini_study(), tmp_path)

    pending = json.loads((tmp_path / "visual_review.json").read_text())
    assert pending["state"] == "pending"
    assert (
        pending["figure_sha256"]
        == hashlib.sha256(
            (tmp_path / "reconstruction_truth_error_grid.png").read_bytes()
        ).hexdigest()
    )


def test_write_report_may_rewrite_existing_pending_review_template(tmp_path: Path):
    api = _api()
    study = _mini_study()
    api.write_report(study, tmp_path)

    api.write_report(study, tmp_path)

    pending = json.loads((tmp_path / "visual_review.json").read_text())
    assert pending["state"] == "pending"
    assert (
        pending["figure_sha256"]
        == hashlib.sha256(
            (tmp_path / "reconstruction_truth_error_grid.png").read_bytes()
        ).hexdigest()
    )


def test_reporting_import_does_not_load_training_or_runtime_frameworks():
    code = """
import sys
import scripts.studies.ablation.reporting
blocked = [name for name in sys.modules if name == 'torch' or name.startswith('tensorflow') or 'train_lightning' in name or name.endswith('.runtime')]
assert not blocked, blocked
print('isolated')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "isolated"


def test_machine_report_outputs_are_deterministic_while_pngs_are_semantic(
    tmp_path: Path,
):
    api = _api()
    study = api.ReportInput(
        "compatibility",
        (_row("run-a-1", "arm-a", 1),),
        (api.RunIdentity("run-a-1", "arm-a", "synthetic", 1),),
        (GateResult.active("numeric", Verdict.PASS),),
    )
    first, second = tmp_path / "first", tmp_path / "second"

    api.write_report(study, first)
    api.write_report(study, second)

    for filename in (
        "report.md",
        "aggregate_metrics.json",
        "aggregate_metrics.csv",
        "arm_seed_status.json",
        "arm_seed_status.csv",
        "verdicts.json",
        "verdicts.csv",
        "figure_row_mapping.json",
        "plot_metadata.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
