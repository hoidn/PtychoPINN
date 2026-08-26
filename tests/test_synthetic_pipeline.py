"""Stage orchestration and reuse contracts for the synthetic workflow."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ptycho.workflows.synthetic_pipeline import (
    METRIC_CONTRACT_VERSION,
    RECONSTRUCTION_SCHEMA,
    EvaluationStageResult,
    ReconstructionStageResult,
    SimulationStageResult,
    SyntheticPipelineRequest,
    TrainingStageResult,
    execute_evaluation_stage,
    execute_reconstruction_stage,
    execute_simulation_stage,
    execute_training_stage,
    run_synthetic_pipeline,
)


def _request(
    root: Path,
    stages: tuple[str, ...],
    *,
    profile: str = "hybrid-resnet-lines",
    extra_file_values: dict[str, object] | None = None,
) -> SyntheticPipelineRequest:
    file_values: dict[str, object] = {
        "workflow": {
            "output_root": root,
            "stages": stages,
        }
    }
    for namespace, values in (extra_file_values or {}).items():
        current = file_values.setdefault(namespace, {})
        assert isinstance(current, dict)
        assert isinstance(values, dict)
        current.update(values)
    return SyntheticPipelineRequest(
        profile=profile,
        file_values=file_values,
        raw_argv=("--output-root", str(root), "--stages", ",".join(stages)),
        script_path="ptycho_synthetic",
    )


def _write(path: Path, payload: bytes = b"artifact") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _initialization_payload(
    mode="ones",
    *,
    gauge=3.25,
    schema_version="rect-s1s2-initialization-v2",
):
    if mode == "ones":
        return {
            "schema_version": schema_version,
            "mode": "ones",
            "solved_gauge": 1.0,
            "method": "unit_default_no_solve",
            "sampled_patterns": 0,
        }
    return {
        "schema_version": schema_version,
        "mode": "dose_closure",
        "solved_gauge": gauge,
        "method": (
            "dose_closure_seeded_uniform_unit_object"
            if schema_version == "rect-s1s2-initialization-v2"
            else "dose_closure_unit_object"
        ),
        "sampled_patterns": 256,
    }


def _write_initialization_summary(path: Path, payload=None) -> Path:
    record = _initialization_payload() if payload is None else payload
    return _write(
        path,
        (json.dumps(record) + "\n").encode("utf-8"),
    )


def _fixture_object() -> np.ndarray:
    return np.ones((12, 12), dtype=np.complex64)


def _fixture_probe() -> np.ndarray:
    return np.ones((8, 8), dtype=np.complex64)


def _write_split_fixture(path: Path, marker: int) -> None:
    np.savez(
        path,
        marker=np.asarray([marker], dtype=np.int64),
        objectGuess=_fixture_object(),
        probeGuess=_fixture_probe(),
    )


class _Executors:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def simulate(self, request):
        self.calls.append("simulate")
        root = request.output_root / "datasets"
        return SimulationStageResult(
            source_path=_write(root / "source.npz"),
            train_path=_write(root / "train.npz"),
            test_path=_write(root / "test.npz"),
            manifest_path=_write(root / "manifest.json", b"{}"),
        )

    def train(self, request):
        self.calls.append("train")
        initialization = _initialization_payload()
        return TrainingStageResult(
            bundle_path=_write(request.output_root / "training" / "wts.h5.zip"),
            training_summary_path=_write_initialization_summary(
                request.output_root / "training" / "training_summary.json",
                initialization,
            ),
            rect_s1s2_initialization=initialization,
        )

    def reconstruct(self, request):
        self.calls.append("reconstruct")
        assert not (
            request.output_root / "reconstruction" / "diagnostics.json"
        ).exists()
        return ReconstructionStageResult(
            reconstruction_path=_write(
                request.output_root / "reconstruction" / "reconstruction.npz"
            ),
            reassembly={"valid_pixel_count": 17, "channel_count": 4},
        )

    def evaluate(self, request):
        self.calls.append("evaluate")
        diagnostics = json.loads(request.diagnostics_path.read_text(encoding="utf-8"))
        assert diagnostics == {
            "schema_version": "synthetic-reconstruction-diagnostics-v1",
            "reassembly": {"channel_count": 4, "valid_pixel_count": 17},
            "metric_validity": None,
            "render": None,
        }
        return EvaluationStageResult(
            metrics_path=_write(
                request.output_root / "reconstruction" / "metrics.json", b"{}"
            ),
            comparison_path=_write(
                request.output_root / "reconstruction" / "comparison.png"
            ),
            metric_validity={"finite": True, "nonzero_mask": True},
            render={"source": "raw_arrays", "valid": True},
        )


def _run(request: SyntheticPipelineRequest, executors: _Executors):
    return run_synthetic_pipeline(
        request,
        simulation_executor=executors.simulate,
        training_executor=executors.train,
        reconstruction_executor=executors.reconstruct,
        evaluation_executor=executors.evaluate,
    )


@pytest.mark.parametrize(
    "stages, message",
    [
        (("train", "simulate"), "workflow order"),
        (("simulate", "simulate"), "duplicates"),
        (("simulate", "unknown"), "Input should be"),
    ],
)
def test_invalid_stage_sequences_fail_before_any_executor(tmp_path, stages, message):
    executors = _Executors()
    with pytest.raises(ValueError, match=message):
        _run(_request(tmp_path, stages), executors)
    assert executors.calls == []


def test_pipeline_runs_stages_in_order_and_publishes_relative_manifest(tmp_path):
    executors = _Executors()
    result = _run(
        _request(tmp_path, ("simulate", "train", "reconstruct", "evaluate")),
        executors,
    )

    assert executors.calls == ["simulate", "train", "reconstruct", "evaluate"]
    assert result.reused_stages == ()
    assert result.completed_stages == (
        "simulate",
        "train",
        "reconstruct",
        "evaluate",
    )
    assert result.resolved_workflow_path == tmp_path / "resolved_workflow.json"
    assert result.stage_manifest_path == tmp_path / "stage_manifest.json"
    assert (tmp_path / "invocation.json").is_file()
    assert (tmp_path / "invocation.sh").read_text(encoding="utf-8").strip() == (
        f"python ptycho_synthetic --output-root {tmp_path} --stages "
        "simulate,train,reconstruct,evaluate"
    )

    manifest = json.loads(result.stage_manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "synthetic-stage-manifest-v2"
    assert manifest["metric_contract_version"] == METRIC_CONTRACT_VERSION
    assert list(manifest["stages"]) == [
        "simulate",
        "train",
        "reconstruct",
        "evaluate",
    ]
    for entry in manifest["stages"].values():
        assert entry["status"] == "complete"
        assert entry["started_at"]
        assert entry["completed_at"]
        assert all(not Path(path).is_absolute() for path in entry["artifacts"])


def test_pipeline_and_evaluator_share_one_metric_contract_version():
    from ptycho_torch.reconstruction_evaluation import (
        METRIC_CONTRACT_VERSION as EVALUATION_METRIC_CONTRACT_VERSION,
    )

    assert METRIC_CONTRACT_VERSION == EVALUATION_METRIC_CONTRACT_VERSION


def test_historical_v1_stage_manifest_requires_new_root_or_retraining(tmp_path):
    _run(_request(tmp_path, ("simulate",)), _Executors())
    manifest_path = tmp_path / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "synthetic-stage-manifest-v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"synthetic-stage-manifest-v1.*new output root|retrain",
    ):
        _run(_request(tmp_path, ("simulate",)), _Executors())


def test_invocation_and_resolution_exist_before_first_expensive_stage(tmp_path):
    class Observing(_Executors):
        def simulate(self, request):
            assert (request.output_root / "invocation.json").is_file()
            assert (request.output_root / "invocation.sh").is_file()
            resolved_path = request.output_root / "resolved_workflow.json"
            assert resolved_path.is_file()
            persisted = json.loads(resolved_path.read_text(encoding="utf-8"))
            assert persisted["profile"] == "hybrid-resnet-lines"
            assert not (request.output_root / "stage_manifest.json").exists()
            return super().simulate(request)

    executors = Observing()
    _run(_request(tmp_path, ("simulate",)), executors)

    assert executors.calls == ["simulate"]


def test_omitted_prerequisite_must_be_complete_and_matching(tmp_path):
    executors = _Executors()
    with pytest.raises(FileNotFoundError, match="simulate prerequisite"):
        _run(_request(tmp_path, ("train",)), executors)
    assert executors.calls == []
    assert not (tmp_path / "stage_manifest.json").exists()


def test_complete_matching_stages_are_reused_without_executor_calls(tmp_path):
    first = _Executors()
    _run(_request(tmp_path, ("simulate", "train")), first)
    second = _Executors()

    result = _run(_request(tmp_path, ("simulate", "train")), second)

    assert second.calls == []
    assert result.reused_stages == ("simulate", "train")
    assert result.completed_stages == ("simulate", "train")


def test_pre_frame_order_default_stage_identity_remains_reusable(tmp_path):
    _run(_request(tmp_path, ("simulate",)), _Executors())
    resolved_path = tmp_path / "resolved_workflow.json"
    historical = json.loads(resolved_path.read_text(encoding="utf-8"))
    historical["simulation"].pop("frame_order_recipe")
    resolved_path.write_text(json.dumps(historical), encoding="utf-8")
    replay = _Executors()

    result = _run(_request(tmp_path, ("simulate",)), replay)

    assert replay.calls == []
    assert result.reused_stages == ("simulate",)


def test_absent_historical_frame_order_does_not_alias_coordinate_major(tmp_path):
    from ptycho.workflows.synthetic_pipeline import _assert_stage_identity

    _run(_request(tmp_path, ("simulate",)), _Executors())
    current = json.loads(
        (tmp_path / "resolved_workflow.json").read_text(encoding="utf-8")
    )
    historical = json.loads(json.dumps(current))
    historical["simulation"].pop("frame_order_recipe")
    coordinate_major = json.loads(json.dumps(historical))
    coordinate_major["simulation"]["frame_order_recipe"] = (
        "coordinate-major-interleaved-v1"
    )

    with pytest.raises(ValueError, match="simulation.frame_order_recipe"):
        _assert_stage_identity(
            "simulate",
            historical,
            coordinate_major,
            {"metric_contract_version": METRIC_CONTRACT_VERSION},
        )


def test_matching_historical_v1_initialization_record_remains_reusable(tmp_path):
    historical_record = _initialization_payload(
        "ones",
        schema_version="rect-s1s2-initialization-v1",
    )

    class HistoricalExecutors(_Executors):
        def train(self, request):
            self.calls.append("train")
            return TrainingStageResult(
                bundle_path=_write(
                    request.output_root / "training" / "wts.h5.zip"
                ),
                training_summary_path=_write_initialization_summary(
                    request.output_root / "training" / "training_summary.json",
                    historical_record,
                ),
                rect_s1s2_initialization=historical_record,
            )

    _run(
        _request(tmp_path, ("simulate", "train")),
        HistoricalExecutors(),
    )
    summary_path = tmp_path / "training" / "training_summary.json"
    historical_summary = summary_path.read_bytes()
    assert json.loads(historical_summary) == historical_record
    manifest = json.loads((tmp_path / "stage_manifest.json").read_text())
    assert manifest["schema_version"] == "synthetic-stage-manifest-v2"
    replay = _Executors()

    result = _run(_request(tmp_path, ("simulate", "train")), replay)

    assert replay.calls == []
    assert result.reused_stages == ("simulate", "train")
    assert summary_path.read_bytes() == historical_summary


@pytest.mark.parametrize(
    ("summary_payload", "message"),
    [
        (
            {
                **_initialization_payload(),
                "schema_version": "obsolete-v0",
            },
            "schema_version",
        ),
        (
            {
                name: value
                for name, value in _initialization_payload().items()
                if name != "method"
            },
            "fields",
        ),
        (
            {
                **_initialization_payload(),
                "solved_gauge": -1.0,
            },
            "solved_gauge",
        ),
    ],
)
def test_fresh_training_rejects_malformed_summary_before_manifest_completion(
    tmp_path,
    summary_payload,
    message,
):
    _run(_request(tmp_path, ("simulate",)), _Executors())

    class Malformed(_Executors):
        def train(self, request):
            self.calls.append("train")
            initialization = _initialization_payload()
            return TrainingStageResult(
                bundle_path=_write(
                    request.output_root / "training" / "wts.h5.zip"
                ),
                training_summary_path=_write_initialization_summary(
                    request.output_root / "training" / "training_summary.json",
                    summary_payload,
                ),
                rect_s1s2_initialization=initialization,
            )

    with pytest.raises(ValueError, match=message):
        _run(_request(tmp_path, ("train",)), Malformed())

    manifest = json.loads((tmp_path / "stage_manifest.json").read_text())
    assert list(manifest["stages"]) == ["simulate"]


def test_fresh_training_rejects_backend_and_persisted_record_mismatch(tmp_path):
    _run(
        _request(
            tmp_path,
            ("simulate",),
            profile="hybrid-resnet-lines-ci",
        ),
        _Executors(),
    )

    class Mismatched(_Executors):
        def train(self, request):
            self.calls.append("train")
            return TrainingStageResult(
                bundle_path=_write(
                    request.output_root / "training" / "wts.h5.zip"
                ),
                training_summary_path=_write_initialization_summary(
                    request.output_root / "training" / "training_summary.json",
                    _initialization_payload("dose_closure", gauge=3.25),
                ),
                rect_s1s2_initialization=_initialization_payload(
                    "dose_closure",
                    gauge=4.0,
                ),
            )

    with pytest.raises(ValueError, match="does not match backend"):
        _run(
            _request(
                tmp_path,
                ("train",),
                profile="hybrid-resnet-lines-ci",
            ),
            Mismatched(),
        )

    manifest = json.loads((tmp_path / "stage_manifest.json").read_text())
    assert list(manifest["stages"]) == ["simulate"]


def test_fresh_training_rejects_record_mode_that_disagrees_with_resolved_mode(
    tmp_path,
):
    _run(_request(tmp_path, ("simulate",)), _Executors())

    class WrongMode(_Executors):
        def train(self, request):
            self.calls.append("train")
            initialization = _initialization_payload("dose_closure")
            return TrainingStageResult(
                bundle_path=_write(
                    request.output_root / "training" / "wts.h5.zip"
                ),
                training_summary_path=_write_initialization_summary(
                    request.output_root / "training" / "training_summary.json",
                    initialization,
                ),
                rect_s1s2_initialization=initialization,
            )

    with pytest.raises(ValueError, match="mode.*resolved.*ones"):
        _run(_request(tmp_path, ("train",)), WrongMode())

    manifest = json.loads((tmp_path / "stage_manifest.json").read_text())
    assert list(manifest["stages"]) == ["simulate"]


def test_training_reuse_parses_summary_and_rejects_malformed_record(tmp_path):
    _run(_request(tmp_path, ("simulate", "train")), _Executors())
    summary_path = tmp_path / "training" / "training_summary.json"
    summary_path.write_text('{"mode": "ones"}\n', encoding="utf-8")
    replay = _Executors()

    with pytest.raises(ValueError, match="fields"):
        _run(_request(tmp_path, ("train",)), replay)

    assert replay.calls == []


def test_training_reuse_rejects_summary_mode_that_disagrees_with_resolved_mode(
    tmp_path,
):
    _run(_request(tmp_path, ("simulate", "train")), _Executors())
    _write_initialization_summary(
        tmp_path / "training" / "training_summary.json",
        _initialization_payload("dose_closure"),
    )
    replay = _Executors()

    with pytest.raises(ValueError, match="mode.*resolved.*ones"):
        _run(_request(tmp_path, ("train",)), replay)

    assert replay.calls == []


def test_inference_only_change_preserves_simulation_and_training_reuse(tmp_path):
    _run(_request(tmp_path, ("simulate", "train")), _Executors())
    changed = _request(
        tmp_path,
        ("simulate", "train"),
        extra_file_values={"inference": {"groups_per_center": 2}},
    )
    executors = _Executors()

    result = _run(changed, executors)

    assert executors.calls == []
    assert result.reused_stages == ("simulate", "train")


def test_validation_group_change_preserves_simulation_and_invalidates_training(
    tmp_path,
):
    _run(_request(tmp_path, ("simulate", "train")), _Executors())
    changed_training = {"training": {"validation_groups": 512}}

    with pytest.raises(ValueError, match=r"training\.validation_groups"):
        _run(
            _request(
                tmp_path,
                ("train",),
                extra_file_values=changed_training,
            ),
            _Executors(),
        )

    executors = _Executors()
    result = _run(
        _request(
            tmp_path,
            ("simulate",),
            extra_file_values=changed_training,
        ),
        executors,
    )

    assert executors.calls == []
    assert result.reused_stages == ("simulate",)
    assert result.completed_stages == ("simulate",)


def test_model_change_invalidates_training_and_names_first_field(tmp_path):
    _run(_request(tmp_path, ("simulate", "train")), _Executors())
    changed = _request(
        tmp_path,
        ("train",),
        extra_file_values={"model": {"fno_width": 40}},
    )

    with pytest.raises(ValueError, match=r"model\.fno_width"):
        _run(changed, _Executors())


def test_simulation_change_invalidates_every_completed_dependent_stage(tmp_path):
    _run(_request(tmp_path, ("simulate", "train")), _Executors())
    changed = _request(
        tmp_path,
        ("simulate",),
        extra_file_values={"simulation": {"seed": 901}},
    )

    with pytest.raises(ValueError, match=r"simulation\.train\.seed"):
        _run(changed, _Executors())


def test_workflow_runtime_change_preserves_training_reuse(tmp_path):
    _run(
        _request(
            tmp_path,
            ("simulate", "train"),
            extra_file_values={"workflow": {"deterministic": True}},
        ),
        _Executors(),
    )
    changed = _request(
        tmp_path,
        ("train",),
        extra_file_values={"workflow": {"deterministic": False}},
    )
    executors = _Executors()

    result = _run(changed, executors)

    assert executors.calls == []
    assert result.reused_stages == ("train",)


def test_inference_change_invalidates_reconstruction_before_executor_work(tmp_path):
    _run(
        _request(tmp_path, ("simulate", "train", "reconstruct")),
        _Executors(),
    )
    changed = _request(
        tmp_path,
        ("reconstruct",),
        extra_file_values={"inference": {"groups_per_center": 2}},
    )
    executors = _Executors()

    with pytest.raises(ValueError, match=r"inference\.groups_per_center"):
        _run(changed, executors)

    assert executors.calls == []


def test_unselected_incompatible_downstream_stages_are_not_reused(tmp_path):
    _run(
        _request(tmp_path, ("simulate", "train", "reconstruct", "evaluate")),
        _Executors(),
    )
    changed = _request(
        tmp_path,
        ("train",),
        extra_file_values={"inference": {"groups_per_center": 2}},
    )
    executors = _Executors()

    result = _run(changed, executors)

    assert executors.calls == []
    assert result.reused_stages == ("train",)
    assert result.completed_stages == ("simulate", "train")
    manifest = json.loads(result.stage_manifest_path.read_text(encoding="utf-8"))
    assert list(manifest["stages"]) == ["simulate", "train"]
    recorded = json.loads(result.resolved_workflow_path.read_text(encoding="utf-8"))
    assert recorded["inference"]["groups_per_center"] == 2
    assert (tmp_path / "reconstruction" / "reconstruction.npz").is_file()


def test_rejected_replay_does_not_publish_downstream_pruning(tmp_path):
    _run(
        _request(tmp_path, ("simulate", "train", "reconstruct", "evaluate")),
        _Executors(),
    )
    manifest_path = tmp_path / "stage_manifest.json"
    resolved_path = tmp_path / "resolved_workflow.json"
    manifest_before = manifest_path.read_bytes()
    resolved_before = resolved_path.read_bytes()
    changed = _request(
        tmp_path,
        ("train",),
        extra_file_values={
            "inference": {"groups_per_center": 2},
            "workflow": {"reuse_complete_artifacts": False},
        },
    )

    with pytest.raises(FileExistsError, match="complete train artifact"):
        _run(changed, _Executors())

    assert manifest_path.read_bytes() == manifest_before
    assert resolved_path.read_bytes() == resolved_before


def test_metric_contract_change_invalidates_evaluation_before_executor_work(tmp_path):
    _run(
        _request(tmp_path, ("simulate", "train", "reconstruct", "evaluate")),
        _Executors(),
    )
    manifest_path = tmp_path / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metric_contract_version"] = "obsolete-metrics-v0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    executors = _Executors()

    with pytest.raises(ValueError, match="metric_contract_version"):
        _run(_request(tmp_path, ("evaluate",)), executors)

    assert executors.calls == []


def test_failed_stage_leaves_previous_manifest_unchanged(tmp_path):
    _run(_request(tmp_path, ("simulate",)), _Executors())
    manifest_path = tmp_path / "stage_manifest.json"
    previous = manifest_path.read_bytes()

    class Failing(_Executors):
        def train(self, request):
            self.calls.append("train")
            raise RuntimeError("training exploded")

    with pytest.raises(RuntimeError, match="training exploded"):
        _run(_request(tmp_path, ("train",)), Failing())

    assert manifest_path.read_bytes() == previous
    assert (tmp_path / "resolved_workflow.json").is_file()
    failure_log = (tmp_path / "stage_logs" / "train.log").read_text(encoding="utf-8")
    assert "stage: train" in failure_log
    assert "status: failed" in failure_log
    assert "error_type: RuntimeError" in failure_log
    assert "error: training exploded" in failure_log


def test_partial_artifact_fails_closed_without_overwrite(tmp_path):
    _run(_request(tmp_path, ("simulate",)), _Executors())
    partial = _write(tmp_path / "training" / "wts.h5.zip", b"partial")
    executors = _Executors()

    with pytest.raises(FileExistsError, match="partial.*training/wts.h5.zip"):
        _run(_request(tmp_path, ("train",)), executors)

    assert partial.read_bytes() == b"partial"
    assert executors.calls == []


def test_stale_reconstruction_diagnostics_fail_before_any_selected_stage(tmp_path):
    stale = _write(tmp_path / "reconstruction" / "diagnostics.json", b"stale")
    executors = _Executors()

    with pytest.raises(FileExistsError, match="reconstruction/diagnostics.json"):
        _run(
            _request(
                tmp_path,
                ("simulate", "train", "reconstruct", "evaluate"),
            ),
            executors,
        )

    assert stale.read_bytes() == b"stale"
    assert executors.calls == []


@pytest.mark.parametrize(
    "managed_path",
    [
        "datasets",
        "stage_logs",
        "training/checkpoints",
        "training/lightning_logs",
        "training/mlruns",
    ],
)
def test_managed_directory_symlink_fails_before_external_write(
    tmp_path,
    managed_path,
):
    label = managed_path.replace("/", "-")
    outside = tmp_path.parent / f"{tmp_path.name}-{label}-outside"
    outside.mkdir()
    (tmp_path / managed_path).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / managed_path).symlink_to(outside, target_is_directory=True)
    executors = _Executors()

    with pytest.raises(ValueError, match=rf"managed.*{managed_path}.*symlink"):
        _run(_request(tmp_path, ("simulate",)), executors)

    assert executors.calls == []
    assert list(outside.iterdir()) == []


def test_managed_invocation_symlink_fails_before_external_write(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-invocation-outside"
    (tmp_path / "invocation.sh").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "invocation.sh").symlink_to(outside)
    executors = _Executors()

    with pytest.raises(ValueError, match=r"managed.*invocation\.sh.*symlink"):
        _run(_request(tmp_path, ("simulate",)), executors)

    assert executors.calls == []
    assert not outside.exists()


@pytest.mark.parametrize(
    "stage",
    ["simulate", "train", "reconstruct", "evaluate"],
)
def test_managed_stage_log_symlink_fails_before_external_write(tmp_path, stage):
    outside = tmp_path.parent / f"{tmp_path.name}-{stage}-log-outside"
    log_path = tmp_path / "stage_logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.symlink_to(outside)
    executors = _Executors()

    with pytest.raises(ValueError, match=rf"managed.*stage_logs/{stage}\.log.*symlink"):
        _run(_request(tmp_path, ("simulate",)), executors)

    assert executors.calls == []
    assert not outside.exists()


def test_executor_cannot_publish_an_unmanaged_stage_path(tmp_path):
    _run(_request(tmp_path, ("simulate",)), _Executors())

    class Unmanaged(_Executors):
        def train(self, request):
            self.calls.append("train")
            initialization = _initialization_payload()
            return TrainingStageResult(
                bundle_path=_write(request.output_root / "outside.zip"),
                training_summary_path=_write_initialization_summary(
                    request.output_root / "training" / "training_summary.json",
                    initialization,
                ),
                rect_s1s2_initialization=initialization,
            )

    with pytest.raises(ValueError, match="training/wts.h5.zip"):
        _run(_request(tmp_path, ("train",)), Unmanaged())


def test_pipeline_owns_and_atomically_extends_diagnostics(tmp_path):
    executors = _Executors()
    result = _run(
        _request(tmp_path, ("simulate", "train", "reconstruct", "evaluate")),
        executors,
    )

    diagnostics = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics == {
        "schema_version": "synthetic-reconstruction-diagnostics-v1",
        "reassembly": {"channel_count": 4, "valid_pixel_count": 17},
        "metric_validity": {"finite": True, "nonzero_mask": True},
        "render": {"source": "raw_arrays", "valid": True},
    }


def test_invalid_reconstruction_output_does_not_publish_diagnostics(tmp_path):
    _run(_request(tmp_path, ("simulate", "train")), _Executors())

    class InvalidReconstruction(_Executors):
        def reconstruct(self, request):
            self.calls.append("reconstruct")
            return ReconstructionStageResult(
                reconstruction_path=request.output_root / "wrong.npz",
                reassembly={"valid_pixel_count": 1},
            )

    with pytest.raises(ValueError, match="reconstruction/reconstruction.npz"):
        _run(_request(tmp_path, ("reconstruct",)), InvalidReconstruction())

    assert not (tmp_path / "reconstruction" / "diagnostics.json").exists()
    failure_log = (tmp_path / "stage_logs" / "reconstruct.log").read_text(
        encoding="utf-8"
    )
    assert "stage: reconstruct" in failure_log
    assert "status: failed" in failure_log
    assert "reconstruction/reconstruction.npz" in failure_log
    manifest = json.loads(
        (tmp_path / "stage_manifest.json").read_text(encoding="utf-8")
    )
    assert "reconstruct" not in manifest["stages"]


def test_invalid_evaluation_output_leaves_pending_diagnostics_unchanged(tmp_path):
    _run(
        _request(tmp_path, ("simulate", "train", "reconstruct")),
        _Executors(),
    )
    diagnostics_path = tmp_path / "reconstruction" / "diagnostics.json"
    pending = diagnostics_path.read_bytes()

    class InvalidEvaluation(_Executors):
        def evaluate(self, request):
            self.calls.append("evaluate")
            return EvaluationStageResult(
                metrics_path=request.output_root / "reconstruction" / "missing.json",
                comparison_path=_write(
                    request.output_root / "reconstruction" / "comparison.png"
                ),
                metric_validity={"finite": True},
                render={"valid": True},
            )

    with pytest.raises(ValueError, match="reconstruction/metrics.json"):
        _run(_request(tmp_path, ("evaluate",)), InvalidEvaluation())

    assert diagnostics_path.read_bytes() == pending
    manifest = json.loads(
        (tmp_path / "stage_manifest.json").read_text(encoding="utf-8")
    )
    assert "evaluate" not in manifest["stages"]


def test_evaluation_exception_restores_pipeline_owned_diagnostics(tmp_path):
    _run(
        _request(tmp_path, ("simulate", "train", "reconstruct")),
        _Executors(),
    )
    diagnostics_path = tmp_path / "reconstruction" / "diagnostics.json"
    pending = diagnostics_path.read_bytes()

    class CorruptingEvaluation(_Executors):
        def evaluate(self, request):
            self.calls.append("evaluate")
            request.diagnostics_path.write_text(
                '{"corrupt": true}',
                encoding="utf-8",
            )
            raise RuntimeError("evaluation exploded")

    with pytest.raises(RuntimeError, match="evaluation exploded"):
        _run(_request(tmp_path, ("evaluate",)), CorruptingEvaluation())

    assert diagnostics_path.read_bytes() == pending
    manifest = json.loads(
        (tmp_path / "stage_manifest.json").read_text(encoding="utf-8")
    )
    assert "evaluate" not in manifest["stages"]
    failure_log = (tmp_path / "stage_logs" / "evaluate.log").read_text(encoding="utf-8")
    assert "stage: evaluate" in failure_log
    assert "status: failed" in failure_log
    assert "error: evaluation exploded" in failure_log


def test_evaluation_validates_pending_diagnostics_before_executor(tmp_path):
    _run(
        _request(tmp_path, ("simulate", "train", "reconstruct")),
        _Executors(),
    )
    diagnostics_path = tmp_path / "reconstruction" / "diagnostics.json"
    diagnostics_path.write_text('{"schema_version": "wrong"}', encoding="utf-8")
    executors = _Executors()

    with pytest.raises(ValueError, match="diagnostics.schema_version"):
        _run(_request(tmp_path, ("evaluate",)), executors)

    assert executors.calls == []


def test_default_simulation_adapter_launches_only_the_cuda_hidden_worker(
    tmp_path,
    monkeypatch,
):
    from ptycho.workflows import synthetic_pipeline
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow

    resolved = resolve_synthetic_workflow(
        file_values={"workflow": {"output_root": tmp_path}}
    )
    request = synthetic_pipeline.SimulationStageRequest(
        profile="hybrid-resnet-lines",
        file_values={"workflow": {"output_root": tmp_path}},
        cli_values=None,
        resolved_workflow=resolved,
        output_root=tmp_path,
    )
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        root = tmp_path / "datasets"
        for name in ("source.npz", "train.npz", "test.npz"):
            _write(root / name)
        _write(root / "manifest.json", b"{}")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(synthetic_pipeline.subprocess, "run", fake_run)
    result = execute_simulation_stage(request)

    assert observed["command"][1:4] == [
        "-m",
        "scripts.simulation.synthetic_simulation_worker",
        "--request-json",
    ]
    assert "CUDA_VISIBLE_DEVICES" not in observed["kwargs"].get("env", {})
    assert result.manifest_path == tmp_path / "datasets" / "manifest.json"


def test_default_simulation_adapter_bounds_failed_subprocess_log(tmp_path, monkeypatch):
    from ptycho.workflows import synthetic_pipeline
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow

    resolved = resolve_synthetic_workflow(
        file_values={"workflow": {"output_root": tmp_path}}
    )
    request = synthetic_pipeline.SimulationStageRequest(
        profile="hybrid-resnet-lines",
        file_values={"workflow": {"output_root": tmp_path}},
        cli_values=None,
        resolved_workflow=resolved,
        output_root=tmp_path,
    )
    huge_stdout = "out" * 50_000
    huge_stderr = "err" * 50_000
    monkeypatch.setattr(
        synthetic_pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=7,
            stdout=huge_stdout,
            stderr=huge_stderr,
        ),
    )

    with pytest.raises(RuntimeError, match=r"returncode=7.*simulate\.log"):
        execute_simulation_stage(request)

    payload = (tmp_path / "stage_logs" / "simulate.log").read_text(encoding="utf-8")
    assert "returncode: 7" in payload
    assert "truncated" in payload
    assert len(payload) < 40_000


def test_default_training_adapter_uses_shared_training_workflow(tmp_path, monkeypatch):
    from ptycho.workflows import synthetic_pipeline
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow

    resolved = resolve_synthetic_workflow(
        file_values={
            "training": {
                "torch_training_seed": 3,
                "batch_order_recipe": "torch-implicit-july2026-v1",
            },
            "workflow": {"output_root": tmp_path},
        }
    )
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir(parents=True)
    train_path = dataset_root / "train.npz"
    test_path = dataset_root / "test.npz"
    _write_split_fixture(train_path, 1)
    _write_split_fixture(test_path, 2)
    manifest_path = _write_training_manifest(train_path, test_path, resolved)
    request = synthetic_pipeline.TrainingStageRequest(
        resolved_workflow=resolved,
        output_root=tmp_path,
        train_path=train_path,
        test_path=test_path,
        dataset_manifest_path=manifest_path,
    )
    bundle = _write(tmp_path / "training" / "wts.h5.zip")
    initialization = _initialization_payload()
    summary = _write_initialization_summary(
        tmp_path / "training" / "training_summary.json",
        initialization,
    )
    captured = {}

    def fake_training(workflow_request):
        captured["request"] = workflow_request
        return SimpleNamespace(
            bundle_path=bundle,
            training_summary_path=summary,
            rect_s1s2_initialization=initialization,
        )

    monkeypatch.setattr(
        synthetic_pipeline,
        "_run_shared_training_workflow",
        fake_training,
    )
    result = execute_training_stage(request)

    shared = captured["request"]
    assert shared.resolved_synthetic_workflow is resolved
    assert shared.train_data_file == request.train_path
    assert shared.test_data_file == request.test_path
    assert shared.output_dir == tmp_path / "training"
    assert shared.do_stitching is False
    assert shared.torch_training_seed == 3
    assert shared.batch_order_recipe == "torch-implicit-july2026-v1"
    assert result.bundle_path == bundle
    assert result.training_summary_path == summary
    assert result.rect_s1s2_initialization.to_jsonable() == initialization


def test_default_training_adapter_rejects_dataset_drift_before_work(
    tmp_path,
    monkeypatch,
):
    from ptycho.workflows import synthetic_pipeline
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow

    resolved = resolve_synthetic_workflow(
        file_values={"workflow": {"output_root": tmp_path}}
    )
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir(parents=True)
    train_path = dataset_root / "train.npz"
    test_path = dataset_root / "test.npz"
    _write_split_fixture(train_path, 1)
    _write_split_fixture(test_path, 2)
    manifest_path = _write_training_manifest(train_path, test_path, resolved)
    np.savez(train_path, marker=np.asarray([99], dtype=np.int64))
    calls = []
    monkeypatch.setattr(
        synthetic_pipeline,
        "_run_shared_training_workflow",
        lambda request: calls.append(request),
    )

    with pytest.raises(ValueError, match=r"splits\.train\.npz_sha256 mismatch"):
        execute_training_stage(
            synthetic_pipeline.TrainingStageRequest(
                resolved_workflow=resolved,
                output_root=tmp_path,
                train_path=train_path,
                test_path=test_path,
                dataset_manifest_path=manifest_path,
            )
        )

    assert calls == []


def test_default_training_rejects_inconsistent_split_recipe_identity(
    tmp_path,
    monkeypatch,
):
    from ptycho.workflows import synthetic_pipeline
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow

    resolved = resolve_synthetic_workflow(
        file_values={"workflow": {"output_root": tmp_path}}
    )
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir(parents=True)
    train_path = dataset_root / "train.npz"
    test_path = dataset_root / "test.npz"
    _write_split_fixture(train_path, 1)
    _write_split_fixture(test_path, 2)
    manifest_path = _write_training_manifest(train_path, test_path, resolved)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["train"]["split_recipe_identity"] = {"split": "corrupt"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        synthetic_pipeline,
        "_run_shared_training_workflow",
        lambda request: calls.append(request),
    )

    with pytest.raises(ValueError, match=r"split_recipe_identity mismatch"):
        execute_training_stage(
            synthetic_pipeline.TrainingStageRequest(
                resolved_workflow=resolved,
                output_root=tmp_path,
                train_path=train_path,
                test_path=test_path,
                dataset_manifest_path=manifest_path,
            )
        )

    assert calls == []


def test_default_training_rejects_missing_custom_probe_source_before_work(
    tmp_path,
    monkeypatch,
):
    from ptycho.workflows import synthetic_pipeline
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow

    probe_path = tmp_path / "probe.npz"
    np.savez(
        probe_path,
        probeGuess=np.ones((128, 128), dtype=np.complex64),
    )
    resolved = resolve_synthetic_workflow(
        file_values={
            "simulation": {
                "probe": {
                    "source": "custom",
                    "source_path": probe_path,
                }
            },
            "workflow": {"output_root": tmp_path},
        }
    )
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    train_path = dataset_root / "train.npz"
    test_path = dataset_root / "test.npz"
    _write_split_fixture(train_path, 1)
    _write_split_fixture(test_path, 2)
    manifest_path = _write_training_manifest(train_path, test_path, resolved)
    probe_path.unlink()
    calls = []
    monkeypatch.setattr(
        synthetic_pipeline,
        "_run_shared_training_workflow",
        lambda request: calls.append(request),
    )

    with pytest.raises(FileNotFoundError, match="custom probe source"):
        execute_training_stage(
            synthetic_pipeline.TrainingStageRequest(
                resolved_workflow=resolved,
                output_root=tmp_path,
                train_path=train_path,
                test_path=test_path,
                dataset_manifest_path=manifest_path,
            )
        )

    assert calls == []


def _default_adapter_reassembly(channel_count: int = 4) -> dict[str, object]:
    return {
        "schema_version": 1,
        "accepted_patches": channel_count,
        "total_patches": channel_count,
        "used_scan_ids": list(range(channel_count)),
        "used_center_scan_ids": [0],
        "expected_scan_ids": list(range(channel_count)),
        "filtered_eligible_scan_ids": [0],
        "s1": 1.0,
        "s2": 1.0,
        "count_metrics": {
            "status": "not_applicable",
            "reason": "legacy_normalized_amplitude",
        },
        "effective_precision": "32-true",
    }


def _resolved_gs2(root: Path):
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow

    return resolve_synthetic_workflow(
        file_values={
            "simulation": {"gridsize": 2},
            "workflow": {"accelerator": "cpu", "output_root": root},
        }
    )


def _resolved_tiled(root: Path):
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow

    return resolve_synthetic_workflow(
        file_values={
            "simulation": {
                "train_patterns": 9,
                "test_patterns": 4,
                "scan": {"position_layout": "fixed_pitch_raster"},
            },
            "training": {
                "train_raw_selection": 4,
                "training_groups": 4,
                "validation_groups": 4,
                "neighbor_count": 1,
                "neighbor_pool_size": 1,
            },
            "inference": {
                "reconstruction_method": "tiled",
                "patch_weighting": "uniform",
                "varpro_scaling": False,
                "metric_crop_border": 2,
            },
            "workflow": {"accelerator": "cpu", "output_root": root},
        }
    )


def _write_source_manifest(source_path: Path, resolved) -> Path:
    from ptycho.simulation.flat_acquisition import (
        OBJECT_PRODUCER_SYMBOLS,
        derive_seed_lineage,
    )
    from ptycho.simulation.identity import array_sha256, file_sha256
    from ptycho.workflows.synthetic_config import synthetic_workflow_to_dict

    with np.load(source_path, allow_pickle=False) as archive:
        truth = np.asarray(archive["objectGuess"])
        probe = np.asarray(archive["probeGuess"])
    lineage = derive_seed_lineage(resolved.simulation.train.seed)
    probe_config = resolved.simulation.train.probe
    probe_source_digest = (
        file_sha256(probe_config.source_path)
        if probe_config.source_path is not None and probe_config.source_path.is_file()
        else None
    )
    manifest_path = source_path.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "flat-acquisition-manifest-v1",
                "storage_layout": "flat_acquisition_v1",
                "profile": resolved.profile,
                "recipe_version": resolved.recipe_version,
                "artifacts": {"source": source_path.name},
                "source_npz_sha256": file_sha256(source_path),
                "simulation": synthetic_workflow_to_dict(resolved)["simulation"],
                "seed_lineage": lineage,
                "measurement_identity": {
                    "measurement_domain": resolved.simulation.measurement_domain,
                    "scale_contract_version": (
                        resolved.simulation.scale_contract_version
                    ),
                },
                "object": {
                    "recipe": resolved.simulation.object_recipe,
                    "producer_symbols": list(OBJECT_PRODUCER_SYMBOLS),
                    "source_commit": "test-source",
                    "array_sha256": array_sha256(truth),
                    "seed": lineage["object"],
                    "shape": list(truth.shape),
                    "dtype": truth.dtype.name,
                },
                "probe": {
                    "source_kind": probe_config.source,
                    "source_path": (
                        str(probe_config.source_path)
                        if probe_config.source_path is not None
                        else None
                    ),
                    "source_file_sha256": probe_source_digest,
                    "raw_probe_sha256": array_sha256(probe),
                    "normalized_transform_pipeline": (probe_config.transform_pipeline),
                    "transformed_probe_sha256": array_sha256(probe),
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_training_manifest(train_path: Path, test_path: Path, resolved) -> Path:
    from ptycho.config import simulation_config_sha256, simulation_config_to_dict
    from ptycho.simulation.flat_acquisition import (
        OBJECT_PRODUCER_SYMBOLS,
        derive_seed_lineage,
    )
    from ptycho.simulation.identity import (
        array_sha256,
        canonical_sha256,
        file_sha256,
    )
    from ptycho.workflows.synthetic_config import synthetic_workflow_to_dict

    lineage = derive_seed_lineage(resolved.simulation.train.seed)
    with np.load(train_path, allow_pickle=False) as archive:
        truth = np.asarray(archive["objectGuess"])
        probe = np.asarray(archive["probeGuess"])
    object_identity = {
        "recipe": resolved.simulation.object_recipe,
        "producer_symbols": list(OBJECT_PRODUCER_SYMBOLS),
        "source_commit": "test-source",
        "array_sha256": array_sha256(truth),
    }
    probe_config = resolved.simulation.train.probe
    probe_hash = array_sha256(probe)
    source_path = train_path.parent / "source.npz"
    np.savez(source_path, objectGuess=truth, probeGuess=probe)
    probe_record = {
        "source_kind": probe_config.source,
        "source_path": (
            str(probe_config.source_path)
            if probe_config.source_path is not None
            else None
        ),
        "source_file_sha256": (
            file_sha256(probe_config.source_path)
            if probe_config.source_path is not None
            and probe_config.source_path.is_file()
            else None
        ),
        "raw_probe_sha256": probe_hash,
        "normalized_transform_pipeline": probe_config.transform_pipeline,
        "transformed_probe_sha256": probe_hash,
    }
    records = {}
    for split, path in (("train", train_path), ("test", test_path)):
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        hashes = {name: array_sha256(array) for name, array in arrays.items()}
        shapes = {name: list(array.shape) for name, array in arrays.items()}
        dtypes = {name: array.dtype.name for name, array in arrays.items()}
        simulation = getattr(resolved.simulation, split)
        measurement_identity = {
            "measurement_domain": resolved.simulation.measurement_domain,
            "scale_contract_version": resolved.simulation.scale_contract_version,
            "photons_per_pattern": float(simulation.detector.photons_per_pattern),
        }
        split_recipe_identity = {
            "split": split,
            "storage_layout": "flat_acquisition_v1",
            "simulation_config_sha256": simulation_config_sha256(simulation),
            "object_identity": object_identity,
            "raw_probe_sha256": probe_hash,
            "transformed_probe_sha256": probe_hash,
            "coordinate_seed": lineage[f"{split}_coordinates"],
            "detector_seed": lineage[f"{split}_noise"],
            "measurement_identity": measurement_identity,
        }
        split_recipe_sha256 = canonical_sha256(split_recipe_identity)
        dataset_identity = {
            "split_recipe_sha256": split_recipe_sha256,
            "array_sha256": hashes,
            "shapes": shapes,
            "dtypes": dtypes,
        }
        records[split] = {
            "artifact_path": path.name,
            "storage_layout": "flat_acquisition_v1",
            "simulation_config": simulation_config_to_dict(simulation),
            "simulation_config_sha256": simulation_config_sha256(simulation),
            "measurement_identity": measurement_identity,
            "seed_lineage": lineage,
            "coordinate_seed": lineage[f"{split}_coordinates"],
            "detector_seed": lineage[f"{split}_noise"],
            "array_sha256": hashes,
            "shapes": shapes,
            "dtypes": dtypes,
            "split_recipe_identity": split_recipe_identity,
            "split_recipe_sha256": split_recipe_sha256,
            "dataset_recipe_sha256": split_recipe_sha256,
            "dataset_identity": dataset_identity,
            "dataset_sha256": canonical_sha256(dataset_identity),
            "npz_sha256": file_sha256(path),
        }
    manifest_path = train_path.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "flat-acquisition-manifest-v1",
                "storage_layout": "flat_acquisition_v1",
                "profile": resolved.profile,
                "recipe_version": resolved.recipe_version,
                "artifacts": {
                    "source": source_path.name,
                    "train": train_path.name,
                    "test": test_path.name,
                },
                "source_npz_sha256": file_sha256(source_path),
                "simulation": synthetic_workflow_to_dict(resolved)["simulation"],
                "seed_lineage": lineage,
                "measurement_identity": {
                    "measurement_domain": resolved.simulation.measurement_domain,
                    "scale_contract_version": (
                        resolved.simulation.scale_contract_version
                    ),
                },
                "object": {
                    **object_identity,
                    "seed": lineage["object"],
                    "shape": list(truth.shape),
                    "dtype": truth.dtype.name,
                },
                "probe": probe_record,
                "splits": records,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_default_reconstruction_adapter_persists_raw_c4_evidence_atomically(
    tmp_path,
    monkeypatch,
):
    from ptycho.workflows import synthetic_pipeline
    from ptycho_torch import inference

    resolved = _resolved_gs2(tmp_path)
    request = synthetic_pipeline.ReconstructionStageRequest(
        resolved_workflow=resolved,
        output_root=tmp_path,
        test_path=_write(tmp_path / "datasets" / "test.npz"),
        dataset_manifest_path=_write(tmp_path / "datasets" / "manifest.json", b"{}"),
        bundle_path=_write(tmp_path / "training" / "wts.h5.zip"),
    )
    canvas = np.arange(100, dtype=np.float32).reshape(10, 10).astype(np.complex64)
    prescale = canvas * np.complex64(2.0 + 0.0j)
    weights = np.ones((10, 10), dtype=np.float32)
    channel_indices = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    anchor = {
        "scan_com": [4.5, 4.5],
        "canvas_shape": [10, 10],
        "canvas_origin_offset": [0.0, 0.0],
    }
    reassembly = _default_adapter_reassembly()
    captured = {}

    def fake_reconstruct(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            complex_canvas=canvas,
            amplitude=np.abs(canvas),
            phase=np.angle(canvas),
            prescale_canvas=prescale,
            canvas_weights=weights,
            canvas_anchor=anchor,
            channel_indices=channel_indices,
            reassembly=SimpleNamespace(to_jsonable=lambda: reassembly),
        )

    monkeypatch.setattr(inference, "reconstruct_npz_barycentric", fake_reconstruct)

    result = execute_reconstruction_stage(request)

    assert captured["args"] == (request.bundle_path, request.test_path)
    assert captured["kwargs"] == {
        "run_root": tmp_path,
        "groups_per_center": 1,
        "expected_workflow": resolved,
        "dataset_manifest_path": request.dataset_manifest_path,
        "device": "cpu",
        "num_workers": 0,
        "inference_batch_size": 16,
        "precision": "32-true",
        "quiet": True,
    }
    assert result.reconstruction_path == (
        tmp_path / "reconstruction" / "reconstruction.npz"
    )
    assert result.reassembly == reassembly
    assert not (tmp_path / "reconstruction" / "diagnostics.json").exists()
    assert not list((tmp_path / "reconstruction").glob(".*.tmp"))
    with np.load(result.reconstruction_path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema_version",
            "complex_canvas",
            "measurement_gauge_canvas",
            "amplitude",
            "phase",
            "prescale_canvas",
            "canvas_weights",
            "canvas_anchor_json",
            "channel_indices",
        }
        assert archive["schema_version"].item() == RECONSTRUCTION_SCHEMA
        np.testing.assert_array_equal(archive["complex_canvas"], canvas)
        np.testing.assert_array_equal(archive["prescale_canvas"], prescale)
        np.testing.assert_array_equal(archive["canvas_weights"], weights)
        np.testing.assert_array_equal(archive["channel_indices"], channel_indices)
        assert json.loads(archive["canvas_anchor_json"].item()) == anchor


def test_tiled_reconstruction_dispatches_to_the_strict_mmap_tiled_adapter(
    tmp_path,
    monkeypatch,
):
    from ptycho.workflows import synthetic_pipeline
    from ptycho_torch import inference

    resolved = _resolved_tiled(tmp_path)
    request = synthetic_pipeline.ReconstructionStageRequest(
        resolved_workflow=resolved,
        output_root=tmp_path,
        test_path=_write(tmp_path / "datasets" / "test.npz"),
        dataset_manifest_path=_write(tmp_path / "datasets" / "manifest.json", b"{}"),
        bundle_path=_write(tmp_path / "training" / "wts.h5.zip"),
    )
    canvas = np.ones((20, 20), dtype=np.complex64)
    channel_indices = np.arange(4, dtype=np.int64).reshape(-1, 1)
    reassembly = _default_adapter_reassembly(channel_count=4)
    reassembly["used_center_scan_ids"] = list(range(4))
    reassembly["filtered_eligible_scan_ids"] = list(range(4))
    reassembly.update(
        {
            "assembly_method": "tiled_raster_v1",
            "requested_middle_trim": 32,
            "effective_tile_size": 10,
            "effective_patch_weighting": "uniform",
            "effective_varpro_scaling": False,
            "lattice_shape": [2, 2],
            "lattice_pitch": [10.0, 10.0],
            "object_amplitude_scale": 1.0,
            "object_amplitude_scale_applied": False,
            "object_gauge": {
                "inference_canvas_before_publication": "raw_source",
                "published_canvas": "raw_source",
                "published_scale_factor": 1.0,
                "count_diagnostics_canvas": "raw_source",
            },
        }
    )
    captured = {}

    def fake_tiled(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            complex_canvas=canvas,
            amplitude=np.abs(canvas),
            phase=np.angle(canvas),
            prescale_canvas=canvas,
            canvas_weights=np.ones(canvas.shape, dtype=np.float32),
            canvas_anchor={
                "scan_com": [69.0, 69.0],
                "canvas_shape": [20, 20],
                "canvas_origin_offset": [-59.0, -59.0],
                "truth_origin": [59, 59],
                "assembly_method": "tiled_raster_v1",
            },
            channel_indices=channel_indices,
            reassembly=reassembly,
        )

    monkeypatch.setattr(inference, "reconstruct_npz_tiled", fake_tiled, raising=False)
    monkeypatch.setattr(
        inference,
        "reconstruct_npz_barycentric",
        lambda *_args, **_kwargs: pytest.fail("barycentric adapter was called"),
    )

    result = execute_reconstruction_stage(request)

    assert captured["args"] == (request.bundle_path, request.test_path)
    assert captured["kwargs"]["expected_workflow"] is resolved
    assert result.reassembly == reassembly
    with np.load(result.reconstruction_path, allow_pickle=False) as archive:
        anchor = json.loads(archive["canvas_anchor_json"].item())
    assert anchor["truth_origin"] == [59, 59]


def test_default_reconstruction_rejects_channel_reassembly_disagreement(
    tmp_path,
    monkeypatch,
):
    from ptycho.workflows import synthetic_pipeline
    from ptycho_torch import inference

    resolved = _resolved_gs2(tmp_path)
    request = synthetic_pipeline.ReconstructionStageRequest(
        resolved_workflow=resolved,
        output_root=tmp_path,
        test_path=_write(tmp_path / "datasets" / "test.npz"),
        dataset_manifest_path=_write(tmp_path / "datasets" / "manifest.json", b"{}"),
        bundle_path=_write(tmp_path / "training" / "wts.h5.zip"),
    )
    canvas = np.ones((10, 10), dtype=np.complex64)
    inconsistent = _default_adapter_reassembly()
    inconsistent["used_scan_ids"] = [0, 1, 2]
    monkeypatch.setattr(
        inference,
        "reconstruct_npz_barycentric",
        lambda *_args, **_kwargs: SimpleNamespace(
            complex_canvas=canvas,
            amplitude=np.abs(canvas),
            phase=np.angle(canvas),
            prescale_canvas=canvas,
            canvas_weights=np.ones(canvas.shape, dtype=np.float32),
            canvas_anchor={
                "scan_com": [4.5, 4.5],
                "canvas_shape": [10, 10],
                "canvas_origin_offset": [0.0, 0.0],
            },
            channel_indices=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
            reassembly=SimpleNamespace(to_jsonable=lambda: inconsistent),
        ),
    )

    with pytest.raises(ValueError, match="reassembly.*channel_indices"):
        execute_reconstruction_stage(request)

    assert not (tmp_path / "reconstruction" / "reconstruction.npz").exists()


def test_default_evaluation_adapter_reloads_raw_artifact_and_source_truth(
    tmp_path,
    monkeypatch,
):
    from ptycho.workflows import synthetic_pipeline
    from ptycho_torch import inference, reconstruction_evaluation

    resolved = _resolved_gs2(tmp_path)
    resolved = replace(
        resolved,
        inference=replace(resolved.inference, metric_crop_border=2),
    )
    canvas = np.full((10, 10), 1.0 + 0.5j, dtype=np.complex64)
    truth = np.full((12, 12), 2.0 - 0.25j, dtype=np.complex64)
    reassembly = _default_adapter_reassembly()

    def fake_reconstruct(*_args, **_kwargs):
        return SimpleNamespace(
            complex_canvas=canvas,
            amplitude=np.abs(canvas),
            phase=np.angle(canvas),
            prescale_canvas=canvas * np.complex64(0.5),
            canvas_weights=np.ones(canvas.shape, dtype=np.float32),
            canvas_anchor={
                "scan_com": [5.5, 5.5],
                "canvas_shape": [10, 10],
                "canvas_origin_offset": [0.0, 0.0],
            },
            channel_indices=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
            reassembly=SimpleNamespace(to_jsonable=lambda: reassembly),
        )

    monkeypatch.setattr(inference, "reconstruct_npz_barycentric", fake_reconstruct)
    reconstruct_request = synthetic_pipeline.ReconstructionStageRequest(
        resolved_workflow=resolved,
        output_root=tmp_path,
        test_path=_write(tmp_path / "datasets" / "test.npz"),
        dataset_manifest_path=_write(tmp_path / "datasets" / "manifest.json", b"{}"),
        bundle_path=_write(tmp_path / "training" / "wts.h5.zip"),
    )
    reconstruction = execute_reconstruction_stage(reconstruct_request)
    source_path = tmp_path / "datasets" / "source.npz"
    np.savez(
        source_path,
        objectGuess=truth,
        probeGuess=_fixture_probe(),
        ignored=np.asarray([17]),
    )
    manifest_path = _write_source_manifest(source_path, resolved)
    diagnostics_path = tmp_path / "reconstruction" / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "schema_version": "synthetic-reconstruction-diagnostics-v1",
                "reassembly": reassembly,
                "metric_validity": None,
                "render": None,
            }
        ),
        encoding="utf-8",
    )
    captured = {}
    metrics_path = tmp_path / "reconstruction" / "metrics.json"
    comparison_path = tmp_path / "reconstruction" / "comparison.png"

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        _write(metrics_path, b"{}")
        _write(comparison_path, b"png")
        return SimpleNamespace(
            metrics_path=metrics_path,
            comparison_path=comparison_path,
            metric_validity={"finite": True},
            render={"source": "raw_arrays", "valid": True},
        )

    monkeypatch.setattr(
        reconstruction_evaluation,
        "evaluate_reconstruction_quality",
        fake_evaluate,
    )
    result = execute_evaluation_stage(
        synthetic_pipeline.EvaluationStageRequest(
            resolved_workflow=resolved,
            output_root=tmp_path,
            source_path=source_path,
            dataset_manifest_path=manifest_path,
            reconstruction_path=reconstruction.reconstruction_path,
            diagnostics_path=diagnostics_path,
        )
    )

    np.testing.assert_array_equal(captured["complex_canvas"], canvas)
    np.testing.assert_array_equal(captured["truth"], truth)
    np.testing.assert_array_equal(
        captured["channel_indices"],
        np.asarray([[0, 1, 2, 3]], dtype=np.int64),
    )
    assert captured["reassembly"] == reassembly
    assert captured["groups_per_center"] == 1
    assert captured["metric_crop_border"] == 2
    assert captured["output_dir"] == tmp_path / "reconstruction"
    assert result.metrics_path == metrics_path
    assert result.comparison_path == comparison_path
    assert (
        json.loads(diagnostics_path.read_text(encoding="utf-8"))["metric_validity"]
        is None
    )


@pytest.mark.parametrize(
    "corruption, message",
    [
        ("missing_channels", "reconstruction artifact fields"),
        ("duplicate_channels", "distinct scan ids"),
        ("malformed_anchor", "canvas_anchor_json is invalid JSON"),
    ],
)
def test_default_evaluation_rejects_corrupt_reconstruction_before_scoring(
    tmp_path,
    monkeypatch,
    corruption,
    message,
):
    from ptycho.workflows import synthetic_pipeline
    from ptycho_torch import reconstruction_evaluation

    resolved = _resolved_gs2(tmp_path)
    canvas = np.ones((10, 10), dtype=np.complex64)
    payload = {
        "schema_version": np.asarray(RECONSTRUCTION_SCHEMA),
        "complex_canvas": canvas,
        "measurement_gauge_canvas": canvas,
        "amplitude": np.abs(canvas),
        "phase": np.angle(canvas),
        "prescale_canvas": canvas,
        "canvas_weights": np.ones(canvas.shape, dtype=np.float32),
        "canvas_anchor_json": np.asarray(
            json.dumps(
                {
                    "scan_com": [5.5, 5.5],
                    "canvas_shape": [10, 10],
                    "canvas_origin_offset": [0.0, 0.0],
                }
            )
        ),
        "channel_indices": np.asarray([[0, 1, 2, 3]], dtype=np.int64),
    }
    if corruption == "missing_channels":
        del payload["channel_indices"]
    elif corruption == "duplicate_channels":
        payload["channel_indices"] = np.asarray([[0, 0, 2, 3]], dtype=np.int64)
    else:
        payload["canvas_anchor_json"] = np.asarray("{")
    reconstruction_path = tmp_path / "reconstruction" / "reconstruction.npz"
    reconstruction_path.parent.mkdir(parents=True)
    np.savez(reconstruction_path, **payload)
    source_path = tmp_path / "datasets" / "source.npz"
    source_path.parent.mkdir(parents=True)
    np.savez(
        source_path,
        objectGuess=np.ones((12, 12), dtype=np.complex64),
        probeGuess=_fixture_probe(),
    )
    manifest_path = _write_source_manifest(source_path, resolved)
    diagnostics_path = tmp_path / "reconstruction" / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "schema_version": "synthetic-reconstruction-diagnostics-v1",
                "reassembly": _default_adapter_reassembly(),
                "metric_validity": None,
                "render": None,
            }
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        reconstruction_evaluation,
        "evaluate_reconstruction_quality",
        lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(ValueError, match=message):
        execute_evaluation_stage(
            synthetic_pipeline.EvaluationStageRequest(
                resolved_workflow=resolved,
                output_root=tmp_path,
                source_path=source_path,
                dataset_manifest_path=manifest_path,
                reconstruction_path=reconstruction_path,
                diagnostics_path=diagnostics_path,
            )
        )

    assert calls == []


def test_default_evaluation_rejects_source_truth_drift_before_scoring(
    tmp_path,
    monkeypatch,
):
    from ptycho.workflows import synthetic_pipeline
    from ptycho_torch import reconstruction_evaluation

    resolved = _resolved_gs2(tmp_path)
    reconstruction_path = tmp_path / "reconstruction" / "reconstruction.npz"
    reconstruction_path.parent.mkdir(parents=True)
    canvas = np.ones((10, 10), dtype=np.complex64)
    np.savez(
        reconstruction_path,
        schema_version=np.asarray(RECONSTRUCTION_SCHEMA),
        complex_canvas=canvas,
        measurement_gauge_canvas=canvas,
        amplitude=np.abs(canvas),
        phase=np.angle(canvas),
        prescale_canvas=canvas,
        canvas_weights=np.ones(canvas.shape, dtype=np.float32),
        canvas_anchor_json=np.asarray(
            json.dumps(
                {
                    "scan_com": [5.5, 5.5],
                    "canvas_shape": [10, 10],
                    "canvas_origin_offset": [0.0, 0.0],
                }
            )
        ),
        channel_indices=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
    )
    source_path = tmp_path / "datasets" / "source.npz"
    source_path.parent.mkdir(parents=True)
    np.savez(
        source_path,
        objectGuess=np.ones((12, 12), dtype=np.complex64),
        probeGuess=_fixture_probe(),
    )
    manifest_path = _write_source_manifest(source_path, resolved)
    np.savez(
        source_path,
        objectGuess=np.full((12, 12), 2.0, dtype=np.complex64),
        probeGuess=_fixture_probe(),
    )
    diagnostics_path = tmp_path / "reconstruction" / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "schema_version": "synthetic-reconstruction-diagnostics-v1",
                "reassembly": _default_adapter_reassembly(),
                "metric_validity": None,
                "render": None,
            }
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        reconstruction_evaluation,
        "evaluate_reconstruction_quality",
        lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(ValueError, match="source_npz_sha256 mismatch"):
        execute_evaluation_stage(
            synthetic_pipeline.EvaluationStageRequest(
                resolved_workflow=resolved,
                output_root=tmp_path,
                source_path=source_path,
                dataset_manifest_path=manifest_path,
                reconstruction_path=reconstruction_path,
                diagnostics_path=diagnostics_path,
            )
        )

    assert calls == []


def test_no_stage_selection_uses_real_default_adapters_in_complete_order(
    tmp_path,
    monkeypatch,
):
    from ptycho_torch import inference, reconstruction_evaluation

    calls = []
    source = np.full((12, 12), 1.0 + 0.0j, dtype=np.complex64)

    def simulate(request):
        calls.append("simulate")
        root = request.output_root / "datasets"
        root.mkdir(parents=True, exist_ok=True)
        np.savez(
            root / "source.npz",
            objectGuess=source,
            probeGuess=_fixture_probe(),
        )
        np.savez(root / "train.npz", marker=np.asarray([1]))
        np.savez(root / "test.npz", marker=np.asarray([2]))
        _write_source_manifest(root / "source.npz", request.resolved_workflow)
        return SimulationStageResult(
            source_path=root / "source.npz",
            train_path=root / "train.npz",
            test_path=root / "test.npz",
            manifest_path=root / "manifest.json",
        )

    def train(request):
        calls.append("train")
        initialization = _initialization_payload()
        return TrainingStageResult(
            bundle_path=_write(request.output_root / "training" / "wts.h5.zip"),
            training_summary_path=_write_initialization_summary(
                request.output_root / "training" / "training_summary.json",
                initialization,
            ),
            rect_s1s2_initialization=initialization,
        )

    def reconstruct(*_args, **kwargs):
        calls.append("reconstruct")
        canvas = np.full((10, 10), 1.0 + 0.25j, dtype=np.complex64)
        channels = int(kwargs["expected_workflow"].data.gridsize ** 2)
        return SimpleNamespace(
            complex_canvas=canvas,
            amplitude=np.abs(canvas),
            phase=np.angle(canvas),
            prescale_canvas=canvas,
            canvas_weights=np.ones(canvas.shape, dtype=np.float32),
            canvas_anchor={
                "scan_com": [5.5, 5.5],
                "canvas_shape": [10, 10],
                "canvas_origin_offset": [0.0, 0.0],
            },
            channel_indices=np.arange(channels, dtype=np.int64).reshape(1, channels),
            reassembly=SimpleNamespace(
                to_jsonable=lambda: _default_adapter_reassembly(channels)
            ),
        )

    def evaluate(**kwargs):
        calls.append("evaluate")
        assert kwargs["expected_channels"] == 1
        root = tmp_path / "reconstruction"
        return SimpleNamespace(
            metrics_path=_write(root / "metrics.json", b"{}"),
            comparison_path=_write(root / "comparison.png", b"png"),
            metric_validity={"finite": True},
            render={"source": "raw_arrays", "valid": True},
        )

    monkeypatch.setattr(inference, "reconstruct_npz_barycentric", reconstruct)
    monkeypatch.setattr(
        reconstruction_evaluation,
        "evaluate_reconstruction_quality",
        evaluate,
    )
    request = SyntheticPipelineRequest(
        file_values={
            "workflow": {"accelerator": "cpu", "output_root": tmp_path},
        }
    )

    result = run_synthetic_pipeline(
        request,
        simulation_executor=simulate,
        training_executor=train,
    )

    assert calls == ["simulate", "train", "reconstruct", "evaluate"]
    assert result.completed_stages == (
        "simulate",
        "train",
        "reconstruct",
        "evaluate",
    )
    diagnostics = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["reassembly"] == _default_adapter_reassembly(1)
    assert diagnostics["metric_validity"] == {"finite": True}
    assert diagnostics["render"] == {"source": "raw_arrays", "valid": True}
    assert set(json.loads(result.resolved_workflow_path.read_text())) >= {
        "simulation",
        "model",
        "training",
        "inference",
        "workflow",
    }
    assert {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } >= {
        "datasets/source.npz",
        "datasets/train.npz",
        "datasets/test.npz",
        "datasets/manifest.json",
        "training/wts.h5.zip",
        "training/training_summary.json",
        "reconstruction/reconstruction.npz",
        "reconstruction/metrics.json",
        "reconstruction/comparison.png",
        "reconstruction/diagnostics.json",
        "stage_manifest.json",
    }


@pytest.mark.parametrize(
    "stages",
    [
        ("train",),
        ("reconstruct",),
        ("evaluate",),
        ("train", "reconstruct"),
        ("reconstruct", "evaluate"),
    ],
)
def test_partial_stage_selections_replay_matching_completed_artifacts(tmp_path, stages):
    _run(
        _request(tmp_path, ("simulate", "train", "reconstruct", "evaluate")),
        _Executors(),
    )
    executors = _Executors()

    result = _run(_request(tmp_path, stages), executors)

    assert executors.calls == []
    assert result.reused_stages == stages
