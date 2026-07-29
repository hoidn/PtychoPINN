"""TDD coverage for the one-variable configuration bridge ladder (plan Task 21).

The ladder walks from the current Task 27 run2 grid-lines Hybrid ResNet reference
(N=128/C=1/dictionary/amplitude/MAE/historical stitch; sealed evidence)
toward the withdrawn study endpoint (N=64/C=4/mmap/count/Poisson/rectangular/
VarPro), changing exactly one configuration group per rung. These tests drive
the CPU plumbing with stubs; no GPU, no real N=128/N=64 data generation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import get_args, get_type_hints

import numpy as np
import pytest

from scripts.studies.ablation import (
    runtime,
    runtime_ladder,
    runtime_ladder_diagnostics,
)
from scripts.studies.ablation.datasets import DatasetError
from scripts.studies.ablation.runtime_errors import (
    RuntimeExecutionError,
    StudyRequestError,
)
from scripts.studies.ablation.runtime_ladder import (
    LadderRequest,
    evaluate_rung_gate,
    run_bridge_ladder,
    verify_baseline,
)
from scripts.studies.ablation.runtime_ladder_evidence import (
    parse_sealed_rung_evidence,
    seal_rung_evidence,
)
from scripts.studies.ablation.runtime_ladder_execution import LadderRunResult
from scripts.studies.ablation.runtime_ladder_spec import (
    LADDER_SPEC_KIND,
    MUTABLE_CONFIG_FIELDS,
    config_delta,
    load_ladder_spec,
)
from scripts.studies.ablation.verdicts import Verdict
from tests.studies.test_grid_lines_reference_performance import (
    GT_SHAPE,
    HYBRID_ID,
    N_SMALL,
    _Harness,
    _bridge_table,
    _hybrid_reference,
    _recipe_table,
    _toml_table,
    _write_dataset_files,
    _write_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKED_SPEC = REPO_ROOT / "scripts/studies/specs/grid_lines_bridge_ladder.toml"
REFERENCE_SPEC = (
    REPO_ROOT / "scripts/studies/specs/grid_lines_reference_performance.toml"
)
IMPLEMENTATION_PLAN = (
    REPO_ROOT
    / "docs/superpowers/plans/2026-07-09-ci-model-compatibility-ablation.md"
)
TASK28_DESIGN = (
    REPO_ROOT
    / "docs/superpowers/specs/2026-07-12-probe-rank-physics-contract-fix-design.md"
)
COMPATIBILITY_DESIGN = (
    REPO_ROOT
    / "docs/superpowers/specs/2026-07-09-ci-model-compatibility-ablation-design.md"
)
DOCS_INDEX = REPO_ROOT / "docs/index.md"
PROBE_LAYOUT_SPEC = REPO_ROOT / "docs/specs/spec-ptycho-torch-probe-layout.md"
TASK28_REPORT = REPO_ROOT / ".superpowers/sdd/task-28-report.md"

HISTORICAL_BASELINE_EVIDENCE_SHA = (
    "1c317b3188b4bdf6bfcf1b1b2317e28e9e9bcff4bd1197a2bd6249894aa4eaf3"
)
TASK27_BASELINE_EVIDENCE = (
    REPO_ROOT
    / ".artifacts/reference_qualification/task27_gain16_hybrid_prequalification_run2"
    / HYBRID_ID
    / "reference_evidence.json"
)
TASK28_BASELINE_EVIDENCE = (
    REPO_ROOT
    / ".artifacts/bridge_ladder/task28_gain16_seed3/rung0_dictionary"
    / HYBRID_ID
    / "reference_evidence.json"
)
TASK28_BASELINE_EVIDENCE_SHA = (
    "155ee5961e31f9cf82c012d6bb61591bd776551f728d66bb19e0f3abee6ad298"
)
TASK28_GENERIC_TWIN_PROVENANCE = (
    REPO_ROOT
    / ".artifacts/bridge_ladder/task28_gain16_seed3/datasets"
    / "n128_run1084_generic/generic_twin_provenance.json"
)
TASK28_GENERIC_TWIN_PROVENANCE_SHA = (
    "3f97e27de19a28eca85528893741e3558f035e338b8fda7c8a5f8636b8cbf569"
)
TASK28_RUNG1A_EVIDENCE = (
    REPO_ROOT
    / ".artifacts/bridge_ladder/task28_gain16_seed3_unit_scaling_rerun"
    / "rung1a_mmap_full_scanset/rung_evidence.json"
)
TASK28_RUNG1A_EVIDENCE_SHA = (
    "a8741f93511cc90eb26777b1ff2cab0235ce812cff3661bad168632fce9ce711"
)
TASK28_LADDER_REPORT = (
    REPO_ROOT
    / ".artifacts/bridge_ladder/task28_gain16_seed3_unit_scaling_rerun"
    / "ladder_report.json"
)
TASK28_LADDER_REPORT_SHA = (
    "2f95c0e4651a2a48449797f4a34aed557ce70962c8e82dced0d40b4aa95a0cf6"
)
TASK28_HISTORICAL_RUNG1A_FAIL_EVIDENCE = (
    REPO_ROOT
    / ".artifacts/bridge_ladder/task28_gain16_seed3/convergence"
    / "rung1a_mmap_full_scanset/rung_evidence.json"
)
TASK28_HISTORICAL_RUNG1A_FAIL_EVIDENCE_SHA = (
    "f7c3eba38f0a93529cc37315d1d3e42f43f1dd5b1e5b239300d68cdd71f9c132"
)
TASK28_HISTORICAL_FAIL_REPORT = (
    REPO_ROOT
    / ".artifacts/bridge_ladder/task28_gain16_seed3/convergence/ladder_report.json"
)
TASK28_HISTORICAL_FAIL_REPORT_SHA = (
    "130096cf45fb9e193308f272c84b0179f1948d2c0abacf700634dfec762303c7"
)
TASK28_PROBE_ARCHIVE_SHA = (
    "9f82cb9eb2c5a853764b98c1657b778600c0e90425296a7d1fdc6e8fdb53c906"
)
TASK28_RAW_PROBE_ARRAY_SHA = (
    "de564a3ed5e70118fde70d8b65214ddfb3f00364228ef8b7c61a3f31a56c309a"
)
TASK28_TRANSFORMED_PROBE_SHA = (
    "eeccb1c92eae6dce36f4102bccda3f814b3eaa16e03e5c805f786edc628d4cd2"
)
FLY001_RAW_PROBE_SHA = (
    "1d0a47db5a8efcc920748f89d7cbf6186ac9b5bf6af134e134ecd0158343b2f6"
)
FLY001_TRANSFORMED_N128_SHA = (
    "f23cf1507882fffe8b2d3a7707c1f9879ae197f090fe7860564bf3aaa519716a"
)
FLY001_TRANSFORMED_N64_SHA = (
    "aa7608cf09c342f7428abe0d41a844373f1f5d8b4c96e45133ae3a8918b04853"
)

CHECKED_RUNG_IDS = (
    "rung1a_mmap_full_scanset",
    "rung1b_bounds_filter",
    "rung2_generic_evaluator",
    "rung3_fly001_probe",
    "rung4_n64",
    "rung5_c4_probe_weighting",
    "rung6_count_poisson",
    "rung7_rectangular",
    "rung8_varpro",
)


# ---------------------------------------------------------------------------
# Miniature ladder spec builders (no GPU, no real data generation)
# ---------------------------------------------------------------------------


def _write_generic_twin(
    tmp_path: Path,
    identity: dict[str, Any],
    *,
    name: str,
    count_domain: bool = False,
    train_count: int = 6,
    test_count: int = 3,
) -> dict[str, Path]:
    """Write a generic-schema twin of the miniature dictionary NPZ pair."""
    rng = np.random.default_rng(23)
    out: dict[str, Path] = {}
    for split, count, with_truth in (
        ("train", train_count, False),
        ("test", test_count, True),
    ):
        diff3d = rng.random((count, N_SMALL, N_SMALL)).astype(np.float32)
        if count_domain:
            diff3d = np.round(diff3d * 50.0).astype(np.float32)
        payload: dict[str, Any] = {
            "diff3d": diff3d,
            # Spread positions so the mmap loader's (0.1, 0.9) range bounds
            # keep interior points eligible.
            "xcoords": (10.0 + 5.0 * rng.random(count) * count).astype(np.float64),
            "ycoords": (10.0 + 5.0 * rng.random(count) * count).astype(np.float64),
            "probeGuess": identity["probe"],
        }
        if with_truth:
            payload["objectGuess"] = (
                rng.normal(size=GT_SHAPE) + 1j * rng.normal(size=GT_SHAPE)
            ).astype(np.complex64)
        else:
            # PtychoDataset requires objectGuess in every file; all-ones is
            # its documented "no object" placeholder.
            payload["objectGuess"] = np.ones(
                (N_SMALL, N_SMALL), dtype=np.complex64
            )
        path = tmp_path / name / f"{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **payload)
        out[split] = path
    return out


def _baseline_config(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "mini_dictionary",
        "loader": "dictionary",
        "gated_evaluator": "historical",
        "probe_normalize": False,
        "mmap_bounds_filter": "off",
        "mmap_scale_convention": "loader",
        "mmap_train_sampler": "sequential",
        "mmap_probe_batch_shape": "modes",
        "N": N_SMALL,
        "gridsize": 1,
        "position_crop_border": 3,
        "training_patch_weighting": "central_mask",
        "measurement_domain": "normalized_amplitude",
        "torch_loss_mode": "mae",
        "count_scale_mode": "off",
        "scale_contract_version": "legacy_v1",
        "physics_forward_mode": "amplitude",
        "amplitude_physics_gain": 16.0,
        "rect_s1s2_trainable": False,
        "varpro_scaling": False,
        "architecture": "hybrid_resnet",
        "seed": 3,
        "epochs": 5,
        "batch_size": 16,
        "infer_batch_size": 16,
        "learning_rate": 2e-4,
        "hybrid_encoder_conv_hidden_scale": 2.0,
        "scheduler": "ReduceLROnPlateau",
        "plateau_factor": 0.5,
        "plateau_patience": 2,
        "plateau_min_lr": 1e-4,
        "plateau_threshold": 0.0,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "generator_output_mode": "real_imag",
        "probe_source": "custom",
        "fno_modes": 12,
        "fno_width": 32,
        "fno_blocks": 4,
        "fno_cnn_blocks": 2,
        "enable_checkpointing": True,
        "logger_backend": "csv",
    }


def _mini_recipe(identity: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    table = _recipe_table(identity)
    table["id"] = dataset_id
    return table


def _default_rungs(identity: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "mini_rung1_loader",
            "group": "loader_schema",
            "dataset": "mini_generic",
            "changes": {"loader": "mmap", "dataset": "mini_generic"},
        },
        {
            "id": "mini_rung2_evaluator",
            "group": "reassembly_alignment",
            "dataset": "mini_generic",
            "changes": {"gated_evaluator": "generic"},
        },
        {
            "id": "mini_rung3_varpro",
            "group": "inference_varpro",
            "dataset": "mini_generic",
            "changes": {"varpro_scaling": True},
        },
    ]


def _resolve_endpoint(
    baseline_config: dict[str, Any], rungs: list[dict[str, Any]]
) -> dict[str, Any]:
    resolved = dict(baseline_config)
    for rung in rungs:
        if rung.get("diagnostic"):
            continue  # diagnostic branches never reach the endpoint
        resolved.update(rung["changes"])
    return resolved


def _ladder_spec_text(
    identity: dict[str, Any],
    *,
    baseline_evidence: Path,
    baseline_evidence_sha256: str,
    reference_spec: Path,
    rungs: list[dict[str, Any]] | None = None,
    groups: dict[str, list[str]] | None = None,
    gate_overrides: dict[str, Any] | None = None,
    endpoint_config: dict[str, Any] | None = None,
    baseline_config_overrides: dict[str, Any] | None = None,
    kind: str = LADDER_SPEC_KIND,
) -> str:
    rungs = _default_rungs(identity) if rungs is None else rungs
    groups = (
        {
            "loader_schema": ["loader", "dataset"],
            "reassembly_alignment": ["gated_evaluator"],
            "grouping_weighting": ["gridsize", "training_patch_weighting"],
            "count_scale_bridge": ["count_scale_mode"],
            "inference_varpro": ["varpro_scaling"],
        }
        if groups is None
        else groups
    )
    if (gate_overrides or {}).get("policy") == "absolute_ssim_delta_v1":
        gate = {
            "policy": "absolute_ssim_delta_v1",
            "threshold_provenance": "locked",
            "max_abs_amp_ssim_delta": 0.02,
            "max_abs_phase_ssim_delta": 0.01,
        }
    else:
        gate = {
            "policy": "retained_ssim_v1",
            "threshold_provenance": "locked",
            "retained_amp_ssim_min_fraction": 0.85,
            "retained_phase_ssim_min_fraction": 0.85,
            "absolute_amp_ssim_floor": 0.50,
        }
    gate.update(gate_overrides or {})
    baseline_config = _baseline_config(identity)
    baseline_config.update(baseline_config_overrides or {})
    endpoint = (
        _resolve_endpoint(baseline_config, rungs)
        if endpoint_config is None
        else endpoint_config
    )
    parts = [
        _toml_table("schema", {"kind": kind, "version": 1}),
        _toml_table("study", {"id": "mini_bridge_ladder"}),
        _toml_table("gate", gate),
        _toml_table(
            "baseline",
            {
                "id": "rung0_reference",
                "reference_spec": str(reference_spec),
                "reference_id": HYBRID_ID,
                "evidence": str(baseline_evidence),
                "evidence_sha256": baseline_evidence_sha256,
                "dataset": "mini_dictionary",
            },
        ),
        _toml_table("baseline.config", baseline_config),
        _toml_table("groups", groups),
        _toml_table("datasets.mini_dictionary", {"expression": "dictionary"}),
        _toml_table(
            "datasets.mini_dictionary.recipe", _mini_recipe(identity, "mini_dictionary")
        ),
        _toml_table("datasets.mini_generic", {"expression": "generic_amplitude"}),
        _toml_table(
            "datasets.mini_generic.recipe", _mini_recipe(identity, "mini_generic")
        ),
        _toml_table(
            "datasets.mini_generic_counts",
            {"expression": "generic_count_intensity"},
        ),
        _toml_table(
            "datasets.mini_generic_counts.recipe",
            _mini_recipe(identity, "mini_generic_counts"),
        ),
    ]
    for rung in rungs:
        top = {
            "id": rung["id"],
            "group": rung["group"],
            "dataset": rung["dataset"],
        }
        for flag in (
            "requires_scan_accounting",
            "requires_normalization_evidence",
            "requires_count_error_evidence",
            "diagnostic",
            "control_rung",
            "execution_status",
        ):
            if flag in rung:
                top[flag] = rung[flag]
        parts.append(_toml_table("rungs", top, array_of_tables=True))
        parts.append(_toml_table("rungs.changes", rung["changes"]))
        for field, entry in rung.get("expected_differences", {}).items():
            parts.append(_toml_table(f"rungs.expected_differences.{field}", entry))
    parts.append(_toml_table("endpoint.config", endpoint))
    parts.append(
        _toml_table(
            "residuals",
            {"id": "mini_residual", "description": "documented residual"},
            array_of_tables=True,
        )
    )
    return "\n".join(parts)


@pytest.fixture()
def mini_ladder(tmp_path: Path) -> dict[str, Any]:
    """Miniature ladder: reference spec + PASSing sealed baseline evidence."""
    identity = _write_dataset_files(tmp_path)
    _write_generic_twin(tmp_path, identity, name="mini_generic")
    reference_spec = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    with pytest.MonkeyPatch.context() as patcher:
        _Harness(patcher, identity)
        request = runtime.ReferenceQualificationRequest(
            spec=reference_spec,
            train_npz=identity["train"],
            test_npz=identity["test"],
            output_root=tmp_path / "baseline_out",
            base_dir=tmp_path,
        )
        outcome = runtime.run_reference_qualification(request)
    assert outcome.passed, "baseline reference qualification fixture must pass"
    evidence = tmp_path / "baseline_out" / HYBRID_ID / "reference_evidence.json"
    return {
        "identity": identity,
        "tmp_path": tmp_path,
        "reference_spec": reference_spec,
        "baseline_evidence": evidence,
        "baseline_evidence_sha256": hashlib.sha256(
            evidence.read_bytes()
        ).hexdigest(),
    }


def _write_ladder_spec(mini: dict[str, Any], **kwargs: Any) -> Path:
    spec = mini["tmp_path"] / "ladder_spec.toml"
    spec.write_text(
        _ladder_spec_text(
            mini["identity"],
            baseline_evidence=mini["baseline_evidence"],
            baseline_evidence_sha256=kwargs.pop(
                "baseline_evidence_sha256", mini["baseline_evidence_sha256"]
            ),
            reference_spec=mini["reference_spec"],
            **kwargs,
        ),
        encoding="utf-8",
    )
    return spec


def _stub_run_result(
    rung: Any,
    config: dict[str, Any],
    materialized: Any,
    *,
    amp_ssim: float = 0.88,
    phase_ssim: float = 0.95,
    scan_accounting: dict[str, Any] | None = None,
    normalization: bool = True,
    canvas_coverage_fraction: float = 0.97,
    count_consistency: dict[str, Any] | None = None,
    physics_scaling_constant: float | None = None,
) -> LadderRunResult:
    reuse = None
    train_sha = infer_sha = None
    if normalization:
        train_sha = hashlib.sha256(b"train-stats").hexdigest()
        infer_sha = hashlib.sha256(b"train-stats").hexdigest()
        reuse = True
    return LadderRunResult(
        rung_id=rung.id,
        materialized=materialized,
        best_checkpoint=Path("checkpoints/best.ckpt"),
        checkpoint_sha256=hashlib.sha256(rung.id.encode()).hexdigest(),
        pre_stitch_patch_sha256=hashlib.sha256(b"patches").hexdigest(),
        historical_canvas_sha256=hashlib.sha256(b"canvas").hexdigest(),
        generic_canvas_sha256=hashlib.sha256(b"canvas").hexdigest(),
        historical_mask_sha256=hashlib.sha256(b"mask").hexdigest(),
        generic_mask_sha256=hashlib.sha256(b"mask").hexdigest(),
        canvases_equivalent=True,
        masks_equivalent=True,
        no_resize_asserted=True,
        gauge_handling="eval_reconstruction_mean_amplitude_plane_phase_v1",
        gated_evaluator=str(config["gated_evaluator"]),
        amp_mae=0.05,
        phase_mae=0.10,
        amp_ssim=amp_ssim,
        phase_ssim=phase_ssim,
        effective_probe_sha256=materialized.probe_sha256,
        effective_probe_matches_recipe=True,
        inference_reuses_training_normalization=reuse,
        training_normalization_sha256=train_sha,
        inference_normalization_sha256=infer_sha,
        varpro_applied=bool(config["varpro_scaling"]),
        varpro_s1=1.0 if config["varpro_scaling"] else None,
        varpro_s2=1.0 if config["varpro_scaling"] else None,
        scan_accounting=scan_accounting,
        canvas_coverage_fraction=canvas_coverage_fraction,
        count_consistency=count_consistency,
        physics_scaling_constant=physics_scaling_constant,
        resolved_config=dict(config),
    )


class _ExecutorStub:
    """Stand-in for execute_ladder_rung with injectable per-rung metrics."""

    def __init__(
        self,
        metrics: dict[str, tuple[float, float]] | None = None,
        **result_kwargs: Any,
    ) -> None:
        self.metrics = metrics or {}
        self.result_kwargs = result_kwargs
        self.executed: list[str] = []

    def __call__(
        self,
        spec: Any,
        rung: Any,
        *,
        train_npz: Path,
        test_npz: Path,
        work_dir: Path,
        seed: int | None = None,
    ) -> LadderRunResult:
        from scripts.studies.ablation.datasets import validate_ladder_npz_pair

        self.executed.append(rung.id)
        materialized = validate_ladder_npz_pair(
            spec.dataset(rung.dataset), train_npz, test_npz
        )
        amp, phase = self.metrics.get(rung.id, (0.88, 0.95))
        config = dict(rung.resolved_config)
        if seed is not None:
            config["seed"] = seed
        return _stub_run_result(
            rung, config, materialized, amp_ssim=amp, phase_ssim=phase,
            **self.result_kwargs,
        )


def _datasets_root(mini: dict[str, Any]) -> Path:
    """Stage per-dataset NPZ pairs under the conventional layout."""
    root = mini["tmp_path"] / "staged_datasets"
    identity = mini["identity"]
    pairs = {
        "mini_dictionary": (identity["train"], identity["test"]),
        "mini_generic": (
            mini["tmp_path"] / "mini_generic" / "train.npz",
            mini["tmp_path"] / "mini_generic" / "test.npz",
        ),
    }
    for dataset_id, (train, test) in pairs.items():
        target = root / dataset_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "train.npz").write_bytes(Path(train).read_bytes())
        (target / "test.npz").write_bytes(Path(test).read_bytes())
    return root


def _run_walk(
    mini: dict[str, Any],
    spec_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub: _ExecutorStub,
    **request_kwargs: Any,
) -> Any:
    monkeypatch.setattr(runtime_ladder, "execute_ladder_rung", stub)
    request = LadderRequest(
        spec=spec_path,
        datasets_root=request_kwargs.pop("datasets_root", None) or _datasets_root(mini),
        output_root=request_kwargs.pop("output_root", mini["tmp_path"] / "ladder_out"),
        base_dir=mini["tmp_path"],
        **request_kwargs,
    )
    return run_bridge_ladder(request)


# ---------------------------------------------------------------------------
# Checked spec: parses, single-group property, endpoint walk, gate proposal
# ---------------------------------------------------------------------------


def test_checked_spec_parses_canonical_chain_from_reference_baseline() -> None:
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    assert tuple(rung.id for rung in spec.rungs) == CHECKED_RUNG_IDS
    assert spec.baseline.reference_id == HYBRID_ID
    assert spec.baseline.status == "current"
    assert spec.baseline.evidence == TASK28_BASELINE_EVIDENCE.resolve()
    assert spec.baseline.evidence != TASK27_BASELINE_EVIDENCE.resolve()
    assert spec.baseline.evidence_sha256 == TASK28_BASELINE_EVIDENCE_SHA
    assert spec.baseline.historical_evidence_declared == (
        ".artifacts/reference_qualification/run1/"
        "grid_lines_hybrid_resnet_reference/reference_evidence.json"
    )
    assert spec.baseline.historical_evidence_sha256 == HISTORICAL_BASELINE_EVIDENCE_SHA
    assert spec.baseline.reference_spec == REFERENCE_SPEC.resolve()
    assert spec.baseline.config["loader"] == "dictionary"
    assert spec.baseline.config["N"] == 128
    assert spec.baseline.config["gridsize"] == 1
    assert spec.baseline.config["seed"] == 3
    assert spec.baseline.config["epochs"] == 5
    assert spec.baseline.config["amplitude_physics_gain"] == 16.0
    assert spec.baseline.config["mmap_scale_convention"] == "dictionary_parity"
    assert spec.rung("rung1a_mmap_full_scanset").resolved_config[
        "mmap_scale_convention"
    ] == "dictionary_parity"
    assert spec.baseline.config["hybrid_encoder_conv_hidden_scale"] == 2.0
    assert spec.endpoint_config["amplitude_physics_gain"] == 1.0


def test_checked_spec_rung_deltas_are_single_group() -> None:
    """MANDATORY: each rung differs from its resolution base — the chain
    predecessor, or the named control rung for control_rung diagnostics —
    in exactly the declared group's fields, every change effective."""
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    by_id = {rung.id: rung for rung in spec.rungs}
    previous = spec.baseline.config
    for rung in spec.rungs:
        base = (
            by_id[rung.control_rung].resolved_config
            if rung.control_rung is not None
            else previous
        )
        delta = config_delta(base, rung.resolved_config)
        assert set(delta) == set(spec.groups[rung.group]), rung.id
        assert set(delta) == set(rung.changes), rung.id
        for field, (before, after) in delta.items():
            assert before != after, f"{rung.id}.{field} is a no-op change"
            assert field in MUTABLE_CONFIG_FIELDS, f"{rung.id}.{field}"
        if not rung.diagnostic:
            # Diagnostic branches never propagate into the chain.
            previous = rung.resolved_config


def test_checked_spec_groups_are_unique_and_cover_the_design_list() -> None:
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    rung_groups = [rung.group for rung in spec.rungs]
    chain_groups = [rung.group for rung in spec.rungs if not rung.diagnostic]
    # Chain rungs own distinct groups; diagnostic branches may share one
    # (1d/1e both probe ingestion_sampler from different controls).
    assert len(set(chain_groups)) == len(chain_groups)
    assert rung_groups == [
        "loader_schema",
        "ingestion_bounds",
        "reassembly_alignment",
        "probe_source_transform",
        "detector_size",
        "grouping_weighting",
        "measurement_domain_loss",
        "rectangular_scaling",
        "inference_varpro",
    ]


def test_checked_spec_dataset_recipes_step_one_group_at_a_time() -> None:
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    previous = spec.dataset(str(spec.baseline.config["dataset"]))
    for rung in spec.rungs:
        current = spec.dataset(rung.dataset)
        if rung.group == "loader_schema":
            assert previous.expression == "dictionary"
            assert current.expression == "generic_amplitude"
            assert (
                current.recipe.transformed_probe_sha256
                == previous.recipe.transformed_probe_sha256
            )
        elif rung.group == "probe_source_transform":
            assert current.recipe.probe_archive_declared.endswith("fly001.npz")
            assert current.recipe.raw_probe_array_sha256 == FLY001_RAW_PROBE_SHA
            assert (
                current.recipe.transformed_probe_sha256
                == FLY001_TRANSFORMED_N128_SHA
            )
            assert current.recipe.probe_smoothing_sigma == 0.0
            assert current.recipe.N == previous.recipe.N
        elif rung.group == "detector_size":
            assert current.recipe.N == 64
            assert previous.recipe.N == 128
            assert (
                current.recipe.transformed_probe_sha256
                == FLY001_TRANSFORMED_N64_SHA
            )
        elif rung.group == "measurement_domain_loss":
            assert previous.expression == "generic_amplitude"
            assert current.expression == "generic_count_intensity"
            assert (
                current.recipe.transformed_probe_sha256
                == previous.recipe.transformed_probe_sha256
            )
        else:
            assert current.id == previous.id, rung.id
        previous = current


def test_checked_spec_endpoint_matches_final_rung() -> None:
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    final = spec.rungs[-1].resolved_config
    assert dict(final) == dict(spec.endpoint_config)
    assert final["loader"] == "mmap"
    assert final["mmap_bounds_filter"] == "endpoint"
    assert final["mmap_scale_convention"] == "loader"
    assert final["mmap_train_sampler"] == "sequential"
    assert final["gated_evaluator"] == "generic"
    assert final["N"] == 64
    assert final["gridsize"] == 2
    assert final["training_patch_weighting"] == "probe"
    assert final["measurement_domain"] == "count_intensity"
    assert final["torch_loss_mode"] == "poisson"
    assert final["scale_contract_version"] == "ci_intensity_v2"
    assert final["physics_forward_mode"] == "rectangular_scaled"
    assert final["rect_s1s2_trainable"] is True
    assert final["varpro_scaling"] is True


def test_checked_spec_gate_thresholds_are_locked() -> None:
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    assert spec.gate.policy == "absolute_ssim_delta_v1"
    assert spec.gate.threshold_provenance == "locked"
    assert spec.gate.max_abs_amp_ssim_delta == 0.02
    assert spec.gate.max_abs_phase_ssim_delta == 0.01


@pytest.mark.parametrize(
    "gate_line",
    [
        "max_abs_amp_ssim_delta = 0.02",
        "max_abs_phase_ssim_delta = 0.01",
    ],
)
def test_absolute_spec_rejects_missing_threshold(
    mini_ladder: dict[str, Any], gate_line: str
) -> None:
    spec = _write_ladder_spec(
        mini_ladder, gate_overrides={"policy": "absolute_ssim_delta_v1"}
    )
    spec.write_text(
        spec.read_text(encoding="utf-8").replace(f"{gate_line}\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(StudyRequestError, match="gate"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])


def test_absolute_spec_rejects_retained_policy_thresholds(
    mini_ladder: dict[str, Any]
) -> None:
    spec = _write_ladder_spec(
        mini_ladder, gate_overrides={"policy": "absolute_ssim_delta_v1"}
    )
    spec.write_text(
        spec.read_text(encoding="utf-8").replace(
            "max_abs_amp_ssim_delta = 0.02",
            "retained_amp_ssim_min_fraction = 0.85",
        ),
        encoding="utf-8",
    )
    with pytest.raises(StudyRequestError, match="gate"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])


def test_historical_retained_spec_still_parses(mini_ladder: dict[str, Any]) -> None:
    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    assert spec.gate.policy == "retained_ssim_v1"


def test_execution_refuses_unlocked_thresholds(
    mini_ladder: dict[str, Any], tmp_path: Path
) -> None:
    spec_path = _write_ladder_spec(
        mini_ladder, gate_overrides={"threshold_provenance": "proposed_task21a"}
    )
    request = LadderRequest(
        spec=spec_path,
        datasets_root=tmp_path,
        output_root=tmp_path / "out",
        base_dir=mini_ladder["tmp_path"],
    )
    with pytest.raises(StudyRequestError, match="locked"):
        run_bridge_ladder(request)


def test_checked_spec_grouping_rung_requires_scan_accounting() -> None:
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    accounting = {rung.id: rung.requires_scan_accounting for rung in spec.rungs}
    assert accounting["rung5_c4_probe_weighting"] is True
    normalization = {
        rung.id: rung.requires_normalization_evidence for rung in spec.rungs
    }
    assert normalization["rung1a_mmap_full_scanset"] is True
    assert normalization["rung6_count_poisson"] is True
    assert normalization["rung7_rectangular"] is True
    assert normalization["rung8_varpro"] is True


def test_checked_rung1a_normalization_evidence_fails_closed() -> None:
    from scripts.studies.ablation.runtime_ladder_gating import protocol_failure

    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)
    rung = spec.rung("rung1a_mmap_full_scanset")
    payload = {
        "canvases_equivalent": True,
        "masks_equivalent": True,
        "effective_probe_matches_recipe": True,
        "canvas_coverage_fraction": 1.0,
        "resolved_config": dict(rung.resolved_config),
    }

    assert protocol_failure(
        rung,
        {
            **payload,
            "normalization": {
                "inference_reuses_training_normalization": False,
            },
        },
    ) == "ladder_normalization_not_reused"
    with pytest.raises(
        StudyRequestError, match="normalization-statistics reuse evidence"
    ):
        protocol_failure(rung, payload)


def test_checked_spec_documents_out_of_ladder_residuals() -> None:
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    residual_ids = {residual.id for residual in spec.residuals}
    assert {
        "object_family_scan_pattern",
        "dose",
        "epoch_budget",
        "architecture_width",
        "driver_inference_reassembly",
    } <= residual_ids


def test_checked_spec_dry_run_stays_free_of_torch_and_tensorflow() -> None:
    code = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from scripts.studies.ablation.runtime_ladder_spec import (
    load_ladder_spec,
    render_ladder_dry_run,
)
spec = load_ladder_spec({str(CHECKED_SPEC)!r}, base_dir={str(REPO_ROOT)!r})
plan = render_ladder_dry_run(spec)
assert "rung1a_mmap_full_scanset" in plan
assert "rung8_varpro" in plan
blocked = [
    name
    for name in sys.modules
    if name == "torch" or name.startswith("tensorflow") or name == "lightning"
]
assert not blocked, blocked
print("isolated")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("isolated")


def test_checked_spec_baseline_pins_task28_rung0_evidence_and_metrics() -> None:
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    control = verify_baseline(spec)

    plan = runtime_ladder.render_ladder_dry_run(spec)
    assert "status=current" in plan
    assert str(TASK28_BASELINE_EVIDENCE.relative_to(REPO_ROOT)) in plan
    assert str(TASK27_BASELINE_EVIDENCE.relative_to(REPO_ROOT)) not in plan
    assert control.evidence_sha256 == TASK28_BASELINE_EVIDENCE_SHA
    assert control.amp_ssim == 0.8858652644013688
    assert control.phase_ssim == 0.9618665959387648

    evidence = json.loads(TASK28_BASELINE_EVIDENCE.read_bytes())
    assert evidence["fixture_amp_mae"] == 0.08168590068817139
    assert evidence["fixture_phase_mae"] == 0.12818376669684495
    assert evidence["fixture_amp_ssim"] == control.amp_ssim
    assert evidence["fixture_phase_ssim"] == control.phase_ssim


@pytest.mark.parametrize(
    ("artifact", "expected_sha256"),
    [
        (TASK28_BASELINE_EVIDENCE, TASK28_BASELINE_EVIDENCE_SHA),
        (TASK28_GENERIC_TWIN_PROVENANCE, TASK28_GENERIC_TWIN_PROVENANCE_SHA),
        (TASK28_RUNG1A_EVIDENCE, TASK28_RUNG1A_EVIDENCE_SHA),
        (TASK28_LADDER_REPORT, TASK28_LADDER_REPORT_SHA),
        (
            TASK28_HISTORICAL_RUNG1A_FAIL_EVIDENCE,
            TASK28_HISTORICAL_RUNG1A_FAIL_EVIDENCE_SHA,
        ),
        (TASK28_HISTORICAL_FAIL_REPORT, TASK28_HISTORICAL_FAIL_REPORT_SHA),
    ],
)
def test_task28_immutable_artifacts_are_tracked_with_pinned_blob_bytes(
    artifact: Path, expected_sha256: str
) -> None:
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)
    assert spec.baseline.evidence == TASK28_BASELINE_EVIDENCE.resolve()
    artifact_path = artifact.relative_to(REPO_ROOT).as_posix()

    indexed = subprocess.run(
        ["git", "ls-files", "--stage", "--", artifact_path],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert indexed.stdout.strip(), f"immutable artifact is not tracked: {artifact_path}"
    blob_sha = indexed.stdout.split(maxsplit=3)[1]
    blob = subprocess.run(
        ["git", "cat-file", "blob", blob_sha],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout

    assert hashlib.sha256(blob).hexdigest() == expected_sha256


def test_task28_rung1a_pass_evidence_is_canonical_v2() -> None:
    evidence, evidence_sha = parse_sealed_rung_evidence(
        TASK28_RUNG1A_EVIDENCE.read_bytes()
    )

    assert evidence_sha == TASK28_RUNG1A_EVIDENCE_SHA
    assert evidence["schema_version"] == "bridge_ladder_rung_evidence_v2"
    assert evidence["rung_id"] == "rung1a_mmap_full_scanset"
    assert evidence["metrics"] == {
        "amp_mae": 0.07664361596107483,
        "amp_ssim": 0.8913340876617375,
        "phase_mae": 0.11914721730481034,
        "phase_ssim": 0.9632217816205675,
    }
    assert evidence["control"] == {
        "amp_ssim": 0.8858652644013688,
        "evidence_sha256": TASK28_BASELINE_EVIDENCE_SHA,
        "phase_ssim": 0.9618665959387648,
        "rung_id": "rung0_reference",
    }
    assert evidence["dataset"] == {
        "id": "n128_run1084_generic",
        "n_test": 729,
        "n_train": 8978,
        "probe_sha256": TASK28_TRANSFORMED_PROBE_SHA,
        "recipe_fingerprint_sha256": (
            "012c0fe660191ec3f12f62f4da04ae944deb812e3a1604cec054a9b5a2413e67"
        ),
        "test_sha256": (
            "17b2aea9a9deeb3ead2ab78771f19b33a2612b2666196e20dd45fa1a51f2275b"
        ),
        "train_sha256": (
            "628cac77ef85c3927e3d5407f509556f054267e71e567aed67500b8de5f6ae4e"
        ),
    }
    assert evidence["effective_probe_matches_recipe"] is True
    assert evidence["effective_probe_sha256"] == TASK28_TRANSFORMED_PROBE_SHA
    assert evidence["resolved_config"]["mmap_probe_batch_shape"] == "modes"
    assert evidence["normalization"] == {
        "inference_normalization_sha256": (
            "279dbef649a7ddb2ca329c3977b652a541b8d0cd5cf6248212db7310953cfc36"
        ),
        "inference_reuses_training_normalization": True,
        "training_normalization_sha256": (
            "757559a3e42ecab03b244ffc8202092948cc4b65547799e491c00b68d10f99a1"
        ),
    }
    assert evidence["gate"] == {
        "abs_amp_delta": 0.0054688232603687,
        "abs_phase_delta": 0.0013551856818027,
        "control": {
            "amp_ssim": 0.8858652644013688,
            "phase_ssim": 0.9618665959387648,
        },
        "current": {
            "amp_ssim": 0.8913340876617375,
            "phase_ssim": 0.9632217816205675,
        },
        "max_abs_amp_ssim_delta": 0.02,
        "max_abs_phase_ssim_delta": 0.01,
        "policy": "absolute_ssim_delta_v1",
        "protocol_failure_reason": None,
        "reason": None,
        "threshold_provenance": "locked",
        "verdict": "pass",
    }


def test_task28_historical_rung1a_fail_remains_pinned() -> None:
    evidence, evidence_sha = parse_sealed_rung_evidence(
        TASK28_HISTORICAL_RUNG1A_FAIL_EVIDENCE.read_bytes()
    )

    assert evidence_sha == TASK28_HISTORICAL_RUNG1A_FAIL_EVIDENCE_SHA
    assert evidence["metrics"]["amp_ssim"] == 0.856505683935826
    assert evidence["metrics"]["phase_ssim"] == 0.9498293416806348
    assert evidence["gate"]["verdict"] == "fail"
    assert evidence["gate"]["reason"] == (
        "ladder_absolute_amp_ssim_delta_exceeded"
    )
    assert evidence["gate"]["protocol_failure_reason"] is None


def test_task28_ladder_report_pins_canonical_pass() -> None:
    report_bytes = TASK28_LADDER_REPORT.read_bytes()
    report = json.loads(report_bytes)

    assert hashlib.sha256(report_bytes).hexdigest() == TASK28_LADDER_REPORT_SHA
    assert set(report) == {
        "baseline",
        "first_material_degradation",
        "gate",
        "rungs",
        "schema_version",
        "spec",
        "study_id",
    }
    assert report["schema_version"] == "grid_lines_bridge_ladder_report_v2"
    assert report["spec"] == "scripts/studies/specs/grid_lines_bridge_ladder.toml"
    assert report["first_material_degradation"] is None
    assert report["baseline"] == {
        "amp_ssim": 0.8858652644013688,
        "evidence_sha256": TASK28_BASELINE_EVIDENCE_SHA,
        "id": "rung0_reference",
        "phase_ssim": 0.9618665959387648,
    }
    assert report["gate"] == {
        "max_abs_amp_ssim_delta": 0.02,
        "max_abs_phase_ssim_delta": 0.01,
        "policy": "absolute_ssim_delta_v1",
        "threshold_provenance": "locked",
    }
    rung1a = report["rungs"][0]
    assert rung1a["id"] == "rung1a_mmap_full_scanset"
    assert rung1a["evidence_path"] == (
        "rung1a_mmap_full_scanset/rung_evidence.json"
    )
    assert rung1a["evidence_sha256"] == TASK28_RUNG1A_EVIDENCE_SHA
    assert rung1a["verdict"] == "pass"
    assert rung1a["reason"] is None
    assert rung1a["gate"]["protocol_failure_reason"] is None


def test_task28_historical_fail_report_remains_pinned() -> None:
    report_bytes = TASK28_HISTORICAL_FAIL_REPORT.read_bytes()
    report = json.loads(report_bytes)

    assert hashlib.sha256(report_bytes).hexdigest() == (
        TASK28_HISTORICAL_FAIL_REPORT_SHA
    )
    assert report["first_material_degradation"] == "rung1a_mmap_full_scanset"
    rung1a = report["rungs"][0]
    assert rung1a["evidence_sha256"] == (
        TASK28_HISTORICAL_RUNG1A_FAIL_EVIDENCE_SHA
    )
    assert rung1a["verdict"] == "fail"
    assert rung1a["reason"] == "ladder_absolute_amp_ssim_delta_exceeded"


def test_task28_generic_twin_provenance_records_exact_probe_lineage() -> None:
    provenance = json.loads(TASK28_GENERIC_TWIN_PROVENANCE.read_bytes())
    lineage = provenance["probe_lineage"]

    assert lineage["input_probe_archive"] == {
        "file_sha256": TASK28_PROBE_ARCHIVE_SHA,
        "path": (
            ".artifacts/bridge_ladder/task28_gain16_seed3/datasets/"
            "Run1084_recon3_postPC_shrunk_3.npz"
        ),
        "probe_key": "probeGuess",
        "raw_probe_array_canonical_sha256": TASK28_RAW_PROBE_ARRAY_SHA,
    }
    assert lineage["dictionary_source_stored_transformed"] == {
        "probe_key": "probeGuess",
        "splits_equal": True,
        "test": {
            "canonical_sha256": TASK28_TRANSFORMED_PROBE_SHA,
            "dtype": "complex64",
            "shape": [128, 128],
        },
        "train": {
            "canonical_sha256": TASK28_TRANSFORMED_PROBE_SHA,
            "dtype": "complex64",
            "shape": [128, 128],
        },
    }
    assert lineage["generic_output"] == {
        "output_equal": True,
        "probe_key": "probeGuess",
        "splits_equal": True,
        "test": {
            "canonical_sha256": TASK28_TRANSFORMED_PROBE_SHA,
            "dtype": "complex64",
            "shape": [128, 128],
        },
        "train": {
            "canonical_sha256": TASK28_TRANSFORMED_PROBE_SHA,
            "dtype": "complex64",
            "shape": [128, 128],
        },
    }


def test_task28_surfaces_route_canonical_pass_and_unblock_task22() -> None:
    plan = IMPLEMENTATION_PLAN.read_text(encoding="utf-8")
    design = TASK28_DESIGN.read_text(encoding="utf-8")
    compatibility_design = COMPATIBILITY_DESIGN.read_text(encoding="utf-8")
    index = DOCS_INDEX.read_text(encoding="utf-8")
    normalized_plan = " ".join(plan.split())
    normalized_design = " ".join(design.split())
    normalized_compatibility_design = " ".join(compatibility_design.split())
    normalized_index = " ".join(index.split())
    canonical_root = str(TASK28_LADDER_REPORT.parent.relative_to(REPO_ROOT))

    task28_section = normalized_plan.split("### Task 28:", 1)[1].split(
        "### Task 29:", 1
    )[0]
    assert "**Status:** Complete final (`complete_final`)" in task28_section
    assert "**Status:** In progress" not in task28_section
    task22_section = normalized_plan.split("### Task 22:", 1)[1].split(
        "### Task 23:", 1
    )[0]
    task23_section = normalized_plan.split("### Task 23:", 1)[1].split(
        "### Task 24:", 1
    )[0]
    task24_section = normalized_plan.split("### Task 24:", 1)[1].split(
        "### Task 25:", 1
    )[0]
    assert "**Status:** Pending and unblocked; next executable task" in task22_section
    assert "has not run" in task22_section
    assert "Task 28" not in task23_section
    assert "Task 28" not in task24_section
    assert canonical_root in normalized_index
    assert "Task 28" in normalized_index and "complete_final" in normalized_index
    assert "Task 22" in normalized_index and "has not run" in normalized_index
    assert "Task 29 producer retirement" in normalized_design
    assert "Task 28 adjudicates only rung0 versus canonical rung1a" in (
        normalized_compatibility_design
    )


def test_task28_normative_spec_retains_historical_support_until_task29() -> None:
    spec = PROBE_LAYOUT_SPEC.read_text(encoding="utf-8")
    compatibility_notes = spec.split("## 4. Compatibility notes", 1)[1]
    lever_contract = next(
        paragraph
        for paragraph in compatibility_notes.split("\n\n")
        if "mmap_probe_batch_shape" in paragraph
    )
    normalized_contract = " ".join(lever_contract.split())

    assert "retained" in normalized_contract
    assert "Task 29" in normalized_contract
    assert "producer retirement" in normalized_contract
    assert "tombstone" in normalized_contract
    assert "Task 28" not in normalized_contract


def test_task28_diagnostic_docstring_records_normalization_recovery() -> None:
    normalized_docstring = " ".join(
        (runtime_ladder_diagnostics.__doc__ or "").split()
    ).lower()

    assert "rung 1c" in normalized_docstring
    assert "unit normalization" in normalized_docstring
    assert "recovered" in normalized_docstring
    assert "normalization ownership" in normalized_docstring
    assert "exonerated the global normalization constants" not in normalized_docstring


def test_task28_report_disambiguates_implementation_and_finalization_reviews() -> None:
    report = TASK28_REPORT.read_text(encoding="utf-8")
    review_section = report.split("## Implementation And Reviews", 1)[1].split(
        "## RED And GREEN Evidence", 1
    )[0]
    normalized_review = " ".join(review_section.split())
    normalized_review_lower = normalized_review.lower()

    assert "implementation spec-compliance review: approved" in (
        normalized_review_lower
    )
    assert "implementation code-quality review: approved" in (
        normalized_review_lower
    )
    assert "Finalization quality review" in normalized_review
    assert "CHANGES REQUIRED" in normalized_review
    assert "re-review pending" in normalized_review


# ---------------------------------------------------------------------------
# Spec validation failures (fail closed at load time)
# ---------------------------------------------------------------------------


def test_spec_rejects_unknown_kind(mini_ladder: dict[str, Any]) -> None:
    spec = _write_ladder_spec(mini_ladder, kind="other_v1")
    with pytest.raises(StudyRequestError, match="kind"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])


def test_spec_rejects_multi_group_rung_delta(mini_ladder: dict[str, Any]) -> None:
    rungs = _default_rungs(mini_ladder["identity"])
    rungs[1]["changes"]["varpro_scaling"] = True
    spec = _write_ladder_spec(mini_ladder, rungs=rungs)
    with pytest.raises(StudyRequestError, match="group"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])


def test_spec_rejects_noop_change(mini_ladder: dict[str, Any]) -> None:
    rungs = _default_rungs(mini_ladder["identity"])
    rungs[1]["changes"]["gated_evaluator"] = "historical"
    spec = _write_ladder_spec(mini_ladder, rungs=rungs)
    with pytest.raises(StudyRequestError, match="no-op|effective"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])


def test_spec_rejects_dictionary_rung_step_of_mmap_scale_convention(
    mini_ladder: dict[str, Any],
) -> None:
    rungs = [
        {
            "id": "mini_dictionary_scale_step",
            "group": "ingestion_normalization",
            "dataset": "mini_dictionary",
            "changes": {"mmap_scale_convention": "dictionary_parity"},
        }
    ]
    groups = {"ingestion_normalization": ["mmap_scale_convention"]}
    spec = _write_ladder_spec(mini_ladder, rungs=rungs, groups=groups)

    with pytest.raises(StudyRequestError, match="mmap_scale_convention.*dictionary"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])


def test_spec_rejects_duplicate_groups(mini_ladder: dict[str, Any]) -> None:
    rungs = _default_rungs(mini_ladder["identity"])
    rungs[2] = {
        "id": "mini_rung3_again",
        "group": "reassembly_alignment",
        "dataset": "mini_generic",
        "changes": {"gated_evaluator": "historical"},
    }
    spec = _write_ladder_spec(mini_ladder, rungs=rungs)
    with pytest.raises(StudyRequestError, match="group"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])


def test_spec_rejects_endpoint_mismatch(mini_ladder: dict[str, Any]) -> None:
    endpoint = _resolve_endpoint(
        _baseline_config(mini_ladder["identity"]),
        _default_rungs(mini_ladder["identity"]),
    )
    endpoint["torch_loss_mode"] = "poisson"
    spec = _write_ladder_spec(mini_ladder, endpoint_config=endpoint)
    with pytest.raises(StudyRequestError, match="endpoint"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])


def test_spec_rejects_group_touching_invariant_field(
    mini_ladder: dict[str, Any],
) -> None:
    groups = {
        "loader_schema": ["loader", "dataset"],
        "reassembly_alignment": ["gated_evaluator", "seed"],
        "inference_varpro": ["varpro_scaling"],
    }
    spec = _write_ladder_spec(mini_ladder, groups=groups)
    with pytest.raises(StudyRequestError, match="invariant|seed"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])


def test_spec_rejects_loader_expression_incoherence(
    mini_ladder: dict[str, Any],
) -> None:
    rungs = _default_rungs(mini_ladder["identity"])
    rungs[0]["dataset"] = "mini_dictionary"
    rungs[0]["changes"] = {"loader": "mmap", "dataset": "mini_dictionary"}
    with pytest.raises(StudyRequestError, match="expression|no-op|effective"):
        load_ladder_spec(
            _write_ladder_spec(mini_ladder, rungs=rungs),
            base_dir=mini_ladder["tmp_path"],
        )


def test_spec_rejects_unlocked_gate_provenance_value(
    mini_ladder: dict[str, Any],
) -> None:
    spec = _write_ladder_spec(
        mini_ladder, gate_overrides={"threshold_provenance": "whatever"}
    )
    with pytest.raises(StudyRequestError, match="provenance"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])


# ---------------------------------------------------------------------------
# Gate math
# ---------------------------------------------------------------------------


def _gate(**overrides: Any) -> Any:
    from scripts.studies.ablation.runtime_ladder_spec import LadderGate

    values = {
        "policy": "retained_ssim_v1",
        "threshold_provenance": "locked",
        "retained_amp_ssim_min_fraction": 0.85,
        "retained_phase_ssim_min_fraction": 0.85,
        "absolute_amp_ssim_floor": 0.50,
    }
    values.update(overrides)
    return LadderGate(**values)


def _absolute_gate(**overrides: Any) -> Any:
    from scripts.studies.ablation.runtime_ladder_spec import LadderGate

    values = {
        "policy": "absolute_ssim_delta_v1",
        "threshold_provenance": "locked",
        "max_abs_amp_ssim_delta": 0.02,
        "max_abs_phase_ssim_delta": 0.01,
    }
    values.update(overrides)
    return LadderGate(**values)


def _control(amp: float = 0.90, phase: float = 0.96) -> Any:
    from scripts.studies.ablation.runtime_ladder import LadderControl

    return LadderControl(
        rung_id="rung0_reference",
        amp_ssim=amp,
        phase_ssim=phase,
        evidence_sha256="a" * 64,
    )


def test_gate_passes_exactly_at_retained_threshold() -> None:
    control = _control(amp=0.90, phase=0.96)
    result = evaluate_rung_gate(
        _gate(),
        "rung_x",
        amp_ssim=0.90 * 0.85,
        phase_ssim=0.96 * 0.85,
        control=control,
    )
    assert result.verdict is Verdict.PASS
    assert result.observed == pytest.approx(0.85)


def test_gate_fails_just_below_retained_amp_threshold() -> None:
    control = _control(amp=0.90, phase=0.96)
    result = evaluate_rung_gate(
        _gate(),
        "rung_x",
        amp_ssim=0.90 * 0.85 - 1e-6,
        phase_ssim=0.96,
        control=control,
    )
    assert result.verdict is Verdict.FAIL
    assert result.reason == "ladder_retained_amp_ssim_below_threshold"


def test_gate_fails_on_retained_phase() -> None:
    control = _control(amp=0.90, phase=0.96)
    result = evaluate_rung_gate(
        _gate(), "rung_x", amp_ssim=0.90, phase_ssim=0.70, control=control
    )
    assert result.verdict is Verdict.FAIL
    assert result.reason == "ladder_retained_phase_ssim_below_threshold"


def test_gate_fails_below_absolute_amp_floor() -> None:
    control = _control(amp=0.55, phase=0.60)
    result = evaluate_rung_gate(
        _gate(), "rung_x", amp_ssim=0.49, phase_ssim=0.58, control=control
    )
    assert result.verdict is Verdict.FAIL
    assert result.reason == "ladder_absolute_amp_ssim_floor_failed"


def test_gate_rejects_nonpositive_control() -> None:
    with pytest.raises(StudyRequestError, match="control"):
        evaluate_rung_gate(
            _gate(),
            "rung_x",
            amp_ssim=0.9,
            phase_ssim=0.9,
            control=_control(amp=0.0),
        )


def test_gate_protocol_failure_reason_overrides_metrics() -> None:
    result = evaluate_rung_gate(
        _gate(),
        "rung_x",
        amp_ssim=0.95,
        phase_ssim=0.99,
        control=_control(),
        failure_reason="ladder_scan_omission",
    )
    assert result.verdict is Verdict.FAIL
    assert result.reason == "ladder_scan_omission"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_abs_amp_ssim_delta", -0.01),
        ("max_abs_amp_ssim_delta", 0.03),
        ("max_abs_amp_ssim_delta", True),
        ("max_abs_phase_ssim_delta", 0.02),
        ("max_abs_phase_ssim_delta", float("inf")),
        ("max_abs_phase_ssim_delta", float("nan")),
    ],
)
def test_absolute_gate_rejects_invalid_thresholds(field: str, value: float) -> None:
    with pytest.raises(StudyRequestError, match=field):
        _absolute_gate(**{field: value})


def test_absolute_gate_requires_locked_provenance() -> None:
    with pytest.raises(StudyRequestError, match="locked"):
        _absolute_gate(threshold_provenance="proposed_task28")


def test_absolute_gate_passes_exact_boundaries_in_either_direction() -> None:
    for amp_ssim, phase_ssim in ((0.88, 0.95), (0.92, 0.97)):
        result = evaluate_rung_gate(
            _absolute_gate(),
            "rung_x",
            amp_ssim=amp_ssim,
            phase_ssim=phase_ssim,
            control=_control(),
        )
        assert result.verdict is Verdict.PASS


def test_absolute_gate_fails_amplitude_delta_independently() -> None:
    result = evaluate_rung_gate(
        _absolute_gate(),
        "rung_x",
        amp_ssim=0.879,
        phase_ssim=0.96,
        control=_control(),
    )
    assert result.verdict is Verdict.FAIL
    assert result.reason == "ladder_absolute_amp_ssim_delta_exceeded"


def test_absolute_gate_fails_phase_delta_independently() -> None:
    result = evaluate_rung_gate(
        _absolute_gate(),
        "rung_x",
        amp_ssim=0.90,
        phase_ssim=0.949,
        control=_control(),
    )
    assert result.verdict is Verdict.FAIL
    assert result.reason == "ladder_absolute_phase_ssim_delta_exceeded"


def test_absolute_gate_protocol_failure_dominates_delta_failure() -> None:
    result = evaluate_rung_gate(
        _absolute_gate(),
        "rung_x",
        amp_ssim=0.10,
        phase_ssim=0.10,
        control=_control(),
        failure_reason="ladder_scan_omission",
    )
    assert result.verdict is Verdict.FAIL
    assert result.reason == "ladder_scan_omission"


# ---------------------------------------------------------------------------
# Baseline verification (rung 0 control linkage)
# ---------------------------------------------------------------------------


def test_baseline_verifies_and_extracts_control_metrics(
    mini_ladder: dict[str, Any],
) -> None:
    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )

    control = verify_baseline(spec)

    assert control.rung_id == "rung0_reference"
    assert control.amp_ssim == pytest.approx(0.90)
    assert control.phase_ssim == pytest.approx(0.97)
    assert control.evidence_sha256 == mini_ladder["baseline_evidence_sha256"]


def test_baseline_requalifies_pinned_evidence_after_atomic_floor_repin(
    tmp_path: Path,
) -> None:
    identity = _write_dataset_files(tmp_path)
    _write_generic_twin(tmp_path, identity, name="mini_generic")
    old_bridge = _bridge_table(
        identity,
        fixture_amp_mae_max=0.0996316,
        fixture_phase_mae_max=0.1583743,
        fixture_amp_ssim_min=0.8408511,
        fixture_phase_ssim_min=0.9404217,
    )
    reference_spec = _write_spec(
        tmp_path,
        identity,
        references=[_hybrid_reference(identity, bridge=old_bridge)],
    )
    with pytest.MonkeyPatch.context() as patcher:
        _Harness(patcher, identity)
        outcome = runtime.run_reference_qualification(
            runtime.ReferenceQualificationRequest(
                spec=reference_spec,
                train_npz=identity["train"],
                test_npz=identity["test"],
                output_root=tmp_path / "baseline_out",
                base_dir=tmp_path,
            )
        )
    assert outcome.passed
    evidence = tmp_path / "baseline_out" / HYBRID_ID / "reference_evidence.json"

    _write_spec(
        tmp_path,
        identity,
        references=[_hybrid_reference(identity)],
    )
    mini = {
        "identity": identity,
        "tmp_path": tmp_path,
        "reference_spec": reference_spec,
        "baseline_evidence": evidence,
        "baseline_evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    }
    spec = load_ladder_spec(_write_ladder_spec(mini), base_dir=tmp_path)

    control = verify_baseline(spec)

    assert control.amp_ssim == pytest.approx(0.90)
    assert control.phase_ssim == pytest.approx(0.97)

    stricter_bridge = _bridge_table(identity, fixture_amp_ssim_min=0.91)
    _write_spec(
        tmp_path,
        identity,
        references=[_hybrid_reference(identity, bridge=stricter_bridge)],
    )
    stricter_spec = load_ladder_spec(_write_ladder_spec(mini), base_dir=tmp_path)
    with pytest.raises(StudyRequestError, match="current re-pinned"):
        verify_baseline(stricter_spec)

    changed_bridge = _bridge_table(identity, probe_scale=5.0)
    _write_spec(
        tmp_path,
        identity,
        references=[_hybrid_reference(identity, bridge=changed_bridge)],
    )
    changed_spec = load_ladder_spec(_write_ladder_spec(mini), base_dir=tmp_path)
    with pytest.raises(
        StudyRequestError, match="integration_bridge_unclassified_difference"
    ):
        verify_baseline(changed_spec)


def test_baseline_sha_mismatch_fails_closed(mini_ladder: dict[str, Any]) -> None:
    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder, baseline_evidence_sha256="b" * 64),
        base_dir=mini_ladder["tmp_path"],
    )
    with pytest.raises(StudyRequestError, match="sha|hash|pin"):
        verify_baseline(spec)


def test_baseline_below_floor_evidence_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity = _write_dataset_files(tmp_path)
    _write_generic_twin(tmp_path, identity, name="mini_generic")
    reference_spec = _write_spec(
        tmp_path, identity, references=[_hybrid_reference(identity)]
    )
    with pytest.MonkeyPatch.context() as patcher:
        _Harness(
            patcher, identity, metrics={"mae": (0.05, 0.10), "ssim": (0.70, 0.97)}
        )
        outcome = runtime.run_reference_qualification(
            runtime.ReferenceQualificationRequest(
                spec=reference_spec,
                train_npz=identity["train"],
                test_npz=identity["test"],
                output_root=tmp_path / "baseline_out",
                base_dir=tmp_path,
            )
        )
    assert not outcome.passed
    evidence = tmp_path / "baseline_out" / HYBRID_ID / "reference_evidence.json"
    mini = {
        "identity": identity,
        "tmp_path": tmp_path,
        "reference_spec": reference_spec,
        "baseline_evidence": evidence,
        "baseline_evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    }
    spec = load_ladder_spec(_write_ladder_spec(mini), base_dir=tmp_path)
    with pytest.raises(StudyRequestError, match="baseline"):
        verify_baseline(spec)


# ---------------------------------------------------------------------------
# Rung dataset recipes (recipe-pinned, content-fingerprinted)
# ---------------------------------------------------------------------------


def test_generic_twin_pair_validates_and_fingerprints(
    mini_ladder: dict[str, Any],
) -> None:
    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    from scripts.studies.ablation.datasets import validate_ladder_npz_pair

    dataset = spec.dataset("mini_generic")
    train = mini_ladder["tmp_path"] / "mini_generic" / "train.npz"
    test = mini_ladder["tmp_path"] / "mini_generic" / "test.npz"

    materialized = validate_ladder_npz_pair(dataset, train, test)

    assert materialized.train_sha256 == hashlib.sha256(train.read_bytes()).hexdigest()
    assert materialized.test_sha256 == hashlib.sha256(test.read_bytes()).hexdigest()
    assert (
        materialized.probe_sha256
        == mini_ladder["identity"]["transformed_probe_sha256"]
    )
    assert materialized.n_train == 6
    assert materialized.n_test == 3
    assert materialized.recipe_fingerprint_sha256 == dataset.fingerprint_sha256
    assert dataset.fingerprint_sha256 != dataset.recipe.fingerprint_sha256


def test_generic_twin_rejects_probe_identity_mismatch(
    mini_ladder: dict[str, Any],
) -> None:
    from scripts.studies.ablation.datasets import validate_ladder_npz_pair

    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    dataset = spec.dataset("mini_generic")
    train = mini_ladder["tmp_path"] / "mini_generic" / "train.npz"
    test = mini_ladder["tmp_path"] / "mini_generic" / "test.npz"
    rng = np.random.default_rng(99)
    with np.load(train) as data:
        payload = {key: data[key] for key in data.files}
    payload["probeGuess"] = (
        rng.normal(size=(N_SMALL, N_SMALL)) + 1j * rng.normal(size=(N_SMALL, N_SMALL))
    ).astype(np.complex64)
    np.savez(train, **payload)
    with pytest.raises(DatasetError, match="probe"):
        validate_ladder_npz_pair(dataset, train, test)


def test_generic_twin_rejects_missing_generic_keys(
    mini_ladder: dict[str, Any],
) -> None:
    from scripts.studies.ablation.datasets import validate_ladder_npz_pair

    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    dataset = spec.dataset("mini_generic")
    train = mini_ladder["tmp_path"] / "mini_generic" / "train.npz"
    test = mini_ladder["tmp_path"] / "mini_generic" / "test.npz"
    with np.load(train) as data:
        payload = {key: data[key] for key in data.files if key != "xcoords"}
    np.savez(train, **payload)
    with pytest.raises(DatasetError, match="xcoords"):
        validate_ladder_npz_pair(dataset, train, test)


def test_count_twin_rejects_negative_measurements(
    mini_ladder: dict[str, Any],
) -> None:
    from scripts.studies.ablation.datasets import validate_ladder_npz_pair

    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    dataset = spec.dataset("mini_generic_counts")
    twin = _write_generic_twin(
        mini_ladder["tmp_path"], mini_ladder["identity"], name="mini_counts"
    )
    with np.load(twin["train"]) as data:
        payload = {key: data[key] for key in data.files}
    payload["diff3d"] = payload["diff3d"] - 5.0
    np.savez(twin["train"], **payload)
    with pytest.raises(DatasetError, match="negative|count"):
        validate_ladder_npz_pair(dataset, twin["train"], twin["test"])


# ---------------------------------------------------------------------------
# Dry run (no NPZ, no torch)
# ---------------------------------------------------------------------------


def test_dry_run_needs_no_npz_and_lists_rungs(mini_ladder: dict[str, Any]) -> None:
    outcome = run_bridge_ladder(
        LadderRequest(
            spec=_write_ladder_spec(mini_ladder),
            dry_run=True,
            base_dir=mini_ladder["tmp_path"],
        )
    )

    assert outcome.plan is not None
    assert "mini_rung1_loader" in outcome.plan
    assert "mini_rung3_varpro" in outcome.plan
    assert "retained_ssim_v1" in outcome.plan
    assert outcome.results == ()
    assert outcome.report_path is None


def test_dry_run_validates_staged_datasets_fail_closed(
    mini_ladder: dict[str, Any],
) -> None:
    """--dry-run with --datasets-root validates every rung dataset's staged
    pair (no torch, no training) and fails closed on missing or invalid
    materializations (Task 21b deliverable 4)."""
    spec_path = _write_ladder_spec(mini_ladder)
    root = _datasets_root(mini_ladder)

    outcome = run_bridge_ladder(
        LadderRequest(
            spec=spec_path,
            dry_run=True,
            datasets_root=root,
            base_dir=mini_ladder["tmp_path"],
        )
    )
    assert outcome.plan is not None
    train_sha = hashlib.sha256(
        (root / "mini_generic" / "train.npz").read_bytes()
    ).hexdigest()
    assert f"train_sha256={train_sha}" in outcome.plan
    assert "dataset mini_generic staged" in outcome.plan

    # Corrupt the staged probe: validation must fail closed.
    train = root / "mini_generic" / "train.npz"
    with np.load(train) as data:
        payload = {key: data[key] for key in data.files}
    payload["probeGuess"] = payload["probeGuess"] * np.complex64(2.0)
    np.savez(train, **payload)
    with pytest.raises(StudyRequestError, match="probe|invalid"):
        run_bridge_ladder(
            LadderRequest(
                spec=spec_path,
                dry_run=True,
                datasets_root=root,
                base_dir=mini_ladder["tmp_path"],
            )
        )

    # A missing pair also fails closed when validation was requested.
    with pytest.raises(StudyRequestError, match="not materialized|missing"):
        run_bridge_ladder(
            LadderRequest(
                spec=spec_path,
                dry_run=True,
                datasets_root=mini_ladder["tmp_path"] / "empty_root",
                base_dir=mini_ladder["tmp_path"],
            )
        )


# ---------------------------------------------------------------------------
# Sealed rung evidence round-trip
# ---------------------------------------------------------------------------


def test_rung_evidence_seal_round_trip(tmp_path: Path) -> None:
    payload = {
        "schema_version": "bridge_ladder_rung_evidence_v1",
        "rung_id": "rung_x",
        "value": 1.0,
        "gate": {
            "policy": "retained_ssim_v1",
            "threshold_provenance": "locked",
            "retained_amp_ssim_min_fraction": 0.85,
            "retained_phase_ssim_min_fraction": 0.85,
            "absolute_amp_ssim_floor": 0.5,
            "verdict": "pass",
            "reason": None,
            "retained_amp_ssim": 0.9,
            "retained_phase_ssim": 0.9,
        },
    }
    path = tmp_path / "rung_evidence.json"

    sealed_sha = seal_rung_evidence(payload, path)
    parsed, parsed_sha = parse_sealed_rung_evidence(path.read_bytes())

    assert parsed["rung_id"] == "rung_x"
    assert parsed_sha == sealed_sha
    assert parsed_sha == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(Exception, match="overwrite"):
        seal_rung_evidence(payload, path)


def test_rung_evidence_rejects_wrong_schema(tmp_path: Path) -> None:
    data = json.dumps({"schema_version": "other", "rung_id": "x"}).encode()
    with pytest.raises(Exception, match="schema"):
        parse_sealed_rung_evidence(data)


def _retained_v2_payload(
    *,
    control_amp: float = 0.9,
    control_phase: float = 0.96,
    current_amp: float = 0.81,
    current_phase: float = 0.864,
    verdict: str = "pass",
    reason: str | None = None,
    protocol_failure_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "bridge_ladder_rung_evidence_v2",
        "rung_id": "rung_x",
        "gate": {
            "policy": "retained_ssim_v1",
            "threshold_provenance": "locked",
            "control": {"amp_ssim": control_amp, "phase_ssim": control_phase},
            "current": {
                "amp_ssim": current_amp,
                "phase_ssim": current_phase,
            },
            "retained_amp_ssim_min_fraction": 0.85,
            "retained_phase_ssim_min_fraction": 0.85,
            "absolute_amp_ssim_floor": 0.5,
            "retained_amp_ssim": current_amp / control_amp,
            "retained_phase_ssim": current_phase / control_phase,
            "protocol_failure_reason": protocol_failure_reason,
            "verdict": verdict,
            "reason": reason,
        },
    }


def test_retained_v2_evidence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "retained_v2.json"
    payload = _retained_v2_payload()
    sealed_sha = seal_rung_evidence(payload, path)

    parsed, parsed_sha = parse_sealed_rung_evidence(path.read_bytes())

    assert parsed == payload
    assert parsed_sha == sealed_sha


@pytest.mark.parametrize("mutation", ["minimal", "unknown", "missing"])
def test_retained_v2_rejects_noncanonical_field_set(mutation: str) -> None:
    payload = _retained_v2_payload()
    if mutation == "minimal":
        payload["gate"] = {"policy": "retained_ssim_v1"}
    elif mutation == "unknown":
        payload["gate"]["unknown"] = 1
    else:
        del payload["gate"]["control"]
    with pytest.raises(RuntimeExecutionError, match="retained|gate|field"):
        parse_sealed_rung_evidence(json.dumps(payload).encode("utf-8"))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("current", "amp_ssim"), float("inf")),
        (("control", "phase_ssim"), float("nan")),
        (("control", "amp_ssim"), 0.0),
        (("retained_amp_ssim_min_fraction",), float("inf")),
        (("absolute_amp_ssim_floor",), float("nan")),
    ],
)
def test_retained_v2_rejects_invalid_operands_and_thresholds(
    path: tuple[str, ...], value: float
) -> None:
    payload = _retained_v2_payload()
    target = payload["gate"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(RuntimeExecutionError, match="retained|finite|control|gate"):
        parse_sealed_rung_evidence(json.dumps(payload).encode("utf-8"))


@pytest.mark.parametrize("ratio", ["retained_amp_ssim", "retained_phase_ssim"])
def test_retained_v2_rejects_tampered_ratio(ratio: str) -> None:
    payload = _retained_v2_payload()
    payload["gate"][ratio] = 0.99
    with pytest.raises(RuntimeExecutionError, match="retained|ratio|gate"):
        parse_sealed_rung_evidence(json.dumps(payload).encode("utf-8"))


@pytest.mark.parametrize(
    ("payload", "expected_reason"),
    [
        (
            _retained_v2_payload(current_amp=0.7),
            "ladder_retained_amp_ssim_below_threshold",
        ),
        (
            _retained_v2_payload(
                current_amp=0.9,
                current_phase=0.7,
                verdict="fail",
                reason="ladder_retained_amp_ssim_below_threshold",
            ),
            "ladder_retained_phase_ssim_below_threshold",
        ),
        (
            _retained_v2_payload(
                control_amp=0.55,
                current_amp=0.49,
                verdict="fail",
                reason="ladder_retained_phase_ssim_below_threshold",
            ),
            "ladder_absolute_amp_ssim_floor_failed",
        ),
        (
            _retained_v2_payload(
                verdict="fail",
                reason="ladder_normalization_not_reused",
                protocol_failure_reason="ladder_scan_omission",
            ),
            "ladder_scan_omission",
        ),
    ],
)
def test_retained_v2_rejects_noncanonical_verdict_or_reason(
    payload: dict[str, Any], expected_reason: str
) -> None:
    with pytest.raises(RuntimeExecutionError, match=expected_reason):
        parse_sealed_rung_evidence(json.dumps(payload).encode("utf-8"))


def _absolute_v2_payload(
    *,
    current_amp: float = 0.88,
    current_phase: float = 0.96,
    verdict: str = "pass",
    reason: str | None = None,
    protocol_failure_reason: str | None = None,
) -> dict[str, Any]:
    control_amp = 0.9
    control_phase = 0.97
    return {
        "schema_version": "bridge_ladder_rung_evidence_v2",
        "rung_id": "rung_x",
        "gate": {
            "policy": "absolute_ssim_delta_v1",
            "threshold_provenance": "locked",
            "control": {"amp_ssim": control_amp, "phase_ssim": control_phase},
            "current": {
                "amp_ssim": current_amp,
                "phase_ssim": current_phase,
            },
            "abs_amp_delta": round(abs(current_amp - control_amp), 12),
            "abs_phase_delta": round(abs(current_phase - control_phase), 12),
            "max_abs_amp_ssim_delta": 0.02,
            "max_abs_phase_ssim_delta": 0.01,
            "protocol_failure_reason": protocol_failure_reason,
            "verdict": verdict,
            "reason": reason,
        },
    }


def test_absolute_v2_evidence_parser_rejects_inconsistent_delta() -> None:
    payload = _absolute_v2_payload()
    payload["gate"]["abs_amp_delta"] = 0.0
    with pytest.raises(RuntimeExecutionError, match="delta|gate"):
        parse_sealed_rung_evidence(json.dumps(payload).encode("utf-8"))


def test_absolute_v2_parser_rejects_threshold_failure_relabeled_pass() -> None:
    payload = _absolute_v2_payload(current_amp=0.87)
    with pytest.raises(RuntimeExecutionError, match="verdict|reason|gate"):
        parse_sealed_rung_evidence(json.dumps(payload).encode("utf-8"))


def test_absolute_v2_parser_rejects_pass_relabeled_failure() -> None:
    payload = _absolute_v2_payload(
        verdict="fail", reason="ladder_absolute_amp_ssim_delta_exceeded"
    )
    with pytest.raises(RuntimeExecutionError, match="verdict|reason|gate"):
        parse_sealed_rung_evidence(json.dumps(payload).encode("utf-8"))


@pytest.mark.parametrize(
    "mutation", ["missing", "null", "wrong", "pass_reason", "pass_missing"]
)
def test_absolute_v2_parser_rejects_noncanonical_reason(mutation: str) -> None:
    payload = _absolute_v2_payload(
        current_amp=0.87,
        verdict="fail",
        reason="ladder_absolute_amp_ssim_delta_exceeded",
    )
    if mutation == "missing":
        del payload["gate"]["reason"]
    elif mutation == "null":
        payload["gate"]["reason"] = None
    elif mutation == "wrong":
        payload["gate"]["reason"] = "ladder_absolute_phase_ssim_delta_exceeded"
    elif mutation == "pass_reason":
        payload = _absolute_v2_payload(
            reason="ladder_absolute_amp_ssim_delta_exceeded"
        )
    else:
        payload = _absolute_v2_payload()
        del payload["gate"]["reason"]
    with pytest.raises(RuntimeExecutionError, match="verdict|reason|gate|field"):
        parse_sealed_rung_evidence(json.dumps(payload).encode("utf-8"))


def test_absolute_v2_parser_accepts_canonical_protocol_failure() -> None:
    payload = _absolute_v2_payload(
        verdict="fail",
        reason="ladder_scan_omission",
        protocol_failure_reason="ladder_scan_omission",
    )
    parsed, _ = parse_sealed_rung_evidence(json.dumps(payload).encode("utf-8"))
    assert parsed["gate"]["reason"] == "ladder_scan_omission"


@pytest.mark.parametrize(
    ("protocol_reason", "final_reason"),
    [
        ("ladder_scan_omission", "ladder_normalization_not_reused"),
        ("not_a_protocol_reason", "not_a_protocol_reason"),
        (["ladder_scan_omission"], "ladder_scan_omission"),
    ],
)
def test_absolute_v2_parser_rejects_inconsistent_protocol_failure(
    protocol_reason: Any, final_reason: str
) -> None:
    payload = _absolute_v2_payload(
        verdict="fail",
        reason=final_reason,
        protocol_failure_reason=protocol_reason,
    )
    with pytest.raises(RuntimeExecutionError, match="protocol|reason|gate"):
        parse_sealed_rung_evidence(json.dumps(payload).encode("utf-8"))


# ---------------------------------------------------------------------------
# Stubbed end-to-end ladder walk
# ---------------------------------------------------------------------------


def test_ladder_walk_passes_and_seals_per_rung_evidence(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(mini_ladder)
    stub = _ExecutorStub()

    outcome = _run_walk(mini_ladder, spec_path, monkeypatch, stub)

    assert outcome.passed
    assert outcome.first_material_degradation is None
    assert stub.executed == [
        "mini_rung1_loader",
        "mini_rung2_evaluator",
        "mini_rung3_varpro",
    ]
    out = mini_ladder["tmp_path"] / "ladder_out"
    report = json.loads((out / "ladder_report.json").read_text())
    assert report["first_material_degradation"] is None
    assert [entry["id"] for entry in report["rungs"]] == stub.executed
    for index, entry in enumerate(report["rungs"]):
        assert entry["verdict"] == "pass"
        evidence_path = out / entry["id"] / "rung_evidence.json"
        assert evidence_path.is_file()
        payload, sha = parse_sealed_rung_evidence(evidence_path.read_bytes())
        assert sha == entry["evidence_sha256"]
        expected_control = (
            "rung0_reference" if index == 0 else stub.executed[index - 1]
        )
        assert payload["control"]["rung_id"] == expected_control
        assert payload["gate"]["verdict"] == "pass"
    first = json.loads(
        (out / "mini_rung1_loader" / "rung_evidence.json").read_bytes()
    )
    assert first["control"]["evidence_sha256"] == (
        mini_ladder["baseline_evidence_sha256"]
    )


def test_ladder_report_bytes_are_output_location_independent(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(mini_ladder)
    roots = [mini_ladder["tmp_path"] / name for name in ("report_a", "report_b")]

    for root in roots:
        _run_walk(
            mini_ladder,
            spec_path,
            monkeypatch,
            _ExecutorStub(),
            output_root=root,
        )

    reports = [(root / "ladder_report.json").read_bytes() for root in roots]
    assert reports[0] == reports[1]
    assert hashlib.sha256(reports[0]).hexdigest() == hashlib.sha256(
        reports[1]
    ).hexdigest()
    report = json.loads(reports[0])
    assert report["spec"] == "ladder_spec.toml"
    for entry in report["rungs"]:
        assert entry["evidence_path"] == f"{entry['id']}/rung_evidence.json"


def test_absolute_gate_evidence_and_report_seal_complete_operands(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(
        mini_ladder, gate_overrides={"policy": "absolute_ssim_delta_v1"}
    )
    metrics = {
        rung_id: (0.88, 0.96)
        for rung_id in (
            "mini_rung1_loader",
            "mini_rung2_evaluator",
            "mini_rung3_varpro",
        )
    }
    outcome = _run_walk(
        mini_ladder, spec_path, monkeypatch, _ExecutorStub(metrics=metrics)
    )

    assert outcome.passed
    assert outcome.first_material_degradation is None

    out = mini_ladder["tmp_path"] / "ladder_out"
    payload, _ = parse_sealed_rung_evidence(
        (out / "mini_rung1_loader" / "rung_evidence.json").read_bytes()
    )
    assert payload["schema_version"] == "bridge_ladder_rung_evidence_v2"
    assert payload["gate"] == {
        "policy": "absolute_ssim_delta_v1",
        "threshold_provenance": "locked",
        "control": {"amp_ssim": 0.9, "phase_ssim": 0.97},
        "current": {"amp_ssim": 0.88, "phase_ssim": 0.96},
        "abs_amp_delta": pytest.approx(0.02),
        "abs_phase_delta": pytest.approx(0.01),
        "max_abs_amp_ssim_delta": 0.02,
        "max_abs_phase_ssim_delta": 0.01,
        "protocol_failure_reason": None,
        "verdict": "pass",
        "reason": None,
    }
    report = json.loads((out / "ladder_report.json").read_text())
    assert report["gate"] == {
        "policy": "absolute_ssim_delta_v1",
        "threshold_provenance": "locked",
        "max_abs_amp_ssim_delta": 0.02,
        "max_abs_phase_ssim_delta": 0.01,
    }
    assert report["first_material_degradation"] is None
    assert report["rungs"][0]["gate"] == payload["gate"]
    assert {entry["control_rung_id"] for entry in report["rungs"]} == {
        "rung0_reference"
    }


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_absolute_gate_evidence_missing_or_tampered_fields_fail_closed(
    mini_ladder: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    spec_path = _write_ladder_spec(
        mini_ladder, gate_overrides={"policy": "absolute_ssim_delta_v1"}
    )
    _run_walk(mini_ladder, spec_path, monkeypatch, _ExecutorStub())
    evidence = (
        mini_ladder["tmp_path"]
        / "ladder_out"
        / "mini_rung1_loader"
        / "rung_evidence.json"
    )
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    if mutation == "missing":
        del payload["gate"]["current"]
    else:
        payload["gate"]["abs_amp_delta"] = 0.0
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((StudyRequestError, RuntimeExecutionError), match="gate|evidence"):
        _run_walk(mini_ladder, spec_path, monkeypatch, _ExecutorStub())


def test_injected_degradation_lands_on_the_right_rung(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(mini_ladder)
    # Rung 2 loses 30% of amplitude SSIM relative to rung 1: material.
    stub = _ExecutorStub(
        metrics={
            "mini_rung1_loader": (0.88, 0.95),
            "mini_rung2_evaluator": (0.60, 0.94),
            "mini_rung3_varpro": (0.87, 0.94),
        }
    )

    outcome = _run_walk(mini_ladder, spec_path, monkeypatch, stub)

    assert not outcome.passed
    assert outcome.first_material_degradation == "mini_rung2_evaluator"
    verdicts = {result.id: result for result in outcome.results}
    assert verdicts["mini_rung2_evaluator"].verdict is Verdict.FAIL
    assert (
        verdicts["mini_rung2_evaluator"].reason
        == "ladder_retained_amp_ssim_below_threshold"
    )
    assert verdicts["mini_rung3_varpro"].verdict is Verdict.PASS


def test_rung_after_failed_rung_gates_against_last_passing_control(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(mini_ladder)
    stub = _ExecutorStub(
        metrics={
            "mini_rung1_loader": (0.88, 0.95),
            "mini_rung2_evaluator": (0.60, 0.94),
            "mini_rung3_varpro": (0.87, 0.94),
        }
    )
    _run_walk(mini_ladder, spec_path, monkeypatch, stub)

    out = mini_ladder["tmp_path"] / "ladder_out"
    rung3, _ = parse_sealed_rung_evidence(
        (out / "mini_rung3_varpro" / "rung_evidence.json").read_bytes()
    )
    # The failed rung 2 must NOT become the control; rung 1 (last PASS) is.
    assert rung3["control"]["rung_id"] == "mini_rung1_loader"
    rung1, rung1_sha = parse_sealed_rung_evidence(
        (out / "mini_rung1_loader" / "rung_evidence.json").read_bytes()
    )
    assert rung3["control"]["evidence_sha256"] == rung1_sha
    assert rung1 is not None


def test_existing_rung_evidence_is_reused_not_reexecuted(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(mini_ladder)
    first_stub = _ExecutorStub()
    _run_walk(mini_ladder, spec_path, monkeypatch, first_stub)

    second_stub = _ExecutorStub()
    outcome = _run_walk(mini_ladder, spec_path, monkeypatch, second_stub)

    assert outcome.passed
    assert second_stub.executed == []


def test_tampered_rung_evidence_fails_closed(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(mini_ladder)
    _run_walk(mini_ladder, spec_path, monkeypatch, _ExecutorStub())
    evidence = (
        mini_ladder["tmp_path"]
        / "ladder_out"
        / "mini_rung2_evaluator"
        / "rung_evidence.json"
    )
    payload = json.loads(evidence.read_text())
    payload["metrics"]["amp_ssim"] = 0.999
    evidence.write_text(json.dumps(payload))

    with pytest.raises(StudyRequestError, match="evidence|tamper|control"):
        _run_walk(mini_ladder, spec_path, monkeypatch, _ExecutorStub())


def test_single_rung_selection_requires_prior_evidence(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(mini_ladder)
    stub = _ExecutorStub()

    with pytest.raises(StudyRequestError, match="prior|missing"):
        _run_walk(mini_ladder, spec_path, monkeypatch, stub, rung="mini_rung2_evaluator")
    assert stub.executed == []

    _run_walk(mini_ladder, spec_path, monkeypatch, stub, rung="mini_rung1_loader")
    assert stub.executed == ["mini_rung1_loader"]
    outcome = _run_walk(
        mini_ladder, spec_path, monkeypatch, stub, rung="mini_rung2_evaluator"
    )
    assert stub.executed == ["mini_rung1_loader", "mini_rung2_evaluator"]
    assert outcome.first_material_degradation is None


def test_grouping_rung_fails_closed_on_scan_omission(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rungs = [
        {
            "id": "mini_rung1_loader",
            "group": "loader_schema",
            "dataset": "mini_generic",
            "changes": {"loader": "mmap", "dataset": "mini_generic"},
        },
        {
            "id": "mini_rung2_grouping",
            "group": "grouping_weighting",
            "dataset": "mini_generic",
            "changes": {"gridsize": 2, "training_patch_weighting": "probe"},
            "requires_scan_accounting": True,
        },
    ]
    spec_path = _write_ladder_spec(mini_ladder, rungs=rungs)
    accounting = {
        "unique_scans_used": 2,
        "unique_scans_expected": 3,
        "scan_utilization_fraction": 2.0 / 3.0,
        "duplicate_scan_uses": 0,
        "group_count": 2,
        "accepted_patch_count": 2,
        "reconstructed_pixel_count": 500,
        "canvas_coverage_fraction": 0.9,
    }
    stub = _ExecutorStub(scan_accounting=accounting)

    outcome = _run_walk(mini_ladder, spec_path, monkeypatch, stub)

    verdicts = {result.id: result for result in outcome.results}
    assert verdicts["mini_rung2_grouping"].verdict is Verdict.FAIL
    assert verdicts["mini_rung2_grouping"].reason == "ladder_scan_omission"
    assert outcome.first_material_degradation == "mini_rung2_grouping"
    payload, _ = parse_sealed_rung_evidence(
        (
            mini_ladder["tmp_path"]
            / "ladder_out"
            / "mini_rung2_grouping"
            / "rung_evidence.json"
        ).read_bytes()
    )
    assert payload["scan_accounting"]["unique_scans_expected"] == 3
    assert payload["scan_accounting"]["group_count"] == 2


def test_grouping_rung_fails_closed_without_accounting_evidence(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rungs = [
        {
            "id": "mini_rung1_loader",
            "group": "loader_schema",
            "dataset": "mini_generic",
            "changes": {"loader": "mmap", "dataset": "mini_generic"},
        },
        {
            "id": "mini_rung2_grouping",
            "group": "grouping_weighting",
            "dataset": "mini_generic",
            "changes": {"gridsize": 2, "training_patch_weighting": "probe"},
            "requires_scan_accounting": True,
        },
    ]
    spec_path = _write_ladder_spec(mini_ladder, rungs=rungs)
    stub = _ExecutorStub(scan_accounting=None)

    with pytest.raises(StudyRequestError, match="scan"):
        _run_walk(mini_ladder, spec_path, monkeypatch, stub)


def test_normalization_recompute_fails_the_rung(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rungs = _default_rungs(mini_ladder["identity"])
    rungs[2]["requires_normalization_evidence"] = True
    spec_path = _write_ladder_spec(mini_ladder, rungs=rungs)

    class _RecomputeStub(_ExecutorStub):
        def __call__(self, spec: Any, rung: Any, **kwargs: Any) -> LadderRunResult:
            result = super().__call__(spec, rung, **kwargs)
            if rung.id == "mini_rung3_varpro":
                import dataclasses

                result = dataclasses.replace(
                    result,
                    inference_reuses_training_normalization=False,
                    inference_normalization_sha256=hashlib.sha256(
                        b"heldout-stats"
                    ).hexdigest(),
                )
            return result

    outcome = _run_walk(mini_ladder, spec_path, monkeypatch, _RecomputeStub())

    verdicts = {result.id: result for result in outcome.results}
    assert verdicts["mini_rung3_varpro"].verdict is Verdict.FAIL
    assert verdicts["mini_rung3_varpro"].reason == "ladder_normalization_not_reused"
    payload, _ = parse_sealed_rung_evidence(
        (
            mini_ladder["tmp_path"]
            / "ladder_out"
            / "mini_rung3_varpro"
            / "rung_evidence.json"
        ).read_bytes()
    )
    assert payload["normalization"]["inference_reuses_training_normalization"] is False


def test_unclassified_canvas_difference_fails_closed_in_walk(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(mini_ladder)

    class _DivergentStub(_ExecutorStub):
        def __call__(self, spec: Any, rung: Any, **kwargs: Any) -> LadderRunResult:
            result = super().__call__(spec, rung, **kwargs)
            if rung.id == "mini_rung2_evaluator":
                import dataclasses

                result = dataclasses.replace(
                    result,
                    generic_canvas_sha256=hashlib.sha256(b"other").hexdigest(),
                    canvases_equivalent=False,
                )
            return result

    outcome = _run_walk(mini_ladder, spec_path, monkeypatch, _DivergentStub())

    verdicts = {result.id: result for result in outcome.results}
    assert verdicts["mini_rung2_evaluator"].verdict is Verdict.FAIL
    assert verdicts["mini_rung2_evaluator"].reason == "ladder_unclassified_difference"


def test_predeclared_harmless_difference_still_passes(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rungs = _default_rungs(mini_ladder["identity"])
    rungs[1]["expected_differences"] = {
        "canvas_equivalence": {
            "classification": "harmless",
            "justification": "dual stitches share the declared crop boundary",
        }
    }
    spec_path = _write_ladder_spec(mini_ladder, rungs=rungs)

    class _DivergentStub(_ExecutorStub):
        def __call__(self, spec: Any, rung: Any, **kwargs: Any) -> LadderRunResult:
            result = super().__call__(spec, rung, **kwargs)
            if rung.id == "mini_rung2_evaluator":
                import dataclasses

                result = dataclasses.replace(
                    result,
                    generic_canvas_sha256=hashlib.sha256(b"other").hexdigest(),
                    canvases_equivalent=False,
                )
            return result

    outcome = _run_walk(mini_ladder, spec_path, monkeypatch, _DivergentStub())

    verdicts = {result.id: result for result in outcome.results}
    assert verdicts["mini_rung2_evaluator"].verdict is Verdict.PASS
    payload, _ = parse_sealed_rung_evidence(
        (
            mini_ladder["tmp_path"]
            / "ladder_out"
            / "mini_rung2_evaluator"
            / "rung_evidence.json"
        ).read_bytes()
    )
    (difference,) = payload["recorded_differences"]
    assert difference["field"] == "canvas_equivalence"
    assert difference["classification"] == "harmless"


def test_seed_override_flows_into_evidence(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(mini_ladder)
    outcome = _run_walk(
        mini_ladder,
        spec_path,
        monkeypatch,
        _ExecutorStub(),
        seed=17,
        output_root=mini_ladder["tmp_path"] / "ladder_out_seed17",
    )

    assert outcome.passed
    payload, _ = parse_sealed_rung_evidence(
        (
            mini_ladder["tmp_path"]
            / "ladder_out_seed17"
            / "mini_rung1_loader"
            / "rung_evidence.json"
        ).read_bytes()
    )
    assert payload["resolved_config"]["seed"] == 17


# ---------------------------------------------------------------------------
# execute_ladder_rung: canonical entry-point dispatch (deep stubs)
# ---------------------------------------------------------------------------


class _ExecHarness:
    """Stub the canonical mmap-flow entry points and record the call order."""

    CHECKPOINT_BYTES = b"ladder-checkpoint-payload"

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        metrics: dict[str, tuple[float, float]] | None = None,
        effective_probe_sha256: str | None = None,
        physics_scaling_constant: float | None = None,
    ) -> None:
        from ptycho import evaluation
        from ptycho_torch import lightning_utils

        from scripts.studies import grid_lines_torch_runner as runner_mod
        from scripts.studies.ablation import runtime_ladder_execution

        self.calls: list[tuple[Any, ...]] = []
        self.results: dict[str, Any] = {
            "training_normalization": {"regime": "legacy", "scaling_constant": 1.0}
        }
        if effective_probe_sha256 is not None:
            self.results["effective_probe_sha256"] = effective_probe_sha256
        if physics_scaling_constant is not None:
            self.results["physics_scaling_constant"] = physics_scaling_constant
        self.metrics = metrics or {"mae": (0.05, 0.10), "ssim": (0.88, 0.95)}
        self.model = object()
        rng = np.random.default_rng(31)
        self.patches = rng.normal(size=(3, N_SMALL, N_SMALL, 1, 2)).astype(np.float32)
        self.historical_canvas = (
            rng.normal(size=GT_SHAPE) + 1j * rng.normal(size=GT_SHAPE)
        ).astype(np.complex64)
        self.generic_canvas = self.historical_canvas + np.complex64(0.25)
        self.evaluated_canvases: list[np.ndarray] = []
        harness = self

        def train_via_generic_loader(
            runner_cfg: Any,
            config: Any,
            recipe: Any,
            train_npz: Path,
            test_npz: Path,
            work: Path,
        ) -> tuple[Any, Path, dict[str, Any]]:
            harness.calls.append(
                ("train_via_generic_loader", Path(train_npz).name, config["seed"])
            )
            checkpoint = Path(work) / "checkpoints" / "best.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(harness.CHECKPOINT_BYTES)
            return harness.model, Path(work), dict(harness.results)

        def find_best_checkpoint(root: Any) -> Path | None:
            harness.calls.append(("find_best_checkpoint",))
            candidates = sorted(Path(root).rglob("*.ckpt"))
            return candidates[0] if candidates else None

        def run_torch_inference(
            model: Any, test_data: dict, cfg: Any, metadata: Any = None
        ) -> np.ndarray:
            harness.calls.append(("run_torch_inference", model is harness.model))
            return harness.patches

        def reassemble(
            pred: np.ndarray,
            ground_truth: np.ndarray,
            test_data: dict,
            metadata: Any,
            cfg: Any,
        ) -> tuple[np.ndarray, str, dict]:
            harness.calls.append(("reassemble", cfg.reassembly_mode))
            if cfg.reassembly_mode == "grid_lines":
                return harness.historical_canvas.copy(), "grid_lines", {}
            return harness.generic_canvas.copy(), "position", {}

        def eval_reconstruction(
            stitched_obj: np.ndarray, ground_truth_obj: np.ndarray, **kwargs: Any
        ) -> dict[str, Any]:
            harness.calls.append(("eval_reconstruction", kwargs.get("label")))
            harness.evaluated_canvases.append(np.squeeze(stitched_obj))
            return {key: tuple(value) for key, value in harness.metrics.items()}

        monkeypatch.setattr(
            runtime_ladder_execution,
            "train_via_generic_loader",
            train_via_generic_loader,
        )
        monkeypatch.setattr(
            lightning_utils, "find_best_checkpoint", find_best_checkpoint
        )
        monkeypatch.setattr(runner_mod, "run_torch_inference", run_torch_inference)
        monkeypatch.setattr(
            runner_mod, "_reassemble_predictions_for_metrics", reassemble
        )
        monkeypatch.setattr(evaluation, "eval_reconstruction", eval_reconstruction)


def test_execute_ladder_rung_drives_mmap_flow_and_gates_historical_canvas(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.studies.ablation import runtime_ladder_execution

    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    harness = _ExecHarness(
        monkeypatch,
        effective_probe_sha256=mini_ladder["identity"]["transformed_probe_sha256"],
    )
    twin = mini_ladder["tmp_path"] / "mini_generic"

    result = runtime_ladder_execution.execute_ladder_rung(
        spec,
        spec.rung("mini_rung1_loader"),
        train_npz=twin / "train.npz",
        test_npz=twin / "test.npz",
        work_dir=mini_ladder["tmp_path"] / "exec_work",
        seed=17,
    )

    assert [call[0] for call in harness.calls] == [
        "train_via_generic_loader",
        "find_best_checkpoint",
        "run_torch_inference",
        "reassemble",
        "reassemble",
        "eval_reconstruction",
    ]
    assert (
        result.checkpoint_sha256
        == hashlib.sha256(harness.CHECKPOINT_BYTES).hexdigest()
    )
    assert result.gated_evaluator == "historical"
    (evaluated,) = harness.evaluated_canvases
    np.testing.assert_array_equal(evaluated, harness.historical_canvas)
    assert not result.canvases_equivalent
    assert result.effective_probe_matches_recipe
    assert result.resolved_config["seed"] == 17
    assert result.varpro_applied is False
    assert result.amp_ssim == 0.88
    # Coverage is measured on the gated canvas (fully finite stub canvas).
    assert result.canvas_coverage_fraction == 1.0
    assert result.count_consistency is None
    assert result.physics_scaling_constant is None


def test_execute_ladder_rung_routes_varpro_and_generic_evaluator(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.studies.ablation import runtime_ladder_execution

    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    harness = _ExecHarness(
        monkeypatch,
        effective_probe_sha256=mini_ladder["identity"]["transformed_probe_sha256"],
    )

    def fake_varpro(
        canvas: np.ndarray, test_data: Any, cfg: Any, patches: Any
    ) -> tuple[np.ndarray, float, float]:
        return canvas * np.complex64(2.0), 2.0, 3.0

    monkeypatch.setattr(
        runtime_ladder_execution, "apply_varpro_to_canvas", fake_varpro
    )
    twin = mini_ladder["tmp_path"] / "mini_generic"

    result = runtime_ladder_execution.execute_ladder_rung(
        spec,
        spec.rung("mini_rung3_varpro"),
        train_npz=twin / "train.npz",
        test_npz=twin / "test.npz",
        work_dir=mini_ladder["tmp_path"] / "exec_work_varpro",
    )

    assert result.gated_evaluator == "generic"
    assert result.varpro_applied is True
    assert result.varpro_s1 == 2.0
    assert result.varpro_s2 == 3.0
    (evaluated,) = harness.evaluated_canvases
    np.testing.assert_allclose(
        evaluated, harness.generic_canvas * np.complex64(2.0), rtol=1e-6
    )


def test_injected_unwired_seam_still_fails_closed(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-closed seam contract: an unwired (raising) seam injected in place
    of a real one must abort the rung, never fabricate evidence."""
    from scripts.studies.ablation import runtime_ladder_execution
    from scripts.studies.ablation.runtime_execution import RuntimeExecutionError

    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    _ExecHarness(
        monkeypatch,
        effective_probe_sha256=mini_ladder["identity"]["transformed_probe_sha256"],
    )

    def unwired_varpro(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeExecutionError("varpro", "seam not wired (injected)")

    monkeypatch.setattr(
        runtime_ladder_execution, "apply_varpro_to_canvas", unwired_varpro
    )
    twin = mini_ladder["tmp_path"] / "mini_generic"

    with pytest.raises(RuntimeExecutionError, match="not wired"):
        runtime_ladder_execution.execute_ladder_rung(
            spec,
            spec.rung("mini_rung3_varpro"),
            train_npz=twin / "train.npz",
            test_npz=twin / "test.npz",
            work_dir=mini_ladder["tmp_path"] / "exec_work_injected",
        )


# ---------------------------------------------------------------------------
# Task 21b: real seam wirings (tiny real arrays through the actual functions)
# ---------------------------------------------------------------------------


def _rung1_payload(mini_ladder: dict[str, Any], **config_overrides: Any) -> Any:
    from scripts.studies.ablation import runtime_ladder_mmap

    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    rung = spec.rung("mini_rung1_loader")
    config = dict(rung.resolved_config)
    config.update(config_overrides)
    recipe = spec.dataset("mini_generic").recipe
    train = mini_ladder["tmp_path"] / "mini_generic" / "train.npz"
    return spec, config, recipe, train, runtime_ladder_mmap


def test_mmap_payload_carries_ladder_config(mini_ladder: dict[str, Any]) -> None:
    spec, config, recipe, train, mmap_mod = _rung1_payload(mini_ladder)

    payload = mmap_mod.build_mmap_dataset_payload(
        config, recipe, train, mini_ladder["tmp_path"] / "payload_out"
    )

    assert payload.pt_data_config.N == N_SMALL
    assert payload.pt_data_config.probe_normalize is False
    assert payload.pt_data_config.n_subsample == 1
    assert tuple(payload.pt_data_config.grid_size) == (1, 1)
    assert payload.pt_data_config.C == 1
    assert payload.pt_data_config.scale_contract_version == "legacy_v1"
    assert payload.pt_data_config.measurement_domain == "normalized_amplitude"
    assert payload.pt_model_config.mode == "Unsupervised"
    assert payload.execution_config.strategy == "auto"
    assert spec is not None


def test_loader_evidence_records_actual_effective_probe(
    mini_ladder: dict[str, Any],
) -> None:
    """The recorded effective probe is the hash of the tensor the REAL mmap
    loader stored — passthrough when probe_normalize=false, the canonical
    normalize_probe_like_tf product when true."""
    from ptycho_torch import helper as hh

    from scripts.studies.ablation.dataset_provenance import canonical_array_sha256

    _, config, recipe, train, mmap_mod = _rung1_payload(mini_ladder)
    work = mini_ladder["tmp_path"] / "mmap_work_passthrough"
    payload = mmap_mod.build_mmap_dataset_payload(config, recipe, train, work)
    dataset = mmap_mod.build_generic_loader_dataset(train, payload, work)
    evidence = mmap_mod.extract_loader_evidence(dataset)

    assert (
        evidence["effective_probe_sha256"]
        == mini_ladder["identity"]["transformed_probe_sha256"]
    )
    assert evidence["probe_scaling"] == 1.0
    assert evidence["training_normalization"]["regime"] == "legacy"
    assert "scaling_constant" in evidence["training_normalization"]

    _, config_norm, recipe, train, mmap_mod = _rung1_payload(
        mini_ladder, probe_normalize=True
    )
    work_norm = mini_ladder["tmp_path"] / "mmap_work_normalized"
    payload_norm = mmap_mod.build_mmap_dataset_payload(
        config_norm, recipe, train, work_norm
    )
    dataset_norm = mmap_mod.build_generic_loader_dataset(train, payload_norm, work_norm)
    evidence_norm = mmap_mod.extract_loader_evidence(dataset_norm)

    expected_probe, expected_scaling = hh.normalize_probe_like_tf(
        mini_ladder["identity"]["probe"], probe_scale=4.0
    )
    assert evidence_norm["effective_probe_sha256"] == canonical_array_sha256(
        np.asarray(expected_probe, dtype=np.complex64)
    )
    assert evidence_norm["effective_probe_sha256"] != (
        mini_ladder["identity"]["transformed_probe_sha256"]
    )
    assert evidence_norm["probe_scaling"] == pytest.approx(float(expected_scaling))


def test_grouped_ingestion_and_scan_accounting_via_real_loader(
    mini_ladder: dict[str, Any],
) -> None:
    from scripts.studies.ablation import runtime_ladder_mmap
    from scripts.studies.ablation.runtime_ladder_seams import collect_scan_accounting

    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    twin = _write_generic_twin(
        mini_ladder["tmp_path"],
        mini_ladder["identity"],
        name="mini_generic_grouped",
        train_count=12,
        test_count=12,
    )
    rung = spec.rung("mini_rung1_loader")
    config = dict(rung.resolved_config)
    config["gridsize"] = 2
    recipe = spec.dataset("mini_generic").recipe
    work = mini_ladder["tmp_path"] / "grouped_work"

    test_data, metadata, grouping = runtime_ladder_mmap.load_grouped_generic_test_dict(
        twin["test"], recipe, config, work
    )

    channels = 4
    groups = test_data["diffraction"].shape[0]
    assert test_data["diffraction"].shape == (groups, N_SMALL, N_SMALL, channels)
    assert test_data["coords_offsets"].shape == (groups, 1, 2, channels)
    assert test_data["coords_nominal"].shape == (groups, 1, 2, channels)
    # Cross-check against BOTH canonical relative-coordinate functions (the
    # reviewer's rung-5 guard): the dict-path conversion
    # coords_relative_from_nominal ((B,1,2,C), sign -1) and the loader's own
    # get_relative_coords ((G,C,1,2), (x,y), sign -1).
    from ptycho_torch.coords import coords_relative_from_nominal
    from ptycho_torch.patch_generator import get_relative_coords

    np.testing.assert_allclose(
        test_data["coords_nominal"],
        coords_relative_from_nominal(test_data["coords_offsets"]),
        atol=1e-5,
    )
    coords_nn = np.stack(
        [
            test_data["coords_offsets"][:, 0, 1, :],  # x
            test_data["coords_offsets"][:, 0, 0, :],  # y
        ],
        axis=-1,
    )[:, :, np.newaxis, :]
    _, loader_relative = get_relative_coords(coords_nn)
    np.testing.assert_allclose(
        test_data["coords_nominal"][:, 0, 0, :], loader_relative[:, :, 0, 1], atol=1e-5
    )
    np.testing.assert_allclose(
        test_data["coords_nominal"][:, 0, 1, :], loader_relative[:, :, 0, 0], atol=1e-5
    )
    # Drift guard: the replicated bounds filter must match the loader's own
    # eligible set for the same configs.
    payload = runtime_ladder_mmap.build_mmap_dataset_payload(
        config, recipe, twin["test"], work / "drift_payload"
    )
    loader_dataset = runtime_ladder_mmap.build_generic_loader_dataset(
        twin["test"], payload, work / "drift_guard"
    )
    assert grouping["filtered_eligible_scan_ids"] == [
        int(i) for i in loader_dataset.valid_indices_per_file[0]
    ]
    assert metadata["additional_parameters"]["offset"] == recipe.offset
    assert grouping["group_count"] == groups
    assert grouping["participant_slots"] == groups * channels
    assert set(grouping["used_scan_ids"]) <= set(grouping["expected_scan_ids"])

    weights = np.ones((8, 8), dtype=np.float64)
    accounting = collect_scan_accounting(grouping, weights)
    for field in (
        "unique_scans_used",
        "unique_scans_expected",
        "duplicate_scan_uses",
        "group_count",
        "accepted_patch_count",
        "reconstructed_pixel_count",
        "scan_utilization_fraction",
        "canvas_coverage_fraction",
    ):
        assert field in accounting, field
    assert accounting["unique_scans_expected"] == 12
    assert accounting["group_count"] == groups
    assert accounting["accepted_patch_count"] == groups * channels
    assert accounting["reconstructed_pixel_count"] == 64
    assert 0.0 < accounting["scan_utilization_fraction"] <= 1.0
    assert accounting["duplicate_scan_uses"] == (
        groups * channels - accounting["unique_scans_used"]
    )
    # Sentinel-wrapped grouping slots must be sealed with the accounting.
    assert accounting["sentinel_wrapped_slots"] == grouping["sentinel_wrapped_slots"]


def test_train_via_generic_loader_drives_real_dataset_into_lightning(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from ptycho_torch.dataloader import PtychoDataset
    from ptycho_torch.workflows import components

    from scripts.studies.ablation import runtime_ladder_mmap

    _, config, recipe, train, _ = _rung1_payload(mini_ladder)
    recorded: dict[str, Any] = {}
    model = object()

    def fake_train_with_lightning(
        train_container: Any,
        test_container: Any,
        tf_config: Any,
        execution_config: Any = None,
        overrides: Any = None,
    ) -> dict[str, Any]:
        recorded["container_type"] = type(train_container).__name__
        recorded["is_ptycho_dataset"] = isinstance(train_container, PtychoDataset)
        # BLOCKER regression (rung-1 bring-up): the strategy=None loader path
        # wraps test_container unconditionally, so it must be a REAL, sized
        # PtychoDataset — never None.
        recorded["test_is_ptycho_dataset"] = isinstance(test_container, PtychoDataset)
        recorded["test_len"] = None if test_container is None else len(test_container)
        recorded["overrides"] = dict(overrides or {})
        recorded["execution_strategy"] = execution_config.strategy
        recorded["train_data_file"] = str(tf_config.train_data_file)
        return {"history": {}, "models": {"diffraction_to_obj": model}}

    monkeypatch.setattr(
        components, "_train_with_lightning", fake_train_with_lightning
    )
    import torch

    seed_calls: list[int] = []
    real_manual_seed = torch.manual_seed
    monkeypatch.setattr(
        torch,
        "manual_seed",
        lambda value: (seed_calls.append(int(value)), real_manual_seed(value))[1],
    )
    from scripts.studies.ablation.runtime_ladder_execution import build_runner_config

    work = mini_ladder["tmp_path"] / "train_wire_work"
    runner_cfg = build_runner_config(
        config,
        train_npz=train,
        test_npz=mini_ladder["tmp_path"] / "mini_generic" / "test.npz",
        output_dir=work,
    )

    test_npz = _write_generic_twin(
        mini_ladder["tmp_path"],
        mini_ladder["identity"],
        name="mini_generic_wire",
        train_count=12,
        test_count=12,
    )["test"]
    trained, checkpoint_root, payload = runtime_ladder_mmap.train_via_generic_loader(
        runner_cfg, config, recipe, train, test_npz, work
    )

    assert trained is model
    assert Path(checkpoint_root) == work
    assert recorded["is_ptycho_dataset"] is True
    assert recorded["test_is_ptycho_dataset"] is True
    assert recorded["test_len"] and recorded["test_len"] > 0
    # Seed parity with the dictionary path (run_torch_training seeds torch).
    assert int(config["seed"]) in seed_calls
    assert payload["seed"] == int(config["seed"])
    assert payload["effective_probe"].shape == (N_SMALL, N_SMALL)
    for key in (
        "training_patch_weighting",
        "physics_forward_mode",
        "rect_s1s2_trainable",
        "scale_contract_version",
        "measurement_domain",
        "cnn_output_mode",
    ):
        assert key in recorded["overrides"], key
    assert recorded["overrides"]["probe_normalize"] is False
    assert recorded["overrides"]["n_subsample"] == 1
    assert "strategy" not in recorded["overrides"]
    assert recorded["execution_strategy"] == "auto"
    assert payload["effective_probe_sha256"] == (
        mini_ladder["identity"]["transformed_probe_sha256"]
    )
    assert payload["training_normalization"]["regime"] == "legacy"


def test_varpro_seam_recovers_known_scales(mini_ladder: dict[str, Any]) -> None:
    import torch

    from ptycho_torch import reassembly

    from scripts.studies.ablation.runtime_ladder_seams import apply_varpro_to_canvas

    rng = np.random.default_rng(5)
    count, n = 6, N_SMALL
    probe = torch.as_tensor(
        mini_ladder["identity"]["probe"], dtype=torch.complex64
    ).reshape(1, 1, 1, n, n)
    texture = rng.normal(size=(count, 1, n, n)) + 1j * rng.normal(
        size=(count, 1, n, n)
    )
    texture_t = torch.as_tensor(texture, dtype=torch.complex64)
    _, _, x1, x2, x3 = reassembly.compute_varpro_basis(
        probe.expand(count, 1, 1, n, n), texture_t.real, texture_t.imag
    )
    s1_true, s2_true = 1.6, 0.7
    measured = (
        s1_true**2 * x1 + s2_true**2 * x2 + (s1_true * s2_true) * x3
    ).numpy()

    canvas = (rng.normal(size=GT_SHAPE) + 1j * rng.normal(size=GT_SHAPE)).astype(
        np.complex64
    )
    test_data = {
        "diffraction": measured.reshape(count, n, n, 1),
        "probeGuess": mini_ladder["identity"]["probe"],
    }

    scaled, s1, s2 = apply_varpro_to_canvas(
        canvas, test_data, None, np.asarray(texture)[:, 0]
    )

    assert s1 == pytest.approx(s1_true, rel=1e-3)
    assert s2 == pytest.approx(s2_true, rel=1e-3)
    np.testing.assert_allclose(scaled.real, s1 * canvas.real, rtol=1e-5)
    np.testing.assert_allclose(scaled.imag, s2 * canvas.imag, rtol=1e-5)


def test_count_consistency_matches_published_formula(
    mini_ladder: dict[str, Any],
) -> None:
    import torch

    from ptycho_torch import reassembly

    from scripts.studies.ablation.runtime_ladder_seams import compute_count_consistency

    rng = np.random.default_rng(9)
    count, n = 4, N_SMALL
    patches = (
        rng.normal(size=(count, n, n)) + 1j * rng.normal(size=(count, n, n))
    ).astype(np.complex64)
    probe = mini_ladder["identity"]["probe"]
    s1, s2 = 1.3, 0.8
    probe_b = torch.as_tensor(probe, dtype=torch.complex64).reshape(
        1, 1, 1, n, n
    ).expand(count, 1, 1, n, n)
    tex = torch.as_tensor(patches[:, np.newaxis], dtype=torch.complex64)
    _, _, x1, x2, x3 = reassembly.compute_varpro_basis(probe_b, tex.real, tex.imag)
    prediction = (s1**2 * x1 + s2**2 * x2 + (s1 * s2) * x3).double().numpy()

    exact = compute_count_consistency(prediction, probe, patches, s1=s1, s2=s2)
    assert exact["relative_l2_intensity_error"] == pytest.approx(0.0, abs=1e-6)
    assert exact["basis"] == "physical_count_space"
    assert exact["n_samples"] == count

    # measured = 2 * prediction => sqrt(sum(pred^2) / sum((2 pred)^2)) = 0.5
    doubled = compute_count_consistency(
        2.0 * prediction, probe, patches, s1=s1, s2=s2
    )
    assert doubled["relative_l2_intensity_error"] == pytest.approx(0.5, rel=1e-5)


def test_normalization_reuse_hashes_real_records() -> None:
    from scripts.studies.ablation.runtime_ladder_seams import (
        resolve_normalization_reuse,
    )
    from scripts.studies.ablation.runtime_execution import RuntimeExecutionError

    training_payload = {
        "training_normalization": {"regime": "legacy", "scaling_constant": 2.5}
    }
    inference_record = {
        "input_scale_factor": 1.0,
        "recomputed_from_heldout": False,
    }

    reuse, train_sha, infer_sha = resolve_normalization_reuse(
        training_payload, inference_record
    )
    assert reuse is True
    assert len(train_sha) == 64 and len(infer_sha) == 64
    # Deterministic canonical hashing.
    again = resolve_normalization_reuse(training_payload, inference_record)
    assert again == (reuse, train_sha, infer_sha)

    recompute = dict(inference_record, recomputed_from_heldout=True)
    assert resolve_normalization_reuse(training_payload, recompute)[0] is False

    with pytest.raises(RuntimeExecutionError, match="normalization"):
        resolve_normalization_reuse({}, inference_record)


def test_gs1_adapter_provides_zero_nominal_and_global_offsets(
    mini_ladder: dict[str, Any],
) -> None:
    """The dictionary path's gs=1 data carries zero relative coords and global
    coords_offsets (the position stitch requires them); the adapter must
    reproduce that contract, not leak global positions into coords_nominal."""
    from scripts.studies.ablation.runtime_ladder_execution import (
        load_generic_test_dict,
    )

    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    recipe = spec.dataset("mini_generic").recipe
    test_npz = mini_ladder["tmp_path"] / "mini_generic" / "test.npz"

    data, metadata = load_generic_test_dict(test_npz, recipe)

    with np.load(test_npz) as src:
        count = src["diff3d"].shape[0]
        np.testing.assert_allclose(
            data["coords_offsets"][:, 0, 0, 0], src["ycoords"]
        )
        np.testing.assert_allclose(
            data["coords_offsets"][:, 0, 1, 0], src["xcoords"]
        )
    np.testing.assert_array_equal(
        data["coords_nominal"], np.zeros((count, 1, 2, 1), dtype=np.float32)
    )
    assert metadata["additional_parameters"]["size"] == recipe.size


# ---------------------------------------------------------------------------
# Review round 1: probe seam must fail CLOSED on mmap rungs (IMPORTANT 1)
# ---------------------------------------------------------------------------


def test_execute_ladder_rung_mmap_requires_recorded_effective_probe(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass-through may be assumed only for the dictionary loader (Task 19
    bridge evidence); an mmap rung whose training path did not record the
    effective probe hash must fail closed even with probe_normalize=false."""
    from scripts.studies.ablation import runtime_ladder_execution
    from scripts.studies.ablation.runtime_execution import RuntimeExecutionError

    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    _ExecHarness(monkeypatch, effective_probe_sha256=None)
    twin = mini_ladder["tmp_path"] / "mini_generic"

    with pytest.raises(RuntimeExecutionError, match="effective.*probe"):
        runtime_ladder_execution.execute_ladder_rung(
            spec,
            spec.rung("mini_rung1_loader"),
            train_npz=twin / "train.npz",
            test_npz=twin / "test.npz",
            work_dir=mini_ladder["tmp_path"] / "exec_work_noprobe",
        )


# ---------------------------------------------------------------------------
# Review round 1: checkbox-2 evidence fields (IMPORTANT 2)
# ---------------------------------------------------------------------------


def test_rung_evidence_records_coverage_and_count_scaling_fields(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(mini_ladder)

    _run_walk(mini_ladder, spec_path, monkeypatch, _ExecutorStub())

    payload, _ = parse_sealed_rung_evidence(
        (
            mini_ladder["tmp_path"]
            / "ladder_out"
            / "mini_rung1_loader"
            / "rung_evidence.json"
        ).read_bytes()
    )
    assert payload["canvas_coverage_fraction"] == pytest.approx(0.97)
    assert payload["count_scaling"] == {
        "mode": "off",
        "physics_scaling_constant": None,
    }
    assert payload["count_consistency"] is None


def test_count_error_evidence_required_on_count_rungs(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rungs = _default_rungs(mini_ladder["identity"])
    rungs[2]["requires_count_error_evidence"] = True
    spec_path = _write_ladder_spec(mini_ladder, rungs=rungs)

    with pytest.raises(StudyRequestError, match="count.consistency"):
        _run_walk(mini_ladder, spec_path, monkeypatch, _ExecutorStub())


def test_count_error_evidence_sealed_when_recorded(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rungs = _default_rungs(mini_ladder["identity"])
    rungs[2]["requires_count_error_evidence"] = True
    spec_path = _write_ladder_spec(mini_ladder, rungs=rungs)
    stub = _ExecutorStub(
        count_consistency={"relative_l2_intensity_error": 0.12}
    )

    outcome = _run_walk(mini_ladder, spec_path, monkeypatch, stub)

    assert outcome.passed
    payload, _ = parse_sealed_rung_evidence(
        (
            mini_ladder["tmp_path"]
            / "ladder_out"
            / "mini_rung3_varpro"
            / "rung_evidence.json"
        ).read_bytes()
    )
    assert payload["count_consistency"]["relative_l2_intensity_error"] == 0.12


def test_auto_count_scale_requires_recorded_constant(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rungs = _default_rungs(mini_ladder["identity"])
    rungs.append(
        {
            "id": "mini_rung4_count_scale",
            "group": "count_scale_bridge",
            "dataset": "mini_generic",
            "changes": {"count_scale_mode": "auto"},
        }
    )
    spec_path = _write_ladder_spec(mini_ladder, rungs=rungs)

    with pytest.raises(StudyRequestError, match="scaling constant"):
        _run_walk(mini_ladder, spec_path, monkeypatch, _ExecutorStub())

    stub = _ExecutorStub(physics_scaling_constant=123.5)
    outcome = _run_walk(
        mini_ladder,
        spec_path,
        monkeypatch,
        stub,
        output_root=mini_ladder["tmp_path"] / "ladder_out_scaled",
    )
    assert outcome.passed
    payload, _ = parse_sealed_rung_evidence(
        (
            mini_ladder["tmp_path"]
            / "ladder_out_scaled"
            / "mini_rung4_count_scale"
            / "rung_evidence.json"
        ).read_bytes()
    )
    assert payload["count_scaling"] == {
        "mode": "auto",
        "physics_scaling_constant": 123.5,
    }


def test_checked_spec_count_rungs_require_count_error_evidence() -> None:
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    flags = {rung.id: rung.requires_count_error_evidence for rung in spec.rungs}
    assert flags["rung6_count_poisson"] is True
    assert flags["rung7_rectangular"] is True
    assert flags["rung8_varpro"] is True
    assert flags["rung1a_mmap_full_scanset"] is False


# ---------------------------------------------------------------------------
# Review round 1: cross-rung dataset-byte binding (IMPORTANT 3)
# ---------------------------------------------------------------------------


def test_dataset_realization_swap_between_rungs_fails_closed(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rung whose recipe step declares the dataset unchanged must consume
    byte-identical NPZs to its predecessor's sealed evidence."""
    spec_path = _write_ladder_spec(mini_ladder)
    root = _datasets_root(mini_ladder)
    stub = _ExecutorStub()
    _run_walk(
        mini_ladder,
        spec_path,
        monkeypatch,
        stub,
        rung="mini_rung1_loader",
        datasets_root=root,
    )

    # Swap the staged realization of the shared dataset between rungs 1 and 2
    # (recipe-valid content: same probe identity, different measurements).
    for split in ("train", "test"):
        path = root / "mini_generic" / f"{split}.npz"
        with np.load(path) as data:
            payload = {key: data[key] for key in data.files}
        payload["diff3d"] = payload["diff3d"] * np.float32(1.5)
        np.savez(path, **payload)

    with pytest.raises(StudyRequestError, match="realization"):
        _run_walk(
            mini_ladder,
            spec_path,
            monkeypatch,
            stub,
            rung="mini_rung2_evaluator",
            datasets_root=root,
        )
    assert stub.executed == ["mini_rung1_loader"]


# ---------------------------------------------------------------------------
# Review round 1: endpoint-gap fields proven inert (ENDPOINT GAP)
# ---------------------------------------------------------------------------


def test_endpoint_gap_fields_are_proven_inert() -> None:
    """Every endpoint-arm field that is neither a ladder config field nor a
    declared residual must be pinned here with a machine-checked proof that
    the reference/runner defaults equal the endpoint declaration."""
    import tomllib

    from scripts.studies.ablation.runtime_ladder_config import (
        ENDPOINT_PROVEN_INERT_FIELDS,
    )

    endpoint_spec = tomllib.loads(
        (
            REPO_ROOT / "scripts/studies/specs/hybrid_resnet_ci_compatibility.toml"
        ).read_text(encoding="utf-8")
    )
    declared = dict(endpoint_spec["base"]["overrides"])
    architecture = next(
        dimension
        for dimension in endpoint_spec["matrix"]["dimensions"]
        if dimension["name"] == "architecture"
    )
    hybrid = next(
        value for value in architecture["values"] if value["id"] == "hybrid_resnet"
    )
    declared.update(hybrid["overrides"])
    for field, pinned in ENDPOINT_PROVEN_INERT_FIELDS.items():
        observed = declared[field]
        if isinstance(observed, list):
            observed = tuple(observed)
        assert observed == pinned, field

    from ptycho_torch.config_params import DataConfig, ModelConfig

    data_defaults = DataConfig()
    model_defaults = ModelConfig()
    assert data_defaults.probe_scale == ENDPOINT_PROVEN_INERT_FIELDS[
        "data.probe_scale"
    ]
    # Systemic lesson from the rung-1 split: bounds value-equality proved
    # constancy across MMAP rungs, not inertness of the rung-0->1 step (the
    # filter is behaviorally active only under the mmap loader). The bounds
    # are therefore a LADDER FIELD (mmap_bounds_filter), not proven-inert.
    assert "data.x_bounds" not in ENDPOINT_PROVEN_INERT_FIELDS
    assert "data.y_bounds" not in ENDPOINT_PROVEN_INERT_FIELDS
    from scripts.studies.ablation.runtime_ladder_mmap import BOUNDS_FILTER_MODES

    assert tuple(declared["data.x_bounds"]) == BOUNDS_FILTER_MODES["endpoint"][0]
    assert tuple(declared["data.y_bounds"]) == BOUNDS_FILTER_MODES["endpoint"][1]
    assert model_defaults.offset == ENDPOINT_PROVEN_INERT_FIELDS["model.offset"]
    # "disabled" maps to None in the study configuration layer; the torch
    # defaults are already None, so the endpoint declaration is a no-op.
    assert model_defaults.amp_loss is None
    assert model_defaults.phase_loss is None
    assert (
        model_defaults.hybrid_encoder_spectral_hidden_scale
        == ENDPOINT_PROVEN_INERT_FIELDS[
            "model.hybrid_encoder_spectral_hidden_scale"
        ]
    )

    from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig

    assert (
        TorchRunnerConfig.__dataclass_fields__[
            "hybrid_encoder_spectral_hidden_scale"
        ].default
        == ENDPOINT_PROVEN_INERT_FIELDS[
            "model.hybrid_encoder_spectral_hidden_scale"
        ]
    )


# ---------------------------------------------------------------------------
# Review round 1: minors (CLI exit codes, --base-dir, accounting schema,
# protocol GateResult operands, probe_normalize/loader coherence)
# ---------------------------------------------------------------------------


def test_cli_per_rung_exit_code_reflects_that_rung(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """--rung invocations exit 0 iff the selected rung PASSes; ladder-wide
    completeness lives in the report (21b automation contract)."""
    spec_path = _write_ladder_spec(mini_ladder)
    stub = _ExecutorStub(
        metrics={
            "mini_rung1_loader": (0.88, 0.95),
            "mini_rung2_evaluator": (0.50, 0.94),
        }
    )
    monkeypatch.setattr(runtime_ladder, "execute_ladder_rung", stub)
    root = _datasets_root(mini_ladder)
    out = mini_ladder["tmp_path"] / "cli_out"
    base_args = [
        "--spec",
        str(spec_path),
        "--datasets-root",
        str(root),
        "--output-root",
        str(out),
        "--base-dir",
        str(mini_ladder["tmp_path"]),
    ]

    assert runtime_ladder.main([*base_args, "--rung", "mini_rung1_loader"]) == 0
    assert runtime_ladder.main([*base_args, "--rung", "mini_rung2_evaluator"]) == 1
    # Full-walk semantics stay ladder-wide.
    assert runtime_ladder.main(base_args) == 1


def test_absolute_gate_cli_returns_quality_failure_exit_one(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(
        mini_ladder, gate_overrides={"policy": "absolute_ssim_delta_v1"}
    )
    monkeypatch.setattr(
        runtime_ladder,
        "execute_ladder_rung",
        _ExecutorStub(metrics={"mini_rung1_loader": (0.87, 0.95)}),
    )
    args = [
        "--spec",
        str(spec_path),
        "--datasets-root",
        str(_datasets_root(mini_ladder)),
        "--output-root",
        str(mini_ladder["tmp_path"] / "absolute_cli_out"),
        "--base-dir",
        str(mini_ladder["tmp_path"]),
        "--rung",
        "mini_rung1_loader",
    ]
    assert runtime_ladder.main(args) == 1
    out = mini_ladder["tmp_path"] / "absolute_cli_out"
    evidence, _ = parse_sealed_rung_evidence(
        (out / "mini_rung1_loader" / "rung_evidence.json").read_bytes()
    )
    assert evidence["gate"]["verdict"] == "fail"
    assert evidence["gate"]["reason"] == "ladder_absolute_amp_ssim_delta_exceeded"
    report = json.loads((out / "ladder_report.json").read_text(encoding="utf-8"))
    assert report["rungs"][0]["gate"]["verdict"] == "fail"
    assert (
        report["rungs"][0]["gate"]["reason"]
        == "ladder_absolute_amp_ssim_delta_exceeded"
    )


def test_scan_accounting_missing_mandated_field_fails_closed(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan mandates the FULL accounting: source scans, duplicate use,
    group count, accepted patches, reconstructed pixels, coverage."""
    rungs = [
        {
            "id": "mini_rung1_loader",
            "group": "loader_schema",
            "dataset": "mini_generic",
            "changes": {"loader": "mmap", "dataset": "mini_generic"},
        },
        {
            "id": "mini_rung2_grouping",
            "group": "grouping_weighting",
            "dataset": "mini_generic",
            "changes": {"gridsize": 2, "training_patch_weighting": "probe"},
            "requires_scan_accounting": True,
        },
    ]
    spec_path = _write_ladder_spec(mini_ladder, rungs=rungs)
    incomplete = {
        "unique_scans_used": 3,
        "unique_scans_expected": 3,
        "scan_utilization_fraction": 1.0,
        "duplicate_scan_uses": 0,
        "accepted_patch_count": 3,
        "reconstructed_pixel_count": 500,
        "canvas_coverage_fraction": 0.99,
        # "group_count" deliberately missing
    }
    stub = _ExecutorStub(scan_accounting=incomplete)

    with pytest.raises(StudyRequestError, match="group_count"):
        _run_walk(mini_ladder, spec_path, monkeypatch, stub)


def test_gate_protocol_failure_carries_no_metric_operands() -> None:
    result = evaluate_rung_gate(
        _gate(),
        "rung_x",
        amp_ssim=0.95,
        phase_ssim=0.99,
        control=_control(),
        failure_reason="ladder_scan_omission",
    )
    assert result.verdict is Verdict.FAIL
    assert result.observed is None
    assert result.threshold is None


def test_spec_rejects_dictionary_loader_with_probe_normalize(
    mini_ladder: dict[str, Any],
) -> None:
    spec = _write_ladder_spec(
        mini_ladder, baseline_config_overrides={"probe_normalize": True}
    )
    with pytest.raises(StudyRequestError, match="probe_normalize"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])


def test_spec_rejects_split_scale_contract_pair(
    mini_ladder: dict[str, Any],
) -> None:
    """The runtime validates (scale_contract_version, measurement_domain) as
    an inseparable pair (scaling_contract.resolve_scale_contract supports only
    legacy_v1+normalized_amplitude and ci_intensity_v2+count_intensity); a
    rung stepping the domain without the version must be rejected at parse."""
    rungs = [
        {
            "id": "mini_rung1_loader",
            "group": "loader_schema",
            "dataset": "mini_generic",
            "changes": {"loader": "mmap", "dataset": "mini_generic"},
        },
        {
            "id": "mini_rung2_domain_only",
            "group": "measurement_domain_loss",
            "dataset": "mini_generic_counts",
            "changes": {
                "dataset": "mini_generic_counts",
                "measurement_domain": "count_intensity",
            },
        },
    ]
    groups = {
        "loader_schema": ["loader", "dataset"],
        "measurement_domain_loss": ["dataset", "measurement_domain"],
    }
    spec = _write_ladder_spec(mini_ladder, rungs=rungs, groups=groups)
    with pytest.raises(StudyRequestError, match="pair"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])


def test_checked_spec_contract_pair_moves_at_the_domain_rung() -> None:
    """Wiring correction: the (version, domain) pair steps together at rung 6
    (declaration), while CI ACTIVATION still happens at rung 7 through the
    physics switch (ci_scaling_active gates on rectangular_scaled)."""
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    rung6 = spec.rung("rung6_count_poisson")
    assert rung6.changes["scale_contract_version"] == "ci_intensity_v2"
    assert rung6.changes["measurement_domain"] == "count_intensity"
    rung7 = spec.rung("rung7_rectangular")
    assert "scale_contract_version" not in rung7.changes
    assert rung7.changes["physics_forward_mode"] == "rectangular_scaled"
    for rung in spec.rungs:
        pair = (
            rung.resolved_config["scale_contract_version"],
            rung.resolved_config["measurement_domain"],
        )
        assert pair in {
            ("legacy_v1", "normalized_amplitude"),
            ("ci_intensity_v2", "count_intensity"),
        }, rung.id


def test_mmap_val_loader_is_real_and_iterable(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rung-1 bring-up blocker regression: the strategy=None loader path
    wraps the test dataset unconditionally, so a None test container yields a
    val loader over a None data_source and Lightning's sanity check dies with
    TypeError(len(None)). The mmap flow must supply a real, iterable val
    dataset."""
    import torch

    from ptycho_torch.workflows.components import (
        _build_dataloaders_from_ptycho_dataset,
    )

    _, config, recipe, _, mmap_mod = _rung1_payload(mini_ladder)
    twin = _write_generic_twin(
        mini_ladder["tmp_path"],
        mini_ladder["identity"],
        name="mini_generic_val",
        train_count=12,
        test_count=12,
    )
    work = mini_ladder["tmp_path"] / "val_loader_work"
    payload = mmap_mod.build_mmap_dataset_payload(config, recipe, twin["train"], work)
    train_ds = mmap_mod.build_generic_loader_dataset(twin["train"], payload, work)
    test_ds = mmap_mod.build_generic_loader_dataset(twin["test"], payload, work)
    # Force the CPU collate path for this test.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    train_loader, val_loader = _build_dataloaders_from_ptycho_dataset(
        train_ptycho_dataset=train_ds,
        payload=payload,
        test_ptycho_dataset=test_ds,
    )

    assert val_loader is not None
    assert len(test_ds) > 0
    batch = next(iter(val_loader))
    assert batch is not None
    train_batch = next(iter(train_loader))
    assert train_batch is not None


def test_inference_normalization_record_is_observed_not_declared(
    mini_ladder: dict[str, Any],
) -> None:
    """Checkbox-6 evidence must be MEASURED from the real inference call: the
    input scale factors actually consumed by forward_predict are observed,
    and only a unit-constant observation attests statistics reuse."""
    import torch

    from scripts.studies import grid_lines_torch_runner as runner_mod
    from scripts.studies.ablation.runtime_ladder_execution import (
        build_inference_normalization_record,
        build_runner_config,
        load_generic_test_dict,
        observe_inference_input_scales,
    )
    from scripts.studies.ablation.runtime_execution import RuntimeExecutionError

    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    rung = spec.rung("mini_rung1_loader")
    config = dict(rung.resolved_config)
    recipe = spec.dataset("mini_generic").recipe
    test_npz = mini_ladder["tmp_path"] / "mini_generic" / "test.npz"
    test_data, metadata = load_generic_test_dict(test_npz, recipe)

    class _Model:
        def eval(self) -> "_Model":
            return self

        def to(self, device: Any) -> "_Model":
            return self

        def parameters(self) -> Any:
            return iter(())

        def forward_predict(
            self, x: Any, positions: Any, probe: Any, input_scale_factor: Any
        ) -> Any:
            return torch.zeros(
                (x.shape[0], x.shape[2], x.shape[3]), dtype=torch.complex64
            )

    model = _Model()
    runner_cfg = build_runner_config(
        config,
        train_npz=test_npz,
        test_npz=test_npz,
        output_dir=mini_ladder["tmp_path"] / "obs_work",
    )

    with observe_inference_input_scales(model) as observed:
        runner_mod.run_torch_inference(model, test_data, runner_cfg, metadata=metadata)

    assert observed, "instrumentation must observe the real inference scales"
    record = build_inference_normalization_record(observed, test_data)
    assert record["input_scale_is_unit_constant"] is True
    assert record["recomputed_from_heldout"] is False
    assert record["observed_input_scale_values"] == [1.0]

    deviated = build_inference_normalization_record([1.0, 2.5], test_data)
    assert deviated["input_scale_is_unit_constant"] is False
    assert deviated["recomputed_from_heldout"] is True

    with pytest.raises(RuntimeExecutionError, match="observ"):
        build_inference_normalization_record([], test_data)


def test_count_scale_operands_fail_closed_without_source(
    mini_ladder: dict[str, Any],
) -> None:
    """No silent unit default: amplitude physics without a recorded auto
    constant has no scaling operand source and must fail closed."""
    from scripts.studies.ablation.runtime_execution import RuntimeExecutionError
    from scripts.studies.ablation.runtime_ladder_seams import (
        resolve_count_scale_operands,
    )

    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    config = dict(spec.rung("mini_rung1_loader").resolved_config)

    with pytest.raises(RuntimeExecutionError, match="operand|scaling"):
        resolve_count_scale_operands(
            config,
            varpro_s1=None,
            varpro_s2=None,
            model=object(),
            training_payload={},
        )
    s1, s2, source = resolve_count_scale_operands(
        config,
        varpro_s1=None,
        varpro_s2=None,
        model=object(),
        training_payload={"physics_scaling_constant": 3.5},
    )
    assert (s1, s2, source) == (3.5, 3.5, "physics_scaling_constant")


# ---------------------------------------------------------------------------
# Rung-1 split: mmap_bounds_filter field, sub-rungs, step-0 diagnostic
# ---------------------------------------------------------------------------


def test_checked_spec_rung1_split_chain() -> None:
    """Adopted decomposition: rung 1a (loader swap at FULL scan-set parity,
    bounds off inherited from the baseline declaration) then rung 1b (the
    bounds filter alone). Every downstream rung carries the endpoint filter."""
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    assert spec.baseline.config["mmap_bounds_filter"] == "off"
    rung_a = spec.rung("rung1a_mmap_full_scanset")
    assert rung_a.group == "loader_schema"
    assert rung_a.resolved_config["mmap_bounds_filter"] == "off"
    assert set(rung_a.changes) == {"loader", "dataset"}
    rung_b = spec.rung("rung1b_bounds_filter")
    assert rung_b.group == "ingestion_bounds"
    assert dict(rung_b.changes) == {"mmap_bounds_filter": "endpoint"}
    assert rung_b.dataset == rung_a.dataset  # byte-bound same twin
    chain = [rung for rung in spec.rungs if not rung.diagnostic]
    for rung in chain[chain.index(rung_b):]:
        assert rung.resolved_config["mmap_bounds_filter"] == "endpoint", rung.id


def test_spec_rejects_dictionary_loader_with_endpoint_bounds(
    mini_ladder: dict[str, Any],
) -> None:
    """The bounds filter is mmap-loader semantics; the dictionary path never
    filters, so its effective declaration must stay 'off'."""
    spec = _write_ladder_spec(
        mini_ladder, baseline_config_overrides={"mmap_bounds_filter": "endpoint"}
    )
    with pytest.raises(StudyRequestError, match="bounds"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])


def test_mmap_payload_applies_bounds_filter_mode(
    mini_ladder: dict[str, Any],
) -> None:
    from scripts.studies.ablation.runtime_ladder_mmap import BOUNDS_FILTER_MODES

    _, config, recipe, train, mmap_mod = _rung1_payload(mini_ladder)
    assert config["mmap_bounds_filter"] == "off"

    payload_off = mmap_mod.build_mmap_dataset_payload(
        config, recipe, train, mini_ladder["tmp_path"] / "bounds_off"
    )
    assert tuple(payload_off.pt_data_config.x_bounds) == (0.0, 1.0)
    assert tuple(payload_off.pt_data_config.y_bounds) == (0.0, 1.0)

    endpoint_config = dict(config, mmap_bounds_filter="endpoint")
    payload_endpoint = mmap_mod.build_mmap_dataset_payload(
        endpoint_config, recipe, train, mini_ladder["tmp_path"] / "bounds_on"
    )
    assert tuple(payload_endpoint.pt_data_config.x_bounds) == (0.1, 0.9)
    assert tuple(payload_endpoint.pt_data_config.y_bounds) == (0.1, 0.9)
    assert BOUNDS_FILTER_MODES["off"] == ((0.0, 1.0), (0.0, 1.0))
    assert BOUNDS_FILTER_MODES["endpoint"] == ((0.1, 0.9), (0.1, 0.9))


def test_bounds_off_keeps_full_scan_set(mini_ladder: dict[str, Any]) -> None:
    """Scan-set parity for rung 1a: with the filter off, the REAL loader
    keeps every position; with the endpoint filter it drops the margin."""
    _, config, recipe, _, mmap_mod = _rung1_payload(mini_ladder)
    twin = _write_generic_twin(
        mini_ladder["tmp_path"],
        mini_ladder["identity"],
        name="mini_generic_bounds",
        train_count=12,
        test_count=12,
    )
    work = mini_ladder["tmp_path"] / "bounds_work"
    payload = mmap_mod.build_mmap_dataset_payload(
        config, recipe, twin["train"], work
    )
    dataset = mmap_mod.build_generic_loader_dataset(twin["train"], payload, work)
    assert list(dataset.valid_indices_per_file[0]) == list(range(12))

    endpoint_config = dict(config, mmap_bounds_filter="endpoint")
    payload_filtered = mmap_mod.build_mmap_dataset_payload(
        endpoint_config, recipe, twin["train"], work / "filtered"
    )
    dataset_filtered = mmap_mod.build_generic_loader_dataset(
        twin["train"], payload_filtered, work / "filtered"
    )
    assert len(dataset_filtered.valid_indices_per_file[0]) < 12


STAGED_DATASETS = REPO_ROOT / ".artifacts/bridge_ladder/datasets"
DICT_N128 = REPO_ROOT / ".artifacts/integration/grid_lines_hybrid_resnet/datasets/N128/gs1"


def test_staged_ingestion_is_bounded_and_cleaned(
    mini_ladder: dict[str, Any], tmp_path: Path
) -> None:
    """RAM/IO hardening: staging must not duplicate bytes (hardlink or copy,
    removed after construction) and ingestion peak RSS must stay within a
    generous multiple of the input size. The dataset stays fully usable
    after cleanup (mmap + in-memory data_dict are self-contained)."""
    import threading
    import time

    psutil = pytest.importorskip("psutil")

    _, config, recipe, _, mmap_mod = _rung1_payload(mini_ladder)
    twin = _write_generic_twin(
        mini_ladder["tmp_path"],
        mini_ladder["identity"],
        name="mini_generic_ram",
        train_count=64,
        test_count=12,
    )
    input_bytes = twin["train"].stat().st_size
    payload = mmap_mod.build_mmap_dataset_payload(
        config, recipe, twin["train"], tmp_path / "payload"
    )

    proc = psutil.Process()
    start = proc.memory_info().rss
    peak = [start]
    running = [True]

    def _sample() -> None:
        while running[0]:
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.005)

    thread = threading.Thread(target=_sample, daemon=True)
    thread.start()
    dataset = mmap_mod.build_generic_loader_dataset(
        twin["train"], payload, tmp_path / "loader"
    )
    running[0] = False
    thread.join(timeout=1.0)

    # No NPZ residue anywhere under the staging subtree (m6: layout-free;
    # the memmap dir legitimately holds a small loader-internal state NPZ).
    assert not list((tmp_path / "loader" / "staged").rglob("*.npz"))
    # Peak RSS delta bounded by a generous multiple of the input size
    # (construction transient: source read + memmap write + torch copies).
    delta = peak[0] - start
    assert delta < max(6 * input_bytes, 256 * 2**20), (
        f"ingestion ballooned: delta={delta/2**20:.0f} MiB for "
        f"{input_bytes/2**20:.0f} MiB input"
    )
    # Dataset remains fully usable after staging cleanup.
    assert len(dataset) > 0
    _ = dataset[0]
    evidence = mmap_mod.extract_loader_evidence(dataset)
    assert (
        evidence["effective_probe_sha256"]
        == mini_ladder["identity"]["transformed_probe_sha256"]
    )


def test_step0_loader_regime_attribution(tmp_path: Path) -> None:
    """Step-0 CPU diagnostic (rung-1 split): ingest the bit-identical generic
    twin through the REAL mmap loader with bounds off and diff the per-sample
    training tensors against the dictionary-path values. A uniform global
    scalar attributes the parent FAIL's V2 (normalization regime); any
    per-sample/order deltas expose V3 (container/pipeline)."""
    train_generic = STAGED_DATASETS / "n128_run1084_generic" / "train.npz"
    train_dict = DICT_N128 / "train.npz"
    if not train_generic.is_file() or not train_dict.is_file():
        pytest.skip("bridge-ladder N128 materializations not staged")
    import torch

    from scripts.studies.ablation import runtime_ladder_mmap as mmap_mod

    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)
    rung = spec.rung("rung1a_mmap_full_scanset")
    config = dict(rung.resolved_config)
    recipe = spec.dataset(rung.dataset).recipe
    payload = mmap_mod.build_mmap_dataset_payload(
        config, recipe, train_generic, tmp_path / "payload"
    )
    dataset = mmap_mod.build_generic_loader_dataset(
        train_generic, payload, tmp_path / "loader"
    )

    # Full scan-set parity (V1 neutralized by construction).
    n_source = 8978
    assert list(dataset.valid_indices_per_file[0]) == list(range(n_source))
    assert len(dataset) == n_source

    mmap = dataset.mmap_ptycho
    nn = np.asarray(mmap["nn_indices"][:].cpu().numpy()).reshape(-1)
    with np.load(train_dict, allow_pickle=True) as archive:
        source = np.asarray(archive["diffraction"])[..., 0]

    probe_indices = np.linspace(0, len(dataset) - 1, 16).astype(int)
    ratios: list[float] = []
    for index in probe_indices:
        source_id = int(nn[index])
        mmap_image = np.asarray(
            mmap["images"][int(index)].cpu().numpy(), dtype=np.float64
        )[0]
        dict_image = np.asarray(source[source_id], dtype=np.float64)
        mask = dict_image > np.max(dict_image) * 1e-3
        assert mask.any()
        pixel_ratios = mmap_image[mask] / dict_image[mask]
        # V3 check: within one sample the transform must be a single scalar
        # (no per-pixel/order/dtype deltas).
        assert float(np.std(pixel_ratios)) < 1e-3 * abs(
            float(np.mean(pixel_ratios))
        ), f"per-pixel deltas at sample {index}: V3 container difference"
        ratios.append(float(np.mean(pixel_ratios)))
    spread = (max(ratios) - min(ratios)) / abs(np.mean(ratios))
    # V2 check: one GLOBAL scalar across samples (Batch rms regime).
    assert spread < 1e-3, f"per-sample scale spread {spread}: V3 suspected"

    scale = float(np.mean(ratios))
    scaling_keys = {
        key: float(np.asarray(mmap[key][0].cpu().numpy()).reshape(-1)[0])
        for key in ("rms_scaling_constant", "physics_scaling_constant")
        if key in set(mmap.keys())
    }
    print(
        f"\nSTEP0 ATTRIBUTION: uniform global image scale = {scale:.6f} "
        f"(V2 normalization regime); loader scaling constants = {scaling_keys}; "
        "no per-sample/order deltas observed (V3 clean at ingestion)"
    )
    assert scale != 1.0 or scaling_keys, "expected a measurable regime difference"
    del dataset, torch


# ---------------------------------------------------------------------------
# Canonical mmap normalization ownership
# ---------------------------------------------------------------------------


def test_checked_spec_canonical_bridge_uses_unit_scaling_until_count_domain() -> None:
    from scripts.studies.ablation.runtime_ladder_mmap import (
        resolve_mmap_normalize_mode,
    )

    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    assert spec.baseline.config["mmap_scale_convention"] == "dictionary_parity"
    for rung_id in (
        "rung1a_mmap_full_scanset",
        "rung1b_bounds_filter",
        "rung2_generic_evaluator",
        "rung3_fly001_probe",
        "rung4_n64",
        "rung5_c4_probe_weighting",
    ):
        assert spec.rung(rung_id).resolved_config[
            "mmap_scale_convention"
        ] == "dictionary_parity", rung_id
    assert resolve_mmap_normalize_mode(
        spec.rung("rung1a_mmap_full_scanset").resolved_config
    ) == "None"
    count_rung = spec.rung("rung6_count_poisson")
    assert count_rung.resolved_config["mmap_scale_convention"] == "loader"
    assert count_rung.resolved_config["measurement_domain"] == "count_intensity"
    assert spec.endpoint_config["mmap_scale_convention"] == "loader"


@pytest.mark.parametrize(
    ("convention", "expected_mode"),
    [("dictionary_parity", "None"), ("loader", "Batch")],
)
def test_mmap_normalize_mode_mapping_is_closed(
    convention: str, expected_mode: str
) -> None:
    from scripts.studies.ablation import runtime_ladder_mmap

    assert runtime_ladder_mmap.resolve_mmap_normalize_mode(
        {"mmap_scale_convention": convention}
    ) == expected_mode


def test_mmap_normalize_mode_mapping_rejects_unknown_convention() -> None:
    from scripts.studies.ablation import runtime_ladder_mmap

    with pytest.raises(RuntimeExecutionError, match="mmap_scale_convention"):
        runtime_ladder_mmap.resolve_mmap_normalize_mode(
            {"mmap_scale_convention": "implicit_default"}
        )


def test_mmap_payload_scale_convention_modes(mini_ladder: dict[str, Any]) -> None:
    """The prebuilt dataset and internal training payload resolve identically."""
    _, config, recipe, train, mmap_mod = _rung1_payload(mini_ladder)
    assert config["mmap_scale_convention"] == "loader"
    runner_cfg = SimpleNamespace(
        training_patch_weighting="central_mask",
        physics_forward_mode="amplitude",
        cnn_output_mode="amp_phase",
        rect_s1s2_trainable=False,
        scale_contract_version="legacy_v1",
        measurement_domain="normalized_amplitude",
        amplitude_physics_gain=16.0,
    )

    payload_loader = mmap_mod.build_mmap_dataset_payload(
        config, recipe, train, mini_ladder["tmp_path"] / "conv_loader"
    )
    assert payload_loader.pt_data_config.normalize == "Batch"
    assert mmap_mod.runner_torch_overrides(runner_cfg, config)["normalize"] == "Batch"

    parity_config = dict(config, mmap_scale_convention="dictionary_parity")
    payload_parity = mmap_mod.build_mmap_dataset_payload(
        parity_config, recipe, train, mini_ladder["tmp_path"] / "conv_parity"
    )
    assert payload_parity.pt_data_config.normalize == "None"
    assert mmap_mod.runner_torch_overrides(runner_cfg, parity_config)["normalize"] == "None"


def test_data_config_normalize_type_includes_supported_none_mode() -> None:
    from ptycho_torch.config_params import DataConfig

    assert DataConfig().normalize == "Batch"
    assert set(get_args(get_type_hints(DataConfig)["normalize"])) == {
        "Group",
        "Batch",
        "None",
    }


def test_parity_convention_yields_unit_scalars_in_real_loader(
    mini_ladder: dict[str, Any],
) -> None:
    _, config, recipe, _, mmap_mod = _rung1_payload(mini_ladder)
    twin = _write_generic_twin(
        mini_ladder["tmp_path"],
        mini_ladder["identity"],
        name="mini_generic_parity",
        train_count=12,
        test_count=12,
    )
    work = mini_ladder["tmp_path"] / "parity_work"

    loader_payload = mmap_mod.build_mmap_dataset_payload(
        config, recipe, twin["train"], work / "loader_payload"
    )
    loader_ds = mmap_mod.build_generic_loader_dataset(
        twin["train"], loader_payload, work / "loader_mode"
    )
    loader_evidence = mmap_mod.extract_loader_evidence(loader_ds)
    assert loader_evidence["training_normalization"]["scaling_constant"] != 1.0

    parity_config = dict(config, mmap_scale_convention="dictionary_parity")
    parity_payload = mmap_mod.build_mmap_dataset_payload(
        parity_config, recipe, twin["train"], work / "parity_payload"
    )
    parity_ds = mmap_mod.build_generic_loader_dataset(
        twin["train"], parity_payload, work / "parity_mode"
    )
    parity_evidence = mmap_mod.extract_loader_evidence(parity_ds)
    assert parity_evidence["training_normalization"]["scaling_constant"] == 1.0
    # I5: pin the ACTUAL per-sample constants training consumes, not just the
    # legacy proxy — an upstream branch split cannot silently break the lever.
    parity_mmap = parity_ds.mmap_ptycho
    for key in ("rms_scaling_constant", "physics_scaling_constant"):
        values = np.asarray(parity_mmap[key][:].cpu().numpy(), dtype=np.float64)
        np.testing.assert_array_equal(values, np.ones_like(values), err_msg=key)
    loader_mmap = loader_ds.mmap_ptycho
    assert not np.allclose(
        np.asarray(loader_mmap["rms_scaling_constant"][:].cpu().numpy()), 1.0
    )
    assert not np.allclose(
        np.asarray(loader_mmap["physics_scaling_constant"][:].cpu().numpy()), 1.0
    )
    # Measurement values stay bit-identical either way (step-0 property).
    assert (
        parity_evidence["effective_probe_sha256"]
        == loader_evidence["effective_probe_sha256"]
    )


def _diagnostic_rungs(identity: dict[str, Any]) -> list[dict[str, Any]]:
    rungs = _default_rungs(identity)
    rungs.insert(
        1,
        {
            "id": "mini_rung_diag",
            "group": "ingestion_normalization",
            "dataset": "mini_generic",
            "changes": {"mmap_scale_convention": "dictionary_parity"},
            "diagnostic": True,
        },
    )
    return rungs


def _diagnostic_groups() -> dict[str, list[str]]:
    return {
        "loader_schema": ["loader", "dataset"],
        "ingestion_normalization": ["mmap_scale_convention"],
        "reassembly_alignment": ["gated_evaluator"],
        "inference_varpro": ["varpro_scaling"],
    }


def test_diagnostic_rung_walk_semantics(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnostic rung gates against the BASELINE control, never becomes a
    control, never sets first_material_degradation, and is excluded from the
    ladder's promotion requirement."""
    spec_path = _write_ladder_spec(
        mini_ladder, rungs=_diagnostic_rungs(mini_ladder["identity"]),
        groups=_diagnostic_groups(),
    )
    stub = _ExecutorStub(
        metrics={
            "mini_rung1_loader": (0.88, 0.95),
            "mini_rung_diag": (0.50, 0.60),  # diagnostic FAIL
            "mini_rung2_evaluator": (0.87, 0.94),
            "mini_rung3_varpro": (0.86, 0.94),
        }
    )

    outcome = _run_walk(mini_ladder, spec_path, monkeypatch, stub)

    assert outcome.passed  # all NON-diagnostic rungs pass
    assert outcome.first_material_degradation is None
    payload, _ = parse_sealed_rung_evidence(
        (
            mini_ladder["tmp_path"] / "ladder_out" / "mini_rung_diag" /
            "rung_evidence.json"
        ).read_bytes()
    )
    # Diagnostic gates against rung 0, regardless of the passing rung 1.
    assert payload["control"]["rung_id"] == "rung0_reference"
    # Successor of the diagnostic still chains its control from rung 1.
    rung2, _ = parse_sealed_rung_evidence(
        (
            mini_ladder["tmp_path"] / "ladder_out" / "mini_rung2_evaluator" /
            "rung_evidence.json"
        ).read_bytes()
    )
    assert rung2["control"]["rung_id"] == "mini_rung1_loader"
    # And its resolved config does NOT inherit the diagnostic step.
    assert rung2["resolved_config"]["mmap_scale_convention"] == "loader"


def test_diagnostic_rung_runs_standalone_without_prior_evidence(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(
        mini_ladder, rungs=_diagnostic_rungs(mini_ladder["identity"]),
        groups=_diagnostic_groups(),
    )
    stub = _ExecutorStub(metrics={"mini_rung_diag": (0.88, 0.95)})

    outcome = _run_walk(
        mini_ladder,
        spec_path,
        monkeypatch,
        stub,
        rung="mini_rung_diag",
        output_root=mini_ladder["tmp_path"] / "diag_only_out",
    )

    assert stub.executed == ["mini_rung_diag"]
    (result,) = outcome.results
    assert result.verdict is Verdict.PASS
    payload, _ = parse_sealed_rung_evidence(
        (
            mini_ladder["tmp_path"] / "diag_only_out" / "mini_rung_diag" /
            "rung_evidence.json"
        ).read_bytes()
    )
    assert payload["control"]["rung_id"] == "rung0_reference"
    assert payload["resolved_config"]["mmap_scale_convention"] == "dictionary_parity"


def _historical_diagnostic_rungs(identity: dict[str, Any]) -> list[dict[str, Any]]:
    rungs = _diagnostic_rungs(identity)
    rungs[1]["execution_status"] = "historical_only"
    return rungs


def test_checked_spec_archives_obsolete_diagnostic_rungs_in_prose_only() -> None:
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)
    ids = {rung.id for rung in spec.rungs}

    assert not ids.intersection(
        {
            "rung1c_normalization_regime",
            "rung1d_sampler_shuffle",
            "rung1e_sampler_plus_unit_norm",
            "rung1f_probe_layout",
        }
    )


@pytest.mark.parametrize("execution_status", ["archived", "", True])
def test_rung_execution_status_is_closed_and_typed(
    mini_ladder: dict[str, Any], execution_status: Any
) -> None:
    rungs = _diagnostic_rungs(mini_ladder["identity"])
    rungs[1]["execution_status"] = execution_status
    spec_path = _write_ladder_spec(
        mini_ladder, rungs=rungs, groups=_diagnostic_groups()
    )

    with pytest.raises(StudyRequestError, match="execution_status"):
        load_ladder_spec(spec_path, base_dir=mini_ladder["tmp_path"])


def test_historical_only_status_is_diagnostic_only(
    mini_ladder: dict[str, Any],
) -> None:
    rungs = _default_rungs(mini_ladder["identity"])
    rungs[0]["execution_status"] = "historical_only"
    spec_path = _write_ladder_spec(mini_ladder, rungs=rungs)

    with pytest.raises(StudyRequestError, match="historical_only|diagnostic"):
        load_ladder_spec(spec_path, base_dir=mini_ladder["tmp_path"])


def test_historical_only_rung_lists_but_cannot_be_selected_or_run(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(
        mini_ladder,
        rungs=_historical_diagnostic_rungs(mini_ladder["identity"]),
        groups=_diagnostic_groups(),
    )
    request_args = {
        "spec": spec_path,
        "base_dir": mini_ladder["tmp_path"],
        "rung": "mini_rung_diag",
    }

    with pytest.raises(StudyRequestError, match="historical_only|not runnable"):
        run_bridge_ladder(LadderRequest(**request_args, dry_run=True))
    with pytest.raises(StudyRequestError, match="historical_only|not runnable"):
        run_bridge_ladder(
            LadderRequest(
                **request_args,
                datasets_root=_datasets_root(mini_ladder),
                output_root=mini_ladder["tmp_path"] / "historical_selected",
            )
        )

    stub = _ExecutorStub()
    outcome = _run_walk(mini_ladder, spec_path, monkeypatch, stub)

    assert outcome.passed
    assert "mini_rung_diag" not in stub.executed
    assert {result.id for result in outcome.results} == {
        "mini_rung1_loader",
        "mini_rung2_evaluator",
        "mini_rung3_varpro",
    }


# ---------------------------------------------------------------------------
# Fix round (task-21c review): I1 diagnostic byte-binding, I2 scoped field
# migration, I3b chain-report protection
# ---------------------------------------------------------------------------


def _synthesize_prefield_evidence(
    source_root: Path,
    target_root: Path,
    rung_id: str,
    drop_field: str,
    *,
    control_sha: str | None = None,
) -> str:
    """Re-seal a rung's evidence with one resolved-config field absent,
    emulating evidence sealed before that field existed. Returns the new
    seal hash; ``control_sha`` rewrites the control linkage so a synthesized
    CHAIN stays internally consistent (like real pre-field evidence)."""
    payload, _ = parse_sealed_rung_evidence(
        (source_root / rung_id / "rung_evidence.json").read_bytes()
    )
    del payload["resolved_config"][drop_field]
    if control_sha is not None:
        payload["control"]["evidence_sha256"] = control_sha
    target = target_root / rung_id / "rung_evidence.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    return seal_rung_evidence(payload, target)


def test_migrated_field_permits_prefield_evidence_reuse(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """I2 (controller decision): evidence sealed before a field existed is
    treated as carrying that field's assumed default ONLY for fields in the
    MIGRATED_CONFIG_FIELDS whitelist (the default branch is bit-preserving);
    everything else stays strict-equality fail-closed."""
    spec_path = _write_ladder_spec(mini_ladder)
    root_a = mini_ladder["tmp_path"] / "prefield_src"
    _run_walk(
        mini_ladder, spec_path, monkeypatch, _ExecutorStub(), output_root=root_a
    )

    # (a) pre-field evidence + default-valued resolved config -> reuse.
    root_b = mini_ladder["tmp_path"] / "prefield_reuse"
    chain_sha: str | None = None
    for rung_id in ("mini_rung1_loader", "mini_rung2_evaluator", "mini_rung3_varpro"):
        chain_sha = _synthesize_prefield_evidence(
            root_a, root_b, rung_id, "mmap_scale_convention", control_sha=chain_sha
        )
    stub = _ExecutorStub()
    outcome = _run_walk(
        mini_ladder, spec_path, monkeypatch, stub, output_root=root_b
    )
    assert stub.executed == []  # all reused despite the absent field
    assert outcome.passed

    # (c) an UNLISTED absent field still refuses.
    root_c = mini_ladder["tmp_path"] / "prefield_refuse"
    _synthesize_prefield_evidence(root_a, root_c, "mini_rung1_loader", "gated_evaluator")
    with pytest.raises(StudyRequestError, match="different resolved config"):
        _run_walk(
            mini_ladder, spec_path, monkeypatch, _ExecutorStub(), output_root=root_c
        )


def test_migrated_field_refuses_nondefault_value(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) pre-field evidence may NOT stand in for a non-default value."""
    rungs = _diagnostic_rungs(mini_ladder["identity"])
    spec_path = _write_ladder_spec(
        mini_ladder, rungs=rungs, groups=_diagnostic_groups()
    )
    root = mini_ladder["tmp_path"] / "diag_nondefault"
    stub = _ExecutorStub()
    _run_walk(
        mini_ladder, spec_path, monkeypatch, stub,
        rung="mini_rung_diag", output_root=root,
    )
    # Re-seal the diagnostic's evidence WITHOUT the field: its resolved
    # config carries the NON-default value, so migration must not apply.
    payload, _ = parse_sealed_rung_evidence(
        (root / "mini_rung_diag" / "rung_evidence.json").read_bytes()
    )
    assert payload["resolved_config"]["mmap_scale_convention"] == "dictionary_parity"
    root2 = mini_ladder["tmp_path"] / "diag_nondefault_refuse"
    _synthesize_prefield_evidence(root, root2, "mini_rung_diag", "mmap_scale_convention")
    with pytest.raises(StudyRequestError, match="different resolved config"):
        _run_walk(
            mini_ladder, spec_path, monkeypatch, _ExecutorStub(),
            rung="mini_rung_diag", output_root=root2,
        )


def test_diagnostic_rung_binds_dataset_bytes_to_chain_evidence(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """I1: a diagnostic rung whose dataset id matches a chain rung with
    sealed evidence in the output root must byte-bind its consumed pair
    against that evidence (hash comparison only) and refuse a swapped
    realization fail-closed."""
    rungs = _diagnostic_rungs(mini_ladder["identity"])
    spec_path = _write_ladder_spec(
        mini_ladder, rungs=rungs, groups=_diagnostic_groups()
    )
    root = mini_ladder["tmp_path"] / "diag_bind"
    datasets_root = _datasets_root(mini_ladder)
    stub = _ExecutorStub()
    _run_walk(
        mini_ladder, spec_path, monkeypatch, stub,
        rung="mini_rung1_loader", output_root=root, datasets_root=datasets_root,
    )

    # Swap the staged realization of the shared dataset.
    for split in ("train", "test"):
        path = datasets_root / "mini_generic" / f"{split}.npz"
        with np.load(path) as data:
            payload = {key: data[key] for key in data.files}
        payload["diff3d"] = payload["diff3d"] * np.float32(1.5)
        np.savez(path, **payload)

    with pytest.raises(StudyRequestError, match="realization"):
        _run_walk(
            mini_ladder, spec_path, monkeypatch, stub,
            rung="mini_rung_diag", output_root=root, datasets_root=datasets_root,
        )
    assert stub.executed == ["mini_rung1_loader"]


def test_diagnostic_selection_never_touches_chain_report(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """I3b: diagnostic verdicts go to ladder_diagnostics_report.json; the
    chain report bytes stay untouched by a diagnostic invocation."""
    rungs = _diagnostic_rungs(mini_ladder["identity"])
    spec_path = _write_ladder_spec(
        mini_ladder, rungs=rungs, groups=_diagnostic_groups()
    )
    root = mini_ladder["tmp_path"] / "diag_report_root"
    # Build the chain report first (full walk adjudicates everything).
    outcome_full = _run_walk(
        mini_ladder, spec_path, monkeypatch, _ExecutorStub(), output_root=root
    )
    assert outcome_full.passed
    chain_report = root / "ladder_report.json"
    before = chain_report.read_bytes()
    report = json.loads(before)
    assert all(not entry.get("diagnostic", False) for entry in report["rungs"])

    outcome_diag = _run_walk(
        mini_ladder, spec_path, monkeypatch, _ExecutorStub(),
        rung="mini_rung_diag", output_root=root,
    )

    assert chain_report.read_bytes() == before  # byte-unchanged
    diagnostics = json.loads((root / "ladder_diagnostics_report.json").read_text())
    assert [entry["id"] for entry in diagnostics["rungs"]] == ["mini_rung_diag"]
    assert outcome_diag.report_path == root / "ladder_diagnostics_report.json"


def test_chain_selection_never_demotes_diagnostics_report(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1 (re-review minor): a CHAIN-selected invocation must not rewrite the
    diagnostics report with skipped entries."""
    rungs = _diagnostic_rungs(mini_ladder["identity"])
    spec_path = _write_ladder_spec(
        mini_ladder, rungs=rungs, groups=_diagnostic_groups()
    )
    root = mini_ladder["tmp_path"] / "r1_root"
    _run_walk(mini_ladder, spec_path, monkeypatch, _ExecutorStub(), output_root=root)
    diagnostics = root / "ladder_diagnostics_report.json"
    before = diagnostics.read_bytes()

    _run_walk(
        mini_ladder, spec_path, monkeypatch, _ExecutorStub(),
        rung="mini_rung1_loader", output_root=root,
    )

    assert diagnostics.read_bytes() == before


def test_staging_residue_cleared_even_when_loader_raises(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """m3: the staged link must not leak if PtychoDataset construction fails."""
    from ptycho_torch import dataloader

    _, config, recipe, train, mmap_mod = _rung1_payload(mini_ladder)
    payload = mmap_mod.build_mmap_dataset_payload(
        config, recipe, train, tmp_path / "payload"
    )

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("loader construction failed (injected)")

    monkeypatch.setattr(dataloader, "PtychoDataset", boom)
    with pytest.raises(RuntimeError, match="injected"):
        mmap_mod.build_generic_loader_dataset(train, payload, tmp_path / "boomwork")
    assert not list((tmp_path / "boomwork" / "staged").rglob("*.npz"))


# ---------------------------------------------------------------------------
# V3b batch-consumption diagnostic (dictionary vs mmap training consumption)
# ---------------------------------------------------------------------------


def test_v3b_identity_index_maps_and_rejects_collisions() -> None:
    from scripts.studies.ablation.runtime_ladder_diagnostics import (
        build_image_identity_index,
    )
    from scripts.studies.ablation.runtime_execution import RuntimeExecutionError

    rng = np.random.default_rng(1)
    images = rng.random((5, 8, 8)).astype(np.float32)
    index = build_image_identity_index(images)
    assert len(index) == 5
    from scripts.studies.ablation.dataset_provenance import canonical_array_sha256

    assert index[canonical_array_sha256(images[3])] == 3

    duplicated = np.stack([images[0], images[0]])
    with pytest.raises(RuntimeExecutionError, match="collision|duplicate"):
        build_image_identity_index(duplicated)


def test_v3b_batch_sample_records_normalizes_both_batch_shapes() -> None:
    import torch

    from scripts.studies.ablation.runtime_ladder_diagnostics import (
        batch_sample_records,
    )

    rng = np.random.default_rng(2)
    images = torch.as_tensor(rng.random((3, 1, 8, 8)).astype(np.float32))
    # Dictionary-path shape: (tensor_dict, probe, scaling)
    dict_batch = (
        {"images": images, "coords_relative": torch.zeros((3, 1, 1, 2))},
        torch.ones((8, 8), dtype=torch.complex64),
        torch.ones((3, 1, 1, 1)),
    )
    records = batch_sample_records(dict_batch)
    assert len(records) == 3
    assert records[0]["image"].shape == (8, 8)
    assert records[0]["center_scan_id"] is None
    np.testing.assert_array_equal(
        records[1]["image"], images[1, 0].numpy()
    )

    # mmap-path shape: TensorDict-like mapping with identities
    mmap_batch = (
        {
            "images": images,
            "coords_relative": torch.zeros((3, 1, 1, 2)),
            "center_scan_id": torch.as_tensor([7, 8, 9]),
        },
        torch.ones((3, 1, 1, 8, 8), dtype=torch.complex64),
        torch.ones((3, 1, 1, 1)),
    )
    records = batch_sample_records(mmap_batch)
    assert [record["center_scan_id"] for record in records] == [7, 8, 9]
    assert records[2]["image"].shape == (8, 8)


def test_v3b_compare_consumption_detects_order_multiset_and_fields() -> None:
    from scripts.studies.ablation.runtime_ladder_diagnostics import (
        EpochRecord,
        compare_consumption,
    )

    image = np.ones((4, 4), dtype=np.float32)
    deep_a = {0: {"image": image, "coords_relative": np.zeros(2)}}
    deep_b_same = {0: {"image": image, "coords_relative": np.zeros(2)}}
    epoch_a = EpochRecord(
        steps=2, batch_sizes=[2, 1], ordered_ids=[0, 1, 2], unmatched=0,
        deep_samples=deep_a,
    )
    epoch_b_reordered = EpochRecord(
        steps=2, batch_sizes=[2, 1], ordered_ids=[2, 0, 1], unmatched=0,
        deep_samples=deep_b_same,
    )
    summary = compare_consumption(
        [epoch_a], [epoch_b_reordered], labels=("dictionary", "mmap")
    )
    epoch_summary = summary["epochs"][0]
    assert epoch_summary["multiset_equal"] is True
    assert epoch_summary["order_equal"] is False
    assert epoch_summary["first_order_divergence"]["position"] == 0
    assert epoch_summary["step_counts"] == [2, 2]
    assert summary["first_field_divergence"] is None

    # Multiset difference
    epoch_b_missing = EpochRecord(
        steps=2, batch_sizes=[2, 1], ordered_ids=[0, 1, 1], unmatched=0,
        deep_samples=deep_b_same,
    )
    summary2 = compare_consumption([epoch_a], [epoch_b_missing], labels=("a", "b"))
    assert summary2["epochs"][0]["multiset_equal"] is False

    # Field byte divergence, aligned by sample id
    deep_b_diverged = {
        0: {"image": image, "coords_relative": np.ones(2, dtype=np.float64)}
    }
    epoch_b_field = EpochRecord(
        steps=2, batch_sizes=[2, 1], ordered_ids=[0, 1, 2], unmatched=0,
        deep_samples=deep_b_diverged,
    )
    summary3 = compare_consumption([epoch_a], [epoch_b_field], labels=("a", "b"))
    divergence = summary3["first_field_divergence"]
    assert divergence is not None
    assert divergence["field"] == "coords_relative"
    assert divergence["sample_id"] == 0
    assert divergence["epoch"] == 0


def test_v3b_field_attestation_covers_every_field_past_first_divergence() -> None:
    """Review V1: a presence-only divergence (coords_center) must NOT
    short-circuit the byte comparison of the remaining shared fields — the
    artifact has to attest EVERY consumed tensor field, with per-field
    divergence counts and an explicit attested-equal list."""
    from scripts.studies.ablation.runtime_ladder_diagnostics import (
        EpochRecord,
        compare_consumption,
    )

    image = np.ones((4, 4), dtype=np.float32)
    coords = np.zeros(2)
    deep_a = {
        0: {"image": image, "coords_relative": coords, "coords_center": None,
            "rms_scaling_constant": np.float32(1.0)},
        1: {"image": image, "coords_relative": coords, "coords_center": None,
            "rms_scaling_constant": np.float32(1.0)},
    }
    deep_b = {
        0: {"image": image, "coords_relative": coords,
            "coords_center": np.full((1, 2), 63.5, dtype=np.float32),
            "rms_scaling_constant": np.float32(1.33)},
        1: {"image": image, "coords_relative": coords,
            "coords_center": np.full((1, 2), 63.5, dtype=np.float32),
            "rms_scaling_constant": np.float32(1.33)},
    }
    epochs_a = [
        EpochRecord(steps=1, batch_sizes=[2], ordered_ids=[0, 1], unmatched=0,
                    deep_samples=deep_a),
        EpochRecord(steps=1, batch_sizes=[2], ordered_ids=[1, 0], unmatched=0,
                    deep_samples=deep_a),
    ]
    epochs_b = [
        EpochRecord(steps=1, batch_sizes=[2], ordered_ids=[0, 1], unmatched=0,
                    deep_samples=deep_b),
        EpochRecord(steps=1, batch_sizes=[2], ordered_ids=[0, 1], unmatched=0,
                    deep_samples=deep_b),
    ]
    summary = compare_consumption(epochs_a, epochs_b, labels=("a", "b"))

    divergences = summary["field_divergences"]
    assert sorted(entry["field"] for entry in divergences) == [
        "coords_center",
        "rms_scaling_constant",
    ]
    by_field = {entry["field"]: entry for entry in divergences}
    # Both epochs x both samples compared and divergent for each field.
    assert by_field["coords_center"]["divergent_samples"] == 4
    assert by_field["rms_scaling_constant"]["divergent_samples"] == 4
    assert by_field["coords_center"]["compared_samples"] == 4
    assert by_field["coords_center"]["first"]["sample_id"] == 0
    assert by_field["coords_center"]["first"]["epoch"] == 0
    # Fields byte-equal on EVERY compared sample are attested by name.
    assert summary["fields_attested_equal"] == ["coords_relative", "image"]
    # Compatibility: first_field_divergence still points at the first entry.
    assert summary["first_field_divergence"]["field"] == "coords_center"
    # V2: measured order facts, not class semantics.
    assert summary["epochs"][0]["identity_raster_order"] == [True, True]
    assert summary["epochs"][1]["identity_raster_order"] == [False, True]
    assert summary["epoch_order_stable"] == [False, True]


def test_v3b_capture_and_diff_on_miniature_twins(
    mini_ladder: dict[str, Any], tmp_path: Path
) -> None:
    """Integration (CPU, real construction paths): capture the loaders the
    dictionary flow (run_torch_training) and the mmap flow
    (train_via_generic_loader) would hand to Lightning, iterate them, and
    diff consumption on a converted (bit-identical) twin pair."""
    from scripts.studies.make_generic_schema_twin import convert_pair

    from scripts.studies.ablation import runtime_ladder_diagnostics as diag

    identity = mini_ladder["identity"]
    pair = _write_dictionary_pair(mini_ladder["tmp_path"], identity, name="v3b_dict")
    twin_dir = mini_ladder["tmp_path"] / "v3b_twin"
    convert_pair(pair["train"], pair["test"], twin_dir)

    spec = load_ladder_spec(
        _write_ladder_spec(mini_ladder), base_dir=mini_ladder["tmp_path"]
    )
    dict_config = dict(spec.baseline.config)
    mmap_config = dict(spec.rung("mini_rung1_loader").resolved_config)
    # Miniature N=16 cannot host fno_modes=12; shrink the (invariant) model
    # knobs identically for both paths — parity is what matters here.
    for config in (dict_config, mmap_config):
        config.update(fno_modes=2, fno_width=8, fno_blocks=4, fno_cnn_blocks=2)
    recipe = spec.dataset("mini_generic").recipe

    captured_dict = diag.capture_dictionary_training_loaders(
        dict_config, pair["train"], pair["test"], tmp_path / "dict_work"
    )
    captured_mmap = diag.capture_mmap_training_loaders(
        mmap_config, recipe, twin_dir / "train.npz", twin_dir / "test.npz",
        tmp_path / "mmap_work",
    )
    assert captured_dict.train_loader is not None
    assert captured_mmap.val_loader is not None

    summary = diag.run_batch_consumption_diff(
        captured_dict,
        captured_mmap,
        train_source=pair["train"],
        test_source=pair["test"],
        epochs=2,
        output_json=tmp_path / "diff.json",
    )

    assert (tmp_path / "diff.json").is_file()
    train_summary = summary["train"]
    assert train_summary["epochs"][0]["step_counts"][0] == train_summary[
        "epochs"
    ][0]["step_counts"][1]
    # The twins are bit-identical, so every consumed image must map back to a
    # source scan on BOTH paths.
    for epoch in train_summary["epochs"]:
        assert epoch["unmatched"] == [0, 0]
        assert epoch["multiset_equal"] is True
    assert summary["val"]["epochs"][0]["multiset_equal"] is True
    assert summary["samplers"]["dictionary"]["train"]
    assert summary["samplers"]["mmap"]["train"]
    # Any first field divergence must name a field (or be None).
    divergence = summary["first_field_divergence"]
    assert divergence is None or "field" in divergence


# ---------------------------------------------------------------------------
# Sampler-isolation diagnostic rungs (1d/1e)
# ---------------------------------------------------------------------------


def test_checked_spec_keeps_sampler_sequential_after_diagnostics_exonerate_it() -> None:
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    for rung in spec.rungs:
        assert rung.resolved_config["mmap_train_sampler"] == "sequential", rung.id


def test_control_rung_only_on_diagnostics_and_must_be_prior(
    mini_ladder: dict[str, Any],
) -> None:
    rungs = _default_rungs(mini_ladder["identity"])
    rungs[1]["control_rung"] = "mini_rung1_loader"  # non-diagnostic rung
    spec = _write_ladder_spec(mini_ladder, rungs=rungs)
    with pytest.raises(StudyRequestError, match="control_rung|diagnostic"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])

    rungs = _diagnostic_rungs(mini_ladder["identity"])
    rungs[1]["control_rung"] = "mini_rung3_varpro"  # later rung
    spec = _write_ladder_spec(mini_ladder, rungs=rungs, groups=_diagnostic_groups())
    with pytest.raises(StudyRequestError, match="control_rung|prior|earlier"):
        load_ladder_spec(spec, base_dir=mini_ladder["tmp_path"])


def _sampler_groups() -> dict[str, list[str]]:
    groups = _diagnostic_groups()
    groups["ingestion_sampler"] = ["mmap_train_sampler"]
    return groups


def _sampler_rungs(identity: dict[str, Any]) -> list[dict[str, Any]]:
    rungs = _diagnostic_rungs(identity)
    rungs.insert(
        2,
        {
            "id": "mini_rung_shuffle",
            "group": "ingestion_sampler",
            "dataset": "mini_generic",
            "changes": {"mmap_train_sampler": "shuffled"},
            "diagnostic": True,
            "control_rung": "mini_rung1_loader",
        },
    )
    rungs.insert(
        3,
        {
            "id": "mini_rung_shuffle_parity",
            "group": "ingestion_sampler",
            "dataset": "mini_generic",
            "changes": {"mmap_train_sampler": "shuffled"},
            "diagnostic": True,
            "control_rung": "mini_rung_diag",
        },
    )
    return rungs


def test_control_rung_resolution_base_makes_single_group_delta(
    mini_ladder: dict[str, Any],
) -> None:
    """A diagnostic's resolution base is its control rung — 1e-style rungs
    are single-group against a DIAGNOSTIC control (two groups vs the chain)."""
    spec = load_ladder_spec(
        _write_ladder_spec(
            mini_ladder,
            rungs=_sampler_rungs(mini_ladder["identity"]),
            groups=_sampler_groups(),
        ),
        base_dir=mini_ladder["tmp_path"],
    )
    parity = spec.rung("mini_rung_shuffle_parity")
    assert parity.resolved_config["mmap_scale_convention"] == "dictionary_parity"
    assert parity.resolved_config["mmap_train_sampler"] == "shuffled"
    from scripts.studies.ablation.runtime_ladder_spec import config_delta

    base = spec.rung("mini_rung_diag").resolved_config
    assert set(config_delta(base, parity.resolved_config)) == {"mmap_train_sampler"}
    # Chain successor still inherits from the chain, not the diagnostics.
    rung2 = spec.rung("mini_rung2_evaluator")
    assert rung2.resolved_config["mmap_train_sampler"] == "sequential"
    assert rung2.resolved_config["mmap_scale_convention"] == "loader"


def test_diagnostic_control_rung_gates_against_its_sealed_evidence(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(
        mini_ladder,
        rungs=_sampler_rungs(mini_ladder["identity"]),
        groups=_sampler_groups(),
    )
    root = mini_ladder["tmp_path"] / "ctrl_root"
    stub = _ExecutorStub(
        metrics={
            "mini_rung1_loader": (0.60, 0.70),
            "mini_rung_shuffle": (0.57, 0.68),
        }
    )
    # Control evidence missing -> fail-closed.
    with pytest.raises(StudyRequestError, match="control"):
        _run_walk(
            mini_ladder, spec_path, monkeypatch, stub,
            rung="mini_rung_shuffle", output_root=root,
        )
    # Seal the control rung (a FAILED chain rung is a valid diagnostic control).
    _run_walk(
        mini_ladder, spec_path, monkeypatch, stub,
        rung="mini_rung1_loader", output_root=root,
    )
    outcome = _run_walk(
        mini_ladder, spec_path, monkeypatch, stub,
        rung="mini_rung_shuffle", output_root=root,
    )
    (result,) = outcome.results
    assert result.verdict is Verdict.PASS  # 0.57/0.60 = 0.95 retained
    payload, _ = parse_sealed_rung_evidence(
        (root / "mini_rung_shuffle" / "rung_evidence.json").read_bytes()
    )
    assert payload["control"]["rung_id"] == "mini_rung1_loader"
    control_evidence = (root / "mini_rung1_loader" / "rung_evidence.json").read_bytes()
    assert payload["control"]["evidence_sha256"] == hashlib.sha256(
        control_evidence
    ).hexdigest()
    assert payload["control"]["amp_ssim"] == 0.60


def test_absolute_diagnostic_control_rung_gates_against_its_sealed_evidence(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_path = _write_ladder_spec(
        mini_ladder,
        rungs=_sampler_rungs(mini_ladder["identity"]),
        groups=_sampler_groups(),
        gate_overrides={"policy": "absolute_ssim_delta_v1"},
    )
    root = mini_ladder["tmp_path"] / "absolute_ctrl_root"
    stub = _ExecutorStub(
        metrics={
            "mini_rung1_loader": (0.60, 0.70),
            "mini_rung_shuffle": (0.59, 0.695),
        }
    )
    _run_walk(
        mini_ladder,
        spec_path,
        monkeypatch,
        stub,
        rung="mini_rung1_loader",
        output_root=root,
    )

    outcome = _run_walk(
        mini_ladder,
        spec_path,
        monkeypatch,
        stub,
        rung="mini_rung_shuffle",
        output_root=root,
    )

    (result,) = outcome.results
    assert result.verdict is Verdict.PASS
    payload, _ = parse_sealed_rung_evidence(
        (root / "mini_rung_shuffle" / "rung_evidence.json").read_bytes()
    )
    assert payload["control"]["rung_id"] == "mini_rung1_loader"
    assert payload["gate"]["control"] == {
        "amp_ssim": 0.60,
        "phase_ssim": 0.70,
    }
    assert payload["gate"]["abs_amp_delta"] == 0.01
    assert payload["gate"]["abs_phase_delta"] == 0.005


def test_diagnostics_report_merges_on_write_across_sibling_invocations(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-1 (third recurrence of the demotion class): the diagnostics report
    must MERGE on every write path — running a sibling diagnostic preserves
    previously adjudicated entries byte-for-byte instead of demoting them to
    skipped placeholders; only a re-adjudication of the SAME rung replaces
    its entry, and placeholders never enter the report."""
    spec_path = _write_ladder_spec(
        mini_ladder,
        rungs=_sampler_rungs(mini_ladder["identity"]),
        groups=_sampler_groups(),
    )
    root = mini_ladder["tmp_path"] / "merge_root"
    stub = _ExecutorStub(
        metrics={
            "mini_rung1_loader": (0.60, 0.70),
            "mini_rung_diag": (0.88, 0.95),
            "mini_rung_shuffle": (0.57, 0.68),
        }
    )
    _run_walk(
        mini_ladder, spec_path, monkeypatch, stub,
        rung="mini_rung1_loader", output_root=root,
    )
    _run_walk(
        mini_ladder, spec_path, monkeypatch, stub,
        rung="mini_rung_diag", output_root=root,
    )
    report_path = root / "ladder_diagnostics_report.json"
    first = json.loads(report_path.read_text())
    (diag_entry,) = [
        e for e in first["rungs"] if e["id"] == "mini_rung_diag"
    ]
    assert diag_entry["status"] == "adjudicated"

    # Sibling diagnostic invocation must not demote mini_rung_diag.
    _run_walk(
        mini_ladder, spec_path, monkeypatch, stub,
        rung="mini_rung_shuffle", output_root=root,
    )
    merged = json.loads(report_path.read_text())
    survivors = {e["id"]: e for e in merged["rungs"]}
    assert survivors["mini_rung_diag"] == diag_entry  # byte-for-byte survival
    assert survivors["mini_rung_shuffle"]["status"] == "adjudicated"
    # Placeholders (skipped/pending) never enter the diagnostics report.
    assert {e["status"] for e in merged["rungs"]} == {"adjudicated"}
    # Re-adjudicating the same rung replaces its own entry (no duplicates).
    _run_walk(
        mini_ladder, spec_path, monkeypatch, stub,
        rung="mini_rung_diag", output_root=root,
    )
    again = json.loads(report_path.read_text())
    assert [e["id"] for e in again["rungs"]].count("mini_rung_diag") == 1


def test_sampler_injection_fails_closed_on_upstream_collision() -> None:
    """S-3: if upstream ever passes sampler/shuffle kwargs for the train
    dataset, the injection must fail loudly, never silently defer."""
    import torch

    from ptycho_torch import dataloader as loader_module
    from scripts.studies.ablation.runtime_errors import RuntimeExecutionError
    from scripts.studies.ablation.runtime_ladder_mmap import (
        _train_sampler_injection,
    )

    train_dataset = list(range(8))
    with _train_sampler_injection(train_dataset, "shuffled", seed=3):
        with pytest.raises(RuntimeExecutionError, match="collided"):
            loader_module.TensorDictDataLoader(
                train_dataset,
                batch_size=2,
                sampler=torch.utils.data.SequentialSampler(train_dataset),
            )
        with pytest.raises(RuntimeExecutionError, match="collided"):
            loader_module.TensorDictDataLoader(
                train_dataset, batch_size=2, shuffle=True
            )


def test_control_from_sealed_evidence_refuses_stale_config(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-2: control loading applies the same rung-id/config checks as the
    chain reuse path — control evidence sealed under a drifted resolved
    config fails closed instead of silently gating against stale metrics."""
    from scripts.studies.ablation.runtime_ladder_reuse import (
        control_from_sealed_evidence,
    )

    spec_path = _write_ladder_spec(
        mini_ladder,
        rungs=_sampler_rungs(mini_ladder["identity"]),
        groups=_sampler_groups(),
    )
    root = mini_ladder["tmp_path"] / "stale_ctrl"
    stub = _ExecutorStub(metrics={"mini_rung1_loader": (0.60, 0.70)})
    _run_walk(
        mini_ladder, spec_path, monkeypatch, stub,
        rung="mini_rung1_loader", output_root=root,
    )
    spec = load_ladder_spec(spec_path, base_dir=mini_ladder["tmp_path"])
    control_rung = spec.rung("mini_rung1_loader")
    control = control_from_sealed_evidence(control_rung, root, None)
    assert control.rung_id == "mini_rung1_loader"
    assert control.amp_ssim == 0.60
    # Drift the sealed resolved config -> fail closed.
    payload, _ = parse_sealed_rung_evidence(
        (root / "mini_rung1_loader" / "rung_evidence.json").read_bytes()
    )
    payload["resolved_config"]["epochs"] = 999
    (root / "mini_rung1_loader" / "rung_evidence.json").unlink()
    seal_rung_evidence(payload, root / "mini_rung1_loader" / "rung_evidence.json")
    with pytest.raises(StudyRequestError, match="different resolved config"):
        control_from_sealed_evidence(control_rung, root, None)


def test_shuffled_sampler_reshuffles_each_epoch_deterministically(
    mini_ladder: dict[str, Any], tmp_path: Path
) -> None:
    """The injected sampler must reshuffle EVERY epoch (like the dictionary
    path) through the same TensorDictDataLoader machinery, deterministically
    given the seed; the sequential default stays byte-preserving."""
    from scripts.studies.ablation import runtime_ladder_diagnostics as diag

    _, config, recipe, _, _ = _rung1_payload(mini_ladder)
    twin = _write_generic_twin(
        mini_ladder["tmp_path"],
        mini_ladder["identity"],
        name="mini_generic_shuffle",
        train_count=24,
        test_count=12,
    )
    for cfg_config in (config,):
        cfg_config.update(fno_modes=2, fno_width=8)

    def capture(sampler_regime: str, work: Path) -> Any:
        run_config = dict(config, mmap_train_sampler=sampler_regime)
        return diag.capture_mmap_training_loaders(
            run_config, recipe, twin["train"], twin["test"], work
        )

    sequential = capture("sequential", tmp_path / "seq")
    assert sequential.sampler_info()["train"]["sampler_class"] == "SequentialSampler"

    shuffled = capture("shuffled", tmp_path / "shuf_a")
    info = shuffled.sampler_info()["train"]
    assert info["loader_class"] == "TensorDictDataLoader"  # same machinery
    assert info["sampler_class"] == "RandomSampler"

    with np.load(twin["train"]) as data:
        index = diag.build_image_identity_index(np.asarray(data["diff3d"]))
    records = diag.record_epochs(shuffled.train_loader, index, epochs=2)
    epoch1, epoch2 = records[0].ordered_ids, records[1].ordered_ids
    assert sorted(epoch1) == sorted(epoch2)  # same multiset
    assert epoch1 != epoch2  # RESHUFFLED between epochs
    # Deterministic across a fresh capture (process-restart equivalent).
    shuffled_again = capture("shuffled", tmp_path / "shuf_b")
    records_again = diag.record_epochs(shuffled_again.train_loader, index, epochs=2)
    assert records_again[0].ordered_ids == epoch1
    assert records_again[1].ordered_ids == epoch2
    # Val loader unchanged.
    assert shuffled.sampler_info()["val"]["sampler_class"] == "SequentialSampler"


def test_probe_layout_injection_flattens_batches_value_preserving(
    mini_ladder: dict[str, Any], tmp_path: Path
) -> None:
    """The historical rung1f injection remains value-preserving evidence.

    Task 25 migrated the current dictionary producer to the contract layout;
    this helper recreates the old flat rank only for sealed-history tests.
    """
    import torch

    from scripts.studies.ablation import runtime_ladder_diagnostics as diag

    _, config, recipe, _, _ = _rung1_payload(mini_ladder)
    twin = _write_generic_twin(
        mini_ladder["tmp_path"],
        mini_ladder["identity"],
        name="mini_generic_probelayout",
        train_count=24,
        test_count=12,
    )
    config.update(fno_modes=2, fno_width=8)

    def capture(layout: str, work: Path) -> Any:
        run_config = dict(config, mmap_probe_batch_shape=layout)
        return diag.capture_mmap_training_loaders(
            run_config, recipe, twin["train"], twin["test"], work
        )

    modes = capture("modes", tmp_path / "modes")
    probe_modes = next(iter(modes.train_loader))[1]
    assert probe_modes.ndim == 5  # (B, C=1, P=1, H, W) current convention

    flat = capture("dictionary_flat", tmp_path / "flat")
    for loader_name in ("train_loader", "val_loader"):
        batch = next(iter(getattr(flat, loader_name)))
        probe_flat = batch[1]
        assert probe_flat.ndim == 3, loader_name  # historical flat rank
    probe_flat = next(iter(flat.train_loader))[1]
    expected = probe_modes[: probe_flat.shape[0]].reshape(
        probe_flat.shape[0], probe_flat.shape[-2], probe_flat.shape[-1]
    )
    assert torch.equal(probe_flat.cpu(), expected.cpu())  # values untouched


def test_probe_layout_injection_refuses_multimode() -> None:
    """Flattening is only defined for single-mode (B,1,1,H,W); a multi-mode
    probe cannot be represented in the dictionary convention -> fail closed."""
    import torch

    from scripts.studies.ablation.runtime_errors import RuntimeExecutionError
    from scripts.studies.ablation.runtime_ladder_mmap import (
        _flatten_probe_batch,
    )

    single = torch.ones((4, 1, 1, 8, 8), dtype=torch.complex64)
    flattened = _flatten_probe_batch(single)
    assert flattened.shape == (4, 8, 8)
    with pytest.raises(RuntimeExecutionError, match="multi-mode"):
        _flatten_probe_batch(torch.ones((4, 1, 2, 8, 8), dtype=torch.complex64))


def test_probe_layout_field_is_migration_whitelisted(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """mmap_probe_batch_shape ships with its migration entry: pre-field
    sealed evidence reuses under the 'modes' default and refuses the
    non-default value."""
    from scripts.studies.ablation.runtime_ladder_config import (
        MIGRATED_CONFIG_FIELDS,
    )

    assert MIGRATED_CONFIG_FIELDS["mmap_probe_batch_shape"] == "modes"
    spec_path = _write_ladder_spec(mini_ladder)
    root_a = mini_ladder["tmp_path"] / "probelayout_src"
    _run_walk(
        mini_ladder, spec_path, monkeypatch, _ExecutorStub(), output_root=root_a
    )
    root_b = mini_ladder["tmp_path"] / "probelayout_reuse"
    chain_sha: str | None = None
    for rung_id in ("mini_rung1_loader", "mini_rung2_evaluator", "mini_rung3_varpro"):
        chain_sha = _synthesize_prefield_evidence(
            root_a, root_b, rung_id, "mmap_probe_batch_shape", control_sha=chain_sha
        )
    stub = _ExecutorStub()
    outcome = _run_walk(
        mini_ladder, spec_path, monkeypatch, stub, output_root=root_b
    )
    assert stub.executed == []
    assert outcome.passed


def test_checked_spec_keeps_only_contract_probe_layout() -> None:
    spec = load_ladder_spec(CHECKED_SPEC, base_dir=REPO_ROOT)

    for rung in spec.rungs:
        assert rung.resolved_config["mmap_probe_batch_shape"] == "modes"
    assert spec.endpoint_config["mmap_probe_batch_shape"] == "modes"


def test_sampler_field_is_migration_whitelisted(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """mmap_train_sampler ships with its migration entry: pre-field sealed
    evidence reuses under the sequential default and refuses otherwise."""
    from scripts.studies.ablation.runtime_ladder_config import (
        MIGRATED_CONFIG_FIELDS,
    )

    assert MIGRATED_CONFIG_FIELDS["mmap_train_sampler"] == "sequential"

    spec_path = _write_ladder_spec(mini_ladder)
    root_a = mini_ladder["tmp_path"] / "sampler_prefield_src"
    _run_walk(mini_ladder, spec_path, monkeypatch, _ExecutorStub(), output_root=root_a)
    root_b = mini_ladder["tmp_path"] / "sampler_prefield_reuse"
    chain_sha: str | None = None
    for rung_id in ("mini_rung1_loader", "mini_rung2_evaluator", "mini_rung3_varpro"):
        chain_sha = _synthesize_prefield_evidence(
            root_a, root_b, rung_id, "mmap_train_sampler", control_sha=chain_sha
        )
    stub = _ExecutorStub()
    outcome = _run_walk(mini_ladder, spec_path, monkeypatch, stub, output_root=root_b)
    assert stub.executed == []
    assert outcome.passed

    # Non-default value refuses (shuffled diagnostic evidence without field).
    rungs = _sampler_rungs(mini_ladder["identity"])
    spec2 = _write_ladder_spec(mini_ladder, rungs=rungs, groups=_sampler_groups())
    root_c = mini_ladder["tmp_path"] / "sampler_nondefault_src"
    stub2 = _ExecutorStub(metrics={"mini_rung1_loader": (0.6, 0.7)})
    _run_walk(mini_ladder, spec2, monkeypatch, stub2, rung="mini_rung1_loader", output_root=root_c)
    _run_walk(mini_ladder, spec2, monkeypatch, stub2, rung="mini_rung_shuffle", output_root=root_c)
    root_d = mini_ladder["tmp_path"] / "sampler_nondefault_refuse"
    import shutil as _shutil

    _shutil.copytree(root_c / "mini_rung1_loader", root_d / "mini_rung1_loader")
    _synthesize_prefield_evidence(
        root_c, root_d, "mini_rung_shuffle", "mmap_train_sampler"
    )
    with pytest.raises(StudyRequestError, match="different resolved config"):
        _run_walk(
            mini_ladder, spec2, monkeypatch, _ExecutorStub(),
            rung="mini_rung_shuffle", output_root=root_d,
        )


def test_diagnostics_may_share_a_group_but_chain_rungs_may_not(
    mini_ladder: dict[str, Any],
) -> None:
    # Two diagnostics on the same group: accepted (branches probe the same
    # variable from different controls).
    spec = load_ladder_spec(
        _write_ladder_spec(
            mini_ladder,
            rungs=_sampler_rungs(mini_ladder["identity"]),
            groups=_sampler_groups(),
        ),
        base_dir=mini_ladder["tmp_path"],
    )
    assert spec.rung("mini_rung_shuffle").group == "ingestion_sampler"
    assert spec.rung("mini_rung_shuffle_parity").group == "ingestion_sampler"


# ---------------------------------------------------------------------------
# Generic-schema twin converter (Task 21b deliverable 2)
# ---------------------------------------------------------------------------


def _write_dictionary_pair(
    tmp_path: Path, identity: dict[str, Any], *, name: str, counts: bool = False
) -> dict[str, Path]:
    """Write a miniature dictionary-schema pair WITH global coords_offsets."""
    rng = np.random.default_rng(41)
    out: dict[str, Path] = {}
    for split, count, with_truth in (("train", 12, False), ("test", 12, True)):
        diffraction = rng.random((count, N_SMALL, N_SMALL, 1)).astype(np.float32)
        if counts:
            diffraction = np.round(diffraction * 100.0).astype(np.float32)
        offsets = np.zeros((count, 1, 2, 1), dtype=np.float32)
        offsets[:, 0, 0, 0] = 30.0 + 3.0 * np.arange(count)  # y (rows)
        offsets[:, 0, 1, 0] = 50.0 + 2.0 * np.arange(count)  # x (cols)
        payload: dict[str, Any] = {
            "diffraction": diffraction,
            "Y_I": rng.random((count, N_SMALL, N_SMALL, 1)).astype(np.float32),
            "Y_phi": rng.random((count, N_SMALL, N_SMALL, 1)).astype(np.float32),
            "coords_nominal": np.zeros((count, 1, 2, 1), dtype=np.float32),
            "coords_offsets": offsets,
            "probeGuess": identity["probe"],
        }
        if with_truth:
            payload["YY_ground_truth"] = (
                rng.normal(size=(*GT_SHAPE, 1)) + 1j * rng.normal(size=(*GT_SHAPE, 1))
            ).astype(np.complex64)
        path = tmp_path / name / f"{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **payload)
        out[split] = path
    return out


def test_generic_schema_twin_converter_round_trip(
    mini_ladder: dict[str, Any],
) -> None:
    from scripts.studies.ablation.datasets import validate_ladder_npz_pair
    from scripts.studies.make_generic_schema_twin import convert_pair

    tmp_path = mini_ladder["tmp_path"]
    identity = mini_ladder["identity"]
    pair = _write_dictionary_pair(tmp_path, identity, name="dict_pair")
    out_dir = tmp_path / "generic_twin_out"

    provenance = convert_pair(pair["train"], pair["test"], out_dir)

    with np.load(pair["test"]) as src, np.load(out_dir / "test.npz") as dst:
        # Measurement values pass through bit-identically.
        np.testing.assert_array_equal(dst["diff3d"], src["diffraction"][..., 0])
        assert dst["diff3d"].dtype == src["diffraction"].dtype
        # Global positions: channel 0 = y (rows), channel 1 = x (cols).
        expected = src["coords_offsets"][:, 0, :, 0] + src["coords_nominal"][:, 0, :, 0]
        np.testing.assert_allclose(dst["ycoords"], expected[:, 0])
        np.testing.assert_allclose(dst["xcoords"], expected[:, 1])
        np.testing.assert_array_equal(dst["probeGuess"], src["probeGuess"])
        np.testing.assert_array_equal(
            dst["objectGuess"], np.squeeze(src["YY_ground_truth"])
        )
        assert dst["objectGuess"].ndim == 2
    with np.load(out_dir / "train.npz") as train_dst:
        # Train carries the loader's all-ones "no object" placeholder (2-D).
        placeholder = train_dst["objectGuess"]
        assert placeholder.ndim == 2
        assert placeholder.sum().real == placeholder.shape[0] * placeholder.shape[1]
    assert provenance["tool"] == "grid_lines_generic_schema_twin_v1"
    assert provenance["probe_lineage"]["input_probe_archive"] is None
    source_probe = provenance["probe_lineage"][
        "dictionary_source_stored_transformed"
    ]
    output_probe = provenance["probe_lineage"]["generic_output"]
    assert source_probe["splits_equal"] is True
    assert output_probe["splits_equal"] is True
    assert output_probe["output_equal"] is True
    for split in ("train", "test"):
        expected_descriptor = {
            "canonical_sha256": identity["transformed_probe_sha256"],
            "dtype": "complex64",
            "shape": [N_SMALL, N_SMALL],
        }
        assert source_probe[split] == expected_descriptor
        assert output_probe[split] == expected_descriptor
    for split in ("train", "test"):
        assert provenance[split]["source_sha256"]
        assert provenance[split]["output_sha256"] == hashlib.sha256(
            (out_dir / f"{split}.npz").read_bytes()
        ).hexdigest()

    # The output is a valid ladder generic pair for the miniature recipe.
    spec = load_ladder_spec(_write_ladder_spec(mini_ladder), base_dir=tmp_path)
    materialized = validate_ladder_npz_pair(
        spec.dataset("mini_generic"), out_dir / "train.npz", out_dir / "test.npz"
    )
    assert materialized.n_train == 12
    assert materialized.probe_sha256 == identity["transformed_probe_sha256"]


def test_generic_schema_twin_converter_direct_cli_loads_repository_helpers() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/studies/make_generic_schema_twin.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--probe-archive" in completed.stdout


def test_generic_schema_twin_converter_computes_optional_probe_archive_lineage(
    mini_ladder: dict[str, Any],
) -> None:
    from scripts.studies.ablation.dataset_provenance import canonical_array_sha256
    from scripts.studies.make_generic_schema_twin import convert_pair

    tmp_path = mini_ladder["tmp_path"]
    pair = _write_dictionary_pair(tmp_path, mini_ladder["identity"], name="dict_probe")
    raw_probe = (
        np.arange(36, dtype=np.float64).reshape(6, 6)
        + 1j * np.arange(36, dtype=np.float64).reshape(6, 6)
    ).astype(np.complex128)
    probe_archive = tmp_path / "raw_probe.npz"
    np.savez(probe_archive, probeGuess=raw_probe)

    provenance = convert_pair(
        pair["train"],
        pair["test"],
        tmp_path / "generic_probe",
        probe_archive=probe_archive,
    )

    assert provenance["probe_lineage"]["input_probe_archive"] == {
        "path": str(probe_archive),
        "file_sha256": hashlib.sha256(probe_archive.read_bytes()).hexdigest(),
        "probe_key": "probeGuess",
        "raw_probe_array_canonical_sha256": canonical_array_sha256(raw_probe),
    }


def test_generic_schema_twin_converter_rejects_probe_mismatch_between_splits(
    mini_ladder: dict[str, Any],
) -> None:
    from scripts.studies.make_generic_schema_twin import ConversionError, convert_pair

    tmp_path = mini_ladder["tmp_path"]
    pair = _write_dictionary_pair(tmp_path, mini_ladder["identity"], name="dict_probe_mismatch")
    with np.load(pair["test"], allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["probeGuess"] = payload["probeGuess"] * np.complex64(2.0)
    np.savez(pair["test"], **payload)

    with pytest.raises(ConversionError, match="source train/test probeGuess mismatch"):
        convert_pair(pair["train"], pair["test"], tmp_path / "mismatched_probe")


@pytest.mark.parametrize("mutation", ["reshape", "reinterpret"])
def test_generic_schema_twin_converter_rejects_same_byte_source_descriptor_mismatch(
    mini_ladder: dict[str, Any], mutation: str
) -> None:
    from scripts.studies.ablation.dataset_provenance import canonical_array_sha256
    from scripts.studies.make_generic_schema_twin import ConversionError, convert_pair

    tmp_path = mini_ladder["tmp_path"]
    pair = _write_dictionary_pair(
        tmp_path, mini_ladder["identity"], name=f"dict_probe_{mutation}_mismatch"
    )
    with np.load(pair["test"], allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    original = np.ascontiguousarray(payload["probeGuess"])
    mutated = (
        original.reshape(N_SMALL // 2, N_SMALL * 2)
        if mutation == "reshape"
        else original.view(np.float64)
    )
    assert canonical_array_sha256(mutated) == canonical_array_sha256(original)
    assert (mutated.shape, mutated.dtype) != (original.shape, original.dtype)
    payload["probeGuess"] = mutated
    np.savez(pair["test"], **payload)

    with pytest.raises(ConversionError, match="source train/test probeGuess mismatch"):
        convert_pair(pair["train"], pair["test"], tmp_path / f"{mutation}_mismatch")


def test_generic_schema_twin_converter_rejects_serialized_probe_tamper(
    mini_ladder: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.studies import make_generic_schema_twin as converter

    tmp_path = mini_ladder["tmp_path"]
    pair = _write_dictionary_pair(tmp_path, mini_ladder["identity"], name="dict_probe_tamper")
    serialize = converter._npz_bytes

    def _tampered_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
        tampered = dict(arrays)
        tampered["probeGuess"] = np.asarray(arrays["probeGuess"]) * np.complex64(2.0)
        return serialize(tampered)

    monkeypatch.setattr(converter, "_npz_bytes", _tampered_npz_bytes)
    with pytest.raises(
        converter.ConversionError,
        match="generic output train probeGuess mismatch",
    ):
        converter.convert_pair(pair["train"], pair["test"], tmp_path / "tampered_probe")


@pytest.mark.parametrize("mutation", ["reshape", "reinterpret"])
def test_generic_schema_twin_converter_rejects_same_byte_output_descriptor_tamper(
    mini_ladder: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from scripts.studies.ablation.dataset_provenance import canonical_array_sha256
    from scripts.studies import make_generic_schema_twin as converter

    tmp_path = mini_ladder["tmp_path"]
    pair = _write_dictionary_pair(
        tmp_path, mini_ladder["identity"], name=f"dict_output_{mutation}_tamper"
    )
    serialize = converter._npz_bytes

    def _tampered_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
        tampered = dict(arrays)
        original = np.ascontiguousarray(arrays["probeGuess"])
        mutated = (
            original.reshape(N_SMALL // 2, N_SMALL * 2)
            if mutation == "reshape"
            else original.view(np.float64)
        )
        assert canonical_array_sha256(mutated) == canonical_array_sha256(original)
        tampered["probeGuess"] = mutated
        return serialize(tampered)

    monkeypatch.setattr(converter, "_npz_bytes", _tampered_npz_bytes)
    with pytest.raises(
        converter.ConversionError,
        match="generic output train probeGuess mismatch",
    ):
        converter.convert_pair(
            pair["train"], pair["test"], tmp_path / f"{mutation}_tamper"
        )


def test_generic_schema_twin_converter_empty_probe_archive_is_controlled_cli_error(
    mini_ladder: dict[str, Any],
) -> None:
    from scripts.studies.make_generic_schema_twin import ConversionError, convert_pair

    tmp_path = mini_ladder["tmp_path"]
    pair = _write_dictionary_pair(tmp_path, mini_ladder["identity"], name="dict_empty_archive")
    probe_archive = tmp_path / "empty_probe.npz"
    probe_archive.write_bytes(b"")

    with pytest.raises(ConversionError, match="input probe archive.*probeGuess"):
        convert_pair(
            pair["train"],
            pair["test"],
            tmp_path / "empty_archive_api",
            probe_archive=probe_archive,
        )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/studies/make_generic_schema_twin.py"),
            "--train-npz",
            str(pair["train"]),
            "--test-npz",
            str(pair["test"]),
            "--output-dir",
            str(tmp_path / "empty_archive_cli"),
            "--probe-archive",
            str(probe_archive),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_generic_schema_twin_converter_is_deterministic(
    mini_ladder: dict[str, Any],
) -> None:
    from scripts.studies.make_generic_schema_twin import convert_pair

    tmp_path = mini_ladder["tmp_path"]
    pair = _write_dictionary_pair(tmp_path, mini_ladder["identity"], name="dict_det")
    out_a = tmp_path / "twin_a"
    out_b = tmp_path / "twin_b"

    convert_pair(pair["train"], pair["test"], out_a)
    convert_pair(pair["train"], pair["test"], out_b)

    for split in ("train", "test"):
        assert (out_a / f"{split}.npz").read_bytes() == (
            out_b / f"{split}.npz"
        ).read_bytes()


def test_generic_schema_twin_converter_requires_truth_and_offsets(
    mini_ladder: dict[str, Any],
) -> None:
    from scripts.studies.make_generic_schema_twin import ConversionError, convert_pair

    tmp_path = mini_ladder["tmp_path"]
    pair = _write_dictionary_pair(tmp_path, mini_ladder["identity"], name="dict_bad")
    with np.load(pair["test"]) as data:
        payload = {k: data[k] for k in data.files if k != "YY_ground_truth"}
    np.savez(pair["test"], **payload)

    with pytest.raises(ConversionError, match="YY_ground_truth|YY_full"):
        convert_pair(pair["train"], pair["test"], tmp_path / "twin_bad")

    pair2 = _write_dictionary_pair(
        tmp_path, mini_ladder["identity"], name="dict_bad_offsets"
    )
    with np.load(pair2["train"]) as data:
        payload = {k: data[k] for k in data.files if k != "coords_offsets"}
    np.savez(pair2["train"], **payload)
    with pytest.raises(ConversionError, match="coords_offsets"):
        convert_pair(pair2["train"], pair2["test"], tmp_path / "twin_bad_offsets")


# ---------------------------------------------------------------------------
# Facade re-exports
# ---------------------------------------------------------------------------


def test_runtime_facade_reexports_ladder_api() -> None:
    assert runtime.run_bridge_ladder is runtime_ladder.run_bridge_ladder
    assert runtime.load_ladder_spec is load_ladder_spec
    assert "run_bridge_ladder" in runtime.__all__
    assert "LadderRequest" in runtime.__all__


def test_datasets_facade_reexports_ladder_dataset_api() -> None:
    from scripts.studies.ablation import dataset_reference, datasets

    assert datasets.parse_ladder_dataset is dataset_reference.parse_ladder_dataset
    assert (
        datasets.validate_ladder_npz_pair is dataset_reference.validate_ladder_npz_pair
    )
    assert "LadderDatasetRecipe" in datasets.__all__
