"""Tests for the BRDT four-row preflight item.

Covers:

- the decision-support dataset profile contract (distinct identity,
  claim boundary, default counts, manifest fields);
- non-destructive separation between smoke and decision-support
  artifact roots;
- the four-row preflight orchestration entrypoint
  (``scripts.studies.born_rytov_dt.run_preflight``):
  row roster resolution, fixed sample-id locking, preflight-manifest
  schema, train/eval handoff, row-level blocker serialization, and the
  refusal to mix mismatched dataset/operator/input contracts across rows.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict

import h5py  # type: ignore
import numpy as np
import pytest
import torch

from scripts.studies.born_rytov_dt import dataset_contract as dc
from scripts.studies.born_rytov_dt import generate_brdt_dataset as gen
from scripts.studies.born_rytov_dt import run_config
from scripts.studies.born_rytov_dt import run_preflight as preflight_mod


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_MODULE = "scripts.studies.born_rytov_dt.generate_brdt_dataset"
PREFLIGHT_MODULE = "scripts.studies.born_rytov_dt.run_preflight"


def _run_generator(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = ["python", "-m", GENERATOR_MODULE, *args]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def _run_preflight(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["python", "-m", PREFLIGHT_MODULE, *args]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


# ----------------------------------------------------------------------
# Profile registry
# ----------------------------------------------------------------------
def test_profile_registry_includes_smoke_and_decision_support():
    assert "smoke" in dc.PROFILE_REGISTRY
    assert "decision_support" in dc.PROFILE_REGISTRY


def test_decision_support_profile_identity_distinct_from_smoke():
    smoke = dc.get_dataset_profile("smoke")
    ds = dc.get_dataset_profile("decision_support")
    # Distinct dataset name, tier, backlog item, claim boundary.
    assert smoke.name != ds.name
    assert smoke.tier != ds.tier
    assert smoke.backlog_item != ds.backlog_item
    assert smoke.claim_boundary != ds.claim_boundary
    # Decision-support is the larger capped split.
    assert ds.default_counts.train > smoke.default_counts.train
    assert ds.default_counts.train == 2048
    assert ds.default_counts.val == 256
    assert ds.default_counts.test == 256


def test_get_dataset_profile_unknown_rejected():
    with pytest.raises(ValueError, match="unknown dataset profile"):
        dc.get_dataset_profile("paper_grade")


# ----------------------------------------------------------------------
# Decision-support manifest contract
# ----------------------------------------------------------------------
def _build_decision_support_manifest(tmp_path: Path) -> Dict:
    profile = dc.get_dataset_profile("decision_support")
    counts = profile.default_counts
    seeds = dc.deterministic_object_seeds(counts, split_seed=42)
    families = dc.assign_phantom_families(counts, split_seed=42)
    rng = np.random.default_rng(0)
    q_train = rng.standard_normal((4, 4, 4)) * 0.01
    stats = dc.compute_train_normalization(q_train)
    return dc.build_manifest(
        output_root=str(tmp_path),
        operator_validation_path=str(tmp_path / "operator_validation.json"),
        counts=counts,
        split_seed=42,
        object_seeds=seeds,
        families=families,
        normalization=stats,
        noise_sigma=1e-3,
        measured_snr=None,
        git_sha="deadbeef",
        git_dirty=False,
        generation_command="python -m scripts.studies.born_rytov_dt.generate_brdt_dataset --dataset-profile decision_support",
        environment={"python": "3.11"},
        artifact_paths={"train": "train.h5", "val": "val.h5", "test": "test.h5"},
        profile=profile,
    )


def test_decision_support_manifest_carries_distinct_identity(tmp_path):
    manifest = _build_decision_support_manifest(tmp_path)
    assert manifest["dataset_identity"]["name"] == dc.DECISION_SUPPORT_DATASET_NAME
    assert manifest["dataset_identity"]["tier"] == "decision_support"
    assert (
        manifest["dataset_identity"]["backlog_item"]
        == "2026-04-29-brdt-four-row-preflight"
    )
    # Claim boundary must NOT match the smoke claim boundary.
    assert manifest["claim_boundary"] != dc.SMOKE_CLAIM_BOUNDARY
    assert "decision_support" in manifest["claim_boundary"].lower()


def test_decision_support_manifest_required_keys(tmp_path):
    manifest = _build_decision_support_manifest(tmp_path)
    for key in dc.manifest_required_keys():
        assert key in manifest, f"missing required key {key}"
    # Operator authority block must still match the locked geometry.
    op = manifest["operator"]
    assert op["mode"] == "born"
    assert op["grid_size"] == 128
    assert op["angle_count"] == 64
    assert op["wavelength_px"] == 8.0


def test_smoke_manifest_is_not_authoritative_for_decision_support(tmp_path):
    profile = dc.get_dataset_profile("decision_support")
    counts = profile.default_counts
    seeds = dc.deterministic_object_seeds(counts, split_seed=42)
    families = dc.assign_phantom_families(counts, split_seed=42)
    smoke_manifest = dc.build_manifest(
        output_root=str(tmp_path),
        operator_validation_path=str(tmp_path / "op.json"),
        counts=counts,
        split_seed=42,
        object_seeds=seeds,
        families=families,
        normalization=None,
        noise_sigma=1e-3,
        measured_snr=None,
        git_sha="x",
        git_dirty=False,
        generation_command="cmd",
        environment={"python": "3.11"},
        artifact_paths={"train": "t.h5", "val": "v.h5", "test": "te.h5"},
        # Default is the smoke profile.
    )
    assert smoke_manifest["dataset_identity"]["name"] == dc.DATASET_NAME
    assert (
        smoke_manifest["dataset_identity"]["backlog_item"]
        == "2026-04-29-brdt-dataset-preflight"
    )
    # The four-row preflight orchestrator must refuse to use a smoke
    # manifest as the decision-support dataset authority.
    with pytest.raises(ValueError, match="decision_support"):
        preflight_mod.assert_decision_support_manifest(smoke_manifest)


def test_decision_support_manifest_passes_orchestrator_authority(tmp_path):
    manifest = _build_decision_support_manifest(tmp_path)
    # Should not raise.
    preflight_mod.assert_decision_support_manifest(manifest)


# ----------------------------------------------------------------------
# Generator wiring for the decision-support profile
# ----------------------------------------------------------------------
def test_dry_run_decision_support_default_counts_under_distinct_root(tmp_path):
    output_root = tmp_path / "ds_root"
    args = [
        "--dataset-profile",
        "decision_support",
        "--dry-run-manifest",
        "--output-root",
        str(output_root),
        "--split-seed",
        "13",
    ]
    _run_generator(*args)
    summary = json.loads((output_root / "dry_run_summary.json").read_text())
    manifest = json.loads((output_root / dc.DRY_RUN_MANIFEST_NAME).read_text())
    assert summary["split"]["counts"] == {"train": 2048, "val": 256, "test": 256}
    assert manifest["dataset_identity"]["name"] == dc.DECISION_SUPPORT_DATASET_NAME
    assert manifest["dataset_identity"]["tier"] == "decision_support"
    # The estimated artifact paths should sit under the distinct root.
    estimated = summary["estimated_artifact_paths"]
    for split_path in estimated.values():
        assert str(output_root) in split_path
        assert dc.DECISION_SUPPORT_DATASET_NAME in split_path


def test_dry_run_decision_support_count_overrides_recorded(tmp_path):
    output_root = tmp_path / "ds_override"
    args = [
        "--dataset-profile",
        "decision_support",
        "--dry-run-manifest",
        "--output-root",
        str(output_root),
        "--train-count",
        "32",
        "--val-count",
        "8",
        "--test-count",
        "8",
    ]
    _run_generator(*args)
    summary = json.loads((output_root / "dry_run_summary.json").read_text())
    assert summary["split"]["counts"] == {"train": 32, "val": 8, "test": 8}


def test_dry_run_decision_support_does_not_overwrite_smoke_root(tmp_path):
    smoke_root = tmp_path / "smoke_root"
    ds_root = tmp_path / "ds_root"
    _run_generator(
        "--dataset-profile",
        "smoke",
        "--dry-run-manifest",
        "--output-root",
        str(smoke_root),
        "--split-seed",
        "1",
    )
    smoke_manifest_before = (smoke_root / dc.DRY_RUN_MANIFEST_NAME).read_bytes()

    _run_generator(
        "--dataset-profile",
        "decision_support",
        "--dry-run-manifest",
        "--output-root",
        str(ds_root),
        "--split-seed",
        "1",
    )

    assert (smoke_root / dc.DRY_RUN_MANIFEST_NAME).read_bytes() == smoke_manifest_before
    # Distinct files exist under each root.
    assert (ds_root / dc.DRY_RUN_MANIFEST_NAME).exists()
    assert (ds_root / dc.DRY_RUN_MANIFEST_NAME).read_bytes() != smoke_manifest_before


def test_live_decision_support_writes_distinct_h5_files(tmp_path):
    output_root = tmp_path / "ds_live"
    _run_generator(
        "--dataset-profile",
        "decision_support",
        "--output-root",
        str(output_root),
        "--device",
        "cpu",
        "--split-seed",
        "13",
        "--train-count",
        "2",
        "--val-count",
        "1",
        "--test-count",
        "1",
    )
    manifest = json.loads((output_root / "dataset_manifest.json").read_text())
    assert manifest["dataset_identity"]["name"] == dc.DECISION_SUPPORT_DATASET_NAME
    for split in ("train", "val", "test"):
        path = Path(manifest["artifacts"][split])
        assert path.exists()
        assert dc.DECISION_SUPPORT_DATASET_NAME in path.name
        with h5py.File(path, "r") as fh:
            assert fh.attrs["dataset_name"] == dc.DECISION_SUPPORT_DATASET_NAME


# ----------------------------------------------------------------------
# Preflight orchestrator: row roster + fixed sample IDs
# ----------------------------------------------------------------------
def _make_live_decision_support_dataset(tmp_path: Path) -> Path:
    output_root = tmp_path / "ds_live"
    _run_generator(
        "--dataset-profile",
        "decision_support",
        "--output-root",
        str(output_root),
        "--device",
        "cpu",
        "--split-seed",
        "13",
        "--train-count",
        "2",
        "--val-count",
        "2",
        "--test-count",
        "2",
    )
    return output_root / "dataset_manifest.json"


def test_resolve_row_roster_default_uses_born_init_image(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    roster = preflight_mod.resolve_row_roster(
        manifest_path=manifest_path,
        hybrid_label="hybrid_resnet",
    )
    assert [r.row_id for r in roster] == [
        "classical_born_backprop",
        "unet",
        "fno_vanilla",
        "hybrid_resnet",
    ]
    for row in roster:
        assert row.input_mode == "born_init_image"


def test_resolve_row_roster_sru_label_freezes_visible_id(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    roster = preflight_mod.resolve_row_roster(
        manifest_path=manifest_path,
        hybrid_label="sru_net",
    )
    visible = {r.row_id for r in roster}
    # Only one Hybrid-family label appears in the roster.
    assert "sru_net" in visible
    assert "hybrid_resnet" not in visible


def test_choose_fixed_sample_ids_locks_test_split(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    ids_a = preflight_mod.choose_fixed_sample_ids(manifest_path, count=1, seed=0)
    ids_b = preflight_mod.choose_fixed_sample_ids(manifest_path, count=1, seed=0)
    assert ids_a == ids_b
    assert len(ids_a) == 1
    # IDs must come from the test split index range (size=2 in this fixture).
    counts = json.loads(Path(manifest_path).read_text())["split"]["counts"]
    assert all(0 <= i < counts["test"] for i in ids_a)


# ----------------------------------------------------------------------
# Preflight orchestrator: dry-run gates + manifest schema
# ----------------------------------------------------------------------
def test_run_preflight_dry_run_writes_manifest_and_row_roster(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "preflight"
    _run_preflight(
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(preflight_root),
        "--dry-run",
    )
    manifest = json.loads((preflight_root / "preflight_manifest.json").read_text())
    assert manifest["claim_boundary"] == "decision_support_preflight_only"
    assert (
        manifest["next_backlog_item"]
        == "2026-04-29-brdt-preflight-summary-promotion-decision"
    )
    row_ids = [r["row_id"] for r in manifest["rows"]]
    assert row_ids == [
        "classical_born_backprop",
        "unet",
        "fno_vanilla",
        "hybrid_resnet",
    ]
    # The manifest must point to the decision-support dataset, not the smoke
    # dataset.
    assert (
        manifest["dataset"]["dataset_id"] == dc.DECISION_SUPPORT_DATASET_NAME
    )
    assert manifest["dataset"]["tier"] == "decision_support"
    # Fixed sample IDs are locked at dry-run time.
    assert isinstance(manifest["fixed_sample_ids"], list)
    assert len(manifest["fixed_sample_ids"]) >= 1
    # Schema fields present.
    assert "metric_schema" in manifest
    assert manifest["metric_schema"]["version"]
    # Operator authority pointer survives.
    assert manifest["operator"]["validation_artifact"]


def test_run_preflight_refuses_smoke_manifest(tmp_path):
    # Generate a smoke-profile dataset and try to drive the preflight from it.
    smoke_root = tmp_path / "smoke_live"
    _run_generator(
        "--dataset-profile",
        "smoke",
        "--output-root",
        str(smoke_root),
        "--device",
        "cpu",
        "--split-seed",
        "1",
        "--train-count",
        "2",
        "--val-count",
        "1",
        "--test-count",
        "1",
    )
    manifest_path = smoke_root / "dataset_manifest.json"
    preflight_root = tmp_path / "smoke_preflight"
    result = _run_preflight(
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(preflight_root),
        "--dry-run",
        check=False,
    )
    assert result.returncode != 0
    assert "decision_support" in (result.stdout + result.stderr)


def test_serialize_blocked_row_includes_reason():
    blocked = run_config.make_blocked_row(
        row_id="fno_vanilla",
        model="fno_vanilla",
        training=run_config.DEFAULT_TRAINING_LABEL,
        dataset_id="brdt128_decision_support_preflight",
        operator_version="op",
        blocker_reason="neuralop_unavailable",
        blocker_message="neuralop is not installed",
    )
    payload = blocked.to_dict()
    assert payload["row_status"] == "blocked"
    assert payload["blocker_reason"] == "neuralop_unavailable"
    assert payload["blocker_message"]


def test_run_preflight_serializes_argv_and_invocation(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "preflight_inv"
    _run_preflight(
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(preflight_root),
        "--dry-run",
    )
    invocation = json.loads((preflight_root / "invocation.json").read_text())
    assert invocation["script"] == "scripts/studies/born_rytov_dt/run_preflight.py"
    assert invocation["extra"]["backlog_item"] == "2026-04-29-brdt-four-row-preflight"


# ----------------------------------------------------------------------
# Writer lock (duplicate-writer protection)
# ----------------------------------------------------------------------
def test_writer_lock_acquires_and_releases(tmp_path):
    output_root = tmp_path / "lock_basic"
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / preflight_mod.WRITER_LOCK_NAME
    with preflight_mod.writer_lock(output_root):
        held = json.loads(lock_path.read_text())
        assert held["pid"] == os.getpid()
        assert held["script"] == preflight_mod.SCRIPT_PATH
    # Lock removed on exit.
    assert not lock_path.exists()


def test_writer_lock_reclaims_stale_pid(tmp_path):
    output_root = tmp_path / "lock_stale"
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / preflight_mod.WRITER_LOCK_NAME
    # Pick a PID extremely unlikely to be alive (32-bit max-1).
    lock_path.write_text(
        json.dumps({"pid": 2**31 - 2, "started_at": "old", "host": "x"})
    )
    with preflight_mod.writer_lock(output_root):
        active = json.loads(lock_path.read_text())
        assert active["pid"] == os.getpid()
    assert not lock_path.exists()


def test_writer_lock_refuses_active_writer(tmp_path):
    """A live PID lock must block a duplicate writer."""
    output_root = tmp_path / "lock_active"
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / preflight_mod.WRITER_LOCK_NAME
    # The pytest parent process is alive; using its PID simulates an
    # active concurrent writer different from the current test thread.
    parent_pid = os.getppid()
    lock_path.write_text(
        json.dumps({"pid": parent_pid, "started_at": "now", "host": "x"})
    )
    with pytest.raises(ValueError, match="active BRDT preflight writer"):
        with preflight_mod.writer_lock(output_root):
            pass
    # The original lock must be left intact (we did not steal it).
    held = json.loads(lock_path.read_text())
    assert held["pid"] == parent_pid


def test_run_preflight_subprocess_refuses_active_writer(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "active_writer"
    preflight_root.mkdir(parents=True, exist_ok=True)
    lock_path = preflight_root / preflight_mod.WRITER_LOCK_NAME
    # Plant a lock pointing at the (alive) pytest process.
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "started_at": "now", "host": "x"})
    )
    res = _run_preflight(
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(preflight_root),
        "--dry-run",
        check=False,
    )
    assert res.returncode != 0
    combined = res.stdout + res.stderr
    assert "active BRDT preflight writer" in combined
    # Plant lock survives (not reclaimed).
    held = json.loads(lock_path.read_text())
    assert held["pid"] == os.getpid()


# ----------------------------------------------------------------------
# Classical narrow-fix attempt
# ----------------------------------------------------------------------
def test_attempt_classical_backend_with_fix_records_attempts():
    backend, attempts = preflight_mod.attempt_classical_backend_with_fix()
    assert isinstance(attempts, list) and attempts
    if backend.name == "odtbrain":
        # ODTbrain installed → at least one succeeded attempt recorded.
        assert any(a.get("outcome") == "succeeded" for a in attempts)
    else:
        # Without ODTbrain, both the initial import and the cache-invalidation
        # retry must be recorded before declaring the row blocked.
        assert backend.name == "local_adjoint"
        assert backend.reason == "odtbrain_unavailable_after_narrow_fix"
        steps = [a["step"] for a in attempts]
        assert "import_odtbrain" in steps
        assert any("invalidate_caches" in s for s in steps)


def test_dry_run_manifest_records_narrow_fix_attempts(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "narrow_fix"
    _run_preflight(
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(preflight_root),
        "--dry-run",
    )
    manifest = json.loads((preflight_root / "preflight_manifest.json").read_text())
    notes = manifest.get("notes") or {}
    attempts = notes.get("classical_narrow_fix_attempts")
    assert attempts, "classical_narrow_fix_attempts must be recorded"
    steps = [a["step"] for a in attempts]
    assert "import_odtbrain" in steps
    fps = notes.get("row_contract_fingerprints")
    assert fps, "per-row contract fingerprints must be recorded"
    for row_id in (
        "classical_born_backprop",
        "unet",
        "fno_vanilla",
        "hybrid_resnet",
    ):
        assert row_id in fps


def test_classical_inverse_authorization_requires_passing_odtbrain_check(tmp_path):
    validation_path = tmp_path / "operator_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "checks": [
                    {
                        "name": "odtbrain_inverse_consistency",
                        "status": "fail",
                        "metric": 0.01,
                        "tolerance": 0.65,
                    }
                ]
            }
        )
    )
    manifest = {
        "operator": {
            "validation_artifact": str(validation_path),
        }
    }

    auth = preflight_mod.classical_inverse_authorization(
        manifest, manifest_path=tmp_path / "dataset_manifest.json"
    )

    assert auth["may_run_classical_row"] is False
    assert auth["blocker_reason"] == "odtbrain_inverse_consistency_failed"


def test_classical_fingerprint_changes_with_model_based_solver_config():
    base = dict(
        row_id="classical_born_backprop",
        model="classical_born_backprop",
        training=run_config.CLASSICAL_TRAINING_LABEL,
        input_mode="born_init_image",
        dataset_id=dc.DECISION_SUPPORT_DATASET_NAME,
        operator_pointer="operator_validation.json",
        training_contract=preflight_mod.TrainingContract().as_dict(),
        fixed_sample_ids=[0],
        in_channels=1,
        classical_backend_name="local_adjoint",
        execution_path="model_based_born_inverse",
        solver_version=preflight_mod.MODEL_BASED_INVERSE_VERSION,
    )
    fp_a = preflight_mod.row_contract_fingerprint(
        **base,
        solver_config={"steps": 2, "learning_rate": 0.01},
    )
    fp_b = preflight_mod.row_contract_fingerprint(
        **base,
        solver_config={"steps": 3, "learning_rate": 0.01},
    )

    assert fp_a != fp_b


def test_maybe_resume_rejects_stale_classical_execution_path(tmp_path):
    row = run_config.default_row_roster(
        dataset_id=dc.DECISION_SUPPORT_DATASET_NAME,
        operator_version="op",
    )[0]
    row_dir = tmp_path / "rows" / row.row_id
    row_dir.mkdir(parents=True)
    (row_dir / preflight_mod.ROW_SUMMARY_NAME).write_text(
        json.dumps(
            {
                "row_id": row.row_id,
                "row_status": "blocked",
                "execution_path": "classical_blocked",
                "contract_fingerprint": "same-fingerprint",
            }
        )
    )

    resumed = preflight_mod._maybe_resume_row(
        row=row,
        row_dir=row_dir,
        source_arrays_dir=tmp_path / "arrays",
        fixed_sample_ids=[0],
        expected_fingerprint="same-fingerprint",
        expected_execution_path="model_based_born_inverse",
    )

    assert resumed is None


def test_run_preflight_executes_model_based_classical_inverse(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("BRDT_MODEL_BASED_INVERSE_STEPS", "2")
    monkeypatch.setenv("BRDT_MODEL_BASED_INVERSE_LR", "0.01")
    monkeypatch.setenv("BRDT_MODEL_BASED_INVERSE_TV", "0.0")
    monkeypatch.setenv("BRDT_MODEL_BASED_INVERSE_L2", "0.0")
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "model_based_classical"

    preflight_mod.run_preflight(
        **_live_preflight_kwargs(manifest_path, preflight_root)
    )

    summary = json.loads(
        (
            preflight_root
            / "rows"
            / "classical_born_backprop"
            / preflight_mod.ROW_SUMMARY_NAME
        ).read_text()
    )
    assert summary["row_status"] == "completed"
    assert summary["execution_path"] == "model_based_born_inverse"
    assert summary["paper_label"] == "Model-based Born inverse"
    solver = summary["runtime"]["extras"]["solver"]
    assert solver["solver"] == "adam_direct_q"
    assert solver["config"]["steps"] == 2


# ----------------------------------------------------------------------
# Per-row invocation/runtime provenance + resume protocol
# ----------------------------------------------------------------------
def _live_preflight_kwargs(manifest_path: Path, preflight_root: Path) -> Dict[str, Any]:
    contract = preflight_mod.TrainingContract(
        epochs=1, batch_size=1, learning_rate=2e-4
    )
    return dict(
        manifest_path=manifest_path,
        output_root=preflight_root,
        contract=contract,
        hybrid_label="hybrid_resnet",
        fixed_sample_count=1,
        fixed_sample_seed=0,
        in_channels=1,
        device_choice="cpu",
        dry_run=False,
        parent_argv=["--manifest", str(manifest_path)],
    )


def test_run_preflight_writes_per_row_invocation_artifacts(tmp_path):
    """Each row directory carries invocation.json + invocation.sh + runtime metadata."""
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "live_inv"
    preflight_mod.run_preflight(**_live_preflight_kwargs(manifest_path, preflight_root))
    rows_dir = preflight_root / "rows"
    expected_rows = (
        "classical_born_backprop",
        "unet",
        "fno_vanilla",
        "hybrid_resnet",
    )
    for row_id in expected_rows:
        row_dir = rows_dir / row_id
        invocation_json = row_dir / "invocation.json"
        invocation_sh = row_dir / "invocation.sh"
        assert invocation_json.exists(), f"missing per-row invocation for {row_id}"
        assert invocation_sh.exists(), f"missing per-row invocation.sh for {row_id}"
        payload = json.loads(invocation_json.read_text())
        assert payload["script"] == preflight_mod.SCRIPT_PATH
        assert payload["parsed_args"]["row_id"] == row_id
        runtime_block = payload["extra"].get("runtime") or {}
        assert "device" in runtime_block
        assert "device_name" in runtime_block
        assert "torch" in runtime_block
        assert payload["extra"].get("contract_fingerprint")
        assert payload["extra"].get("backlog_item") == preflight_mod.BACKLOG_ITEM
    summary = json.loads(
        (rows_dir / "classical_born_backprop" / preflight_mod.ROW_SUMMARY_NAME).read_text()
    )
    if summary.get("row_status") == "blocked":
        assert "narrow_fix_attempts" in summary
        assert summary["blocker_reason"] == "odtbrain_unavailable"


def test_run_preflight_resumes_completed_rows_with_matching_fingerprint(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "live_resume"
    kwargs = _live_preflight_kwargs(manifest_path, preflight_root)
    first = preflight_mod.run_preflight(**kwargs)
    assert first.get("resumed_rows") == []
    second = preflight_mod.run_preflight(**kwargs)
    expected_rows = {
        "classical_born_backprop",
        "unet",
        "fno_vanilla",
        "hybrid_resnet",
    }
    assert set(second.get("resumed_rows") or []) == expected_rows


def test_run_preflight_skips_resume_when_contract_changes(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "live_resume_change"
    kwargs = _live_preflight_kwargs(manifest_path, preflight_root)
    preflight_mod.run_preflight(**kwargs)
    # Mutate the training contract → fingerprints diverge → no resume.
    altered = dict(kwargs)
    altered["contract"] = preflight_mod.TrainingContract(
        epochs=2, batch_size=1, learning_rate=2e-4
    )
    second = preflight_mod.run_preflight(**altered)
    assert (second.get("resumed_rows") or []) == []


# ----------------------------------------------------------------------
# Training seed: contract capture, fingerprint propagation, reproducibility
# ----------------------------------------------------------------------
def test_training_contract_records_seed_field():
    """Seed is captured in the contract dict so the manifest and fingerprints carry it."""
    contract = preflight_mod.TrainingContract(
        epochs=1, batch_size=1, learning_rate=2e-4, seed=123
    )
    payload = contract.as_dict()
    assert payload["seed"] == 123


def test_row_contract_fingerprint_changes_with_seed():
    """Different seeds must produce different per-row fingerprints (no false resume)."""

    def _fp(seed: int) -> str:
        contract = preflight_mod.TrainingContract(
            epochs=1, batch_size=1, learning_rate=2e-4, seed=seed
        )
        return preflight_mod.row_contract_fingerprint(
            row_id="unet",
            model="unet",
            training="supervised + Born consistency",
            input_mode="born_init_image",
            dataset_id="brdt128_decision_support_preflight",
            operator_pointer="op.json",
            training_contract=contract.as_dict(),
            fixed_sample_ids=[0],
            in_channels=1,
            classical_backend_name="local_adjoint",
        )

    assert _fp(0) != _fp(1)


def test_dry_run_preflight_manifest_records_training_seed(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "preflight_seed_manifest"
    _run_preflight(
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(preflight_root),
        "--dry-run",
        "--seed",
        "777",
    )
    manifest = json.loads((preflight_root / "preflight_manifest.json").read_text())
    assert manifest["training_contract"]["seed"] == 777


def test_run_preflight_is_reproducible_under_same_seed(tmp_path):
    """Two runs with the same training seed must yield identical neural metrics."""
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    out_a = tmp_path / "seed_run_a"
    out_b = tmp_path / "seed_run_b"
    contract_kwargs = dict(
        manifest_path=manifest_path,
        contract=preflight_mod.TrainingContract(
            epochs=1, batch_size=1, learning_rate=2e-4, seed=2026
        ),
        hybrid_label="hybrid_resnet",
        fixed_sample_count=1,
        fixed_sample_seed=0,
        in_channels=1,
        device_choice="cpu",
        dry_run=False,
        parent_argv=["--manifest", str(manifest_path)],
    )
    preflight_mod.run_preflight(output_root=out_a, **contract_kwargs)
    preflight_mod.run_preflight(output_root=out_b, **contract_kwargs)

    metrics_a = json.loads((out_a / "metrics.json").read_text())
    metrics_b = json.loads((out_b / "metrics.json").read_text())
    rows_a = {row["row_id"]: row for row in metrics_a["rows"]}
    rows_b = {row["row_id"]: row for row in metrics_b["rows"]}
    for neural_row in ("unet", "fno_vanilla", "hybrid_resnet"):
        assert rows_a[neural_row]["row_status"] == "completed"
        assert rows_b[neural_row]["row_status"] == "completed"
        for bucket in ("image", "measurement", "supporting"):
            for key, val_a in rows_a[neural_row][bucket].items():
                val_b = rows_b[neural_row][bucket][key]
                assert val_a == pytest.approx(val_b, rel=0, abs=0), (
                    f"{neural_row}.{bucket}.{key} drifted between seeded reruns: "
                    f"{val_a} vs {val_b}"
                )


# ----------------------------------------------------------------------
# Physics-only objective ablation: objective preset, row selection,
# baseline lineage, output dynamic-range diagnostics.
# ----------------------------------------------------------------------
def test_resolve_objective_preset_relative_physics_only_weights():
    weights, training_label = run_config.resolve_objective_preset(
        "relative_physics_only"
    )
    assert weights.as_dict() == {
        "image": 0.0,
        "physics": 0.0,
        "relative_physics": 1.0,
        "tv": 0.0,
        "positivity": 0.0,
    }
    assert training_label == run_config.RELATIVE_PHYSICS_ONLY_TRAINING_LABEL


def test_resolve_objective_preset_supervised_plus_born_default_weights():
    weights, training_label = run_config.resolve_objective_preset(
        "supervised_plus_born"
    )
    assert weights.as_dict() == run_config.LossWeights().as_dict()
    assert training_label == run_config.DEFAULT_TRAINING_LABEL


def test_resolve_objective_preset_unknown_rejected():
    with pytest.raises(ValueError, match="unknown objective preset"):
        run_config.resolve_objective_preset("paper_grade")


def test_resolve_row_roster_selected_rows_excludes_classical(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    roster = preflight_mod.resolve_row_roster(
        manifest_path=manifest_path,
        hybrid_label="hybrid_resnet",
        selected_row_ids=["unet", "fno_vanilla", "hybrid_resnet"],
    )
    assert [r.row_id for r in roster] == ["unet", "fno_vanilla", "hybrid_resnet"]
    assert all(r.row_id != "classical_born_backprop" for r in roster)


def test_resolve_row_roster_unknown_selected_row_rejected(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    with pytest.raises(ValueError, match="unknown selected row_ids"):
        preflight_mod.resolve_row_roster(
            manifest_path=manifest_path,
            hybrid_label="hybrid_resnet",
            selected_row_ids=["unet", "no_such_row"],
        )


def test_resolve_row_roster_relative_physics_only_label(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    roster = preflight_mod.resolve_row_roster(
        manifest_path=manifest_path,
        hybrid_label="hybrid_resnet",
        neural_training_label=run_config.RELATIVE_PHYSICS_ONLY_TRAINING_LABEL,
        selected_row_ids=["unet", "fno_vanilla", "hybrid_resnet"],
    )
    for row in roster:
        assert row.training == run_config.RELATIVE_PHYSICS_ONLY_TRAINING_LABEL


def test_run_preflight_default_path_keeps_supervised_plus_born_contract(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "default_preset"
    _run_preflight(
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(preflight_root),
        "--dry-run",
    )
    manifest = json.loads((preflight_root / "preflight_manifest.json").read_text())
    weights = manifest["training_contract"]["loss_weights"]
    assert weights == run_config.LossWeights().as_dict()
    # The default supervised-plus-Born path must not be silently turned
    # into an ablation manifest by the new ablation surface; the runner
    # must omit the ablation-only notes when the default preset is used.
    notes = manifest.get("notes") or {}
    assert "objective_preset" not in notes
    assert "selected_row_ids" not in notes
    assert "baseline_lineage" not in notes
    # The default path must continue to advertise the baseline claim
    # boundary, both in the manifest and in the dry-run metric schema.
    assert manifest["claim_boundary"] == "decision_support_preflight_only"
    schema = json.loads((preflight_root / "metric_schema.json").read_text())
    assert schema["claim_boundary"] == "decision_support_preflight_only"


def test_run_preflight_relative_physics_only_writes_consistent_claim_boundary(tmp_path):
    """preflight_manifest, metric_schema, and metrics.json must all agree.

    Regression for an implementation-review finding where the ablation
    bundle's machine-consumed metric artifacts kept advertising the
    baseline claim boundary while the manifest already used the
    ablation boundary.
    """
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "physics_only_consistency"
    preflight_mod.run_preflight(
        **_physics_only_kwargs(manifest_path, preflight_root)
    )
    manifest = json.loads(
        (preflight_root / preflight_mod.PREFLIGHT_MANIFEST_NAME).read_text()
    )
    schema = json.loads((preflight_root / "metric_schema.json").read_text())
    metrics = json.loads((preflight_root / "metrics.json").read_text())
    assert manifest["claim_boundary"] == "decision_support_append_only"
    assert schema["claim_boundary"] == "decision_support_append_only"
    assert metrics["claim_boundary"] == "decision_support_append_only"


def test_run_preflight_ablation_comparison_fails_fast_on_corrupt_baseline(tmp_path):
    """Required comparison artifacts must not be silently downgraded.

    Regression for an implementation-review finding where comparison
    emission was wrapped in a blanket try/except that masked failures
    as ``notes.comparison_emission_error`` even when baseline lineage
    was present.
    """
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    baseline_root = tmp_path / "corrupt_baseline_bundle"
    baseline_root.mkdir()
    (baseline_root / preflight_mod.PREFLIGHT_MANIFEST_NAME).write_text(
        json.dumps({"backlog_item": "baseline"})
    )
    # Write a metrics.json that exists (so baseline_present is True) but
    # is not valid JSON, forcing comparison emission to raise.
    (baseline_root / "metrics.json").write_text("not-json{")

    ablation_root = tmp_path / "ablation_corrupt_bundle"
    kwargs = dict(_physics_only_kwargs(manifest_path, ablation_root))
    kwargs["baseline_root"] = baseline_root
    with pytest.raises(json.JSONDecodeError):
        preflight_mod.run_preflight(**kwargs)


def test_run_preflight_relative_physics_only_writes_zero_supervised_weights(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "physics_only_dry"
    _run_preflight(
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(preflight_root),
        "--dry-run",
        "--objective-preset",
        "relative_physics_only",
        "--rows",
        "unet,fno_vanilla,hybrid_resnet",
    )
    manifest = json.loads((preflight_root / "preflight_manifest.json").read_text())
    weights = manifest["training_contract"]["loss_weights"]
    assert weights == {
        "image": 0.0,
        "physics": 0.0,
        "relative_physics": 1.0,
        "tv": 0.0,
        "positivity": 0.0,
    }
    notes = manifest.get("notes") or {}
    assert notes.get("objective_preset") == "relative_physics_only"
    assert notes.get("selected_row_ids") == ["unet", "fno_vanilla", "hybrid_resnet"]
    # Append-only ablation must not relabel itself as the supervised+Born
    # baseline backlog item.
    assert manifest["backlog_item"] == preflight_mod.PHYSICS_ONLY_BACKLOG_ITEM
    assert manifest["claim_boundary"] == "decision_support_append_only"
    # Selected rows must exclude the classical row in the manifest.
    row_ids = [r["row_id"] for r in manifest["rows"]]
    assert row_ids == ["unet", "fno_vanilla", "hybrid_resnet"]


def test_run_preflight_baseline_root_lineage_recorded(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    # Stand up a synthetic baseline root containing the two pointers used
    # by the lineage block. The actual contents are irrelevant for the
    # dry-run lineage check; only presence and path strings matter.
    baseline_root = tmp_path / "synthetic_baseline"
    baseline_root.mkdir()
    (baseline_root / preflight_mod.PREFLIGHT_MANIFEST_NAME).write_text(
        json.dumps({"backlog_item": "baseline"})
    )
    (baseline_root / "metrics.json").write_text(
        json.dumps({"schema_version": "x", "rows": []})
    )
    preflight_root = tmp_path / "lineage_dry"
    _run_preflight(
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(preflight_root),
        "--dry-run",
        "--objective-preset",
        "relative_physics_only",
        "--rows",
        "unet,fno_vanilla,hybrid_resnet",
        "--baseline-root",
        str(baseline_root),
    )
    manifest = json.loads((preflight_root / "preflight_manifest.json").read_text())
    lineage = (manifest.get("notes") or {}).get("baseline_lineage") or {}
    assert lineage["baseline_root"] == str(baseline_root.resolve())
    assert lineage["baseline_present"] is True
    assert lineage["baseline_preflight_manifest"].endswith(
        preflight_mod.PREFLIGHT_MANIFEST_NAME
    )
    assert lineage["baseline_metrics_json"].endswith("metrics.json")


def _physics_only_kwargs(manifest_path: Path, preflight_root: Path) -> Dict[str, Any]:
    contract = preflight_mod.TrainingContract(
        epochs=1, batch_size=1, learning_rate=2e-4
    )
    return dict(
        manifest_path=manifest_path,
        output_root=preflight_root,
        contract=contract,
        hybrid_label="hybrid_resnet",
        fixed_sample_count=1,
        fixed_sample_seed=0,
        in_channels=1,
        device_choice="cpu",
        dry_run=False,
        parent_argv=["--manifest", str(manifest_path)],
        objective_preset="relative_physics_only",
        selected_row_ids=["unet", "fno_vanilla", "hybrid_resnet"],
    )


def test_run_preflight_records_output_dynamic_range_diagnostics(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "physics_only_live"
    preflight_mod.run_preflight(**_physics_only_kwargs(manifest_path, preflight_root))
    rows_dir = preflight_root / "rows"
    for row_id in ("unet", "fno_vanilla", "hybrid_resnet"):
        summary = json.loads((rows_dir / row_id / preflight_mod.ROW_SUMMARY_NAME).read_text())
        odr = summary.get("output_dynamic_range")
        assert odr is not None, f"missing output_dynamic_range for {row_id}"
        for key in (
            "physical_q_min",
            "physical_q_max",
            "physical_q_mean",
            "physical_q_std",
            "physical_q_ptp",
        ):
            assert key in odr
        # Same diagnostic must also surface in the runtime extras under the
        # aggregated metrics, so external readers do not need to walk per-row
        # summaries to detect collapse.
        runtime = summary["runtime"]
        assert runtime["extras"]["output_dynamic_range"] == odr


def test_run_preflight_emits_comparison_when_baseline_root_supplied(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    # First, generate a tiny supervised+Born baseline bundle.
    baseline_root = tmp_path / "baseline_bundle"
    baseline_kwargs = dict(
        manifest_path=manifest_path,
        output_root=baseline_root,
        contract=preflight_mod.TrainingContract(
            epochs=1, batch_size=1, learning_rate=2e-4
        ),
        hybrid_label="hybrid_resnet",
        fixed_sample_count=1,
        fixed_sample_seed=0,
        in_channels=1,
        device_choice="cpu",
        dry_run=False,
        parent_argv=["--manifest", str(manifest_path)],
        objective_preset="supervised_plus_born",
        selected_row_ids=["unet", "fno_vanilla", "hybrid_resnet"],
    )
    preflight_mod.run_preflight(**baseline_kwargs)

    # Then run the physics-only ablation with the baseline-root pointer.
    ablation_root = tmp_path / "ablation_bundle"
    ablation_kwargs = dict(_physics_only_kwargs(manifest_path, ablation_root))
    ablation_kwargs["baseline_root"] = baseline_root
    preflight_mod.run_preflight(**ablation_kwargs)

    comparison_json = ablation_root / "comparison_to_supervised_plus_born.json"
    comparison_csv = ablation_root / "comparison_to_supervised_plus_born.csv"
    assert comparison_json.exists()
    assert comparison_csv.exists()
    payload = json.loads(comparison_json.read_text())
    assert payload["baseline"]["objective_preset"] == "supervised_plus_born"
    assert payload["ablation"]["objective_preset"] == "relative_physics_only"
    row_ids = {row["row_id"] for row in payload["rows"]}
    assert row_ids == {"unet", "fno_vanilla", "hybrid_resnet"}
    # The ablation manifest must not silently overwrite the baseline bundle.
    assert (baseline_root / "metrics.json").exists()
    baseline_manifest = json.loads(
        (baseline_root / preflight_mod.PREFLIGHT_MANIFEST_NAME).read_text()
    )
    assert baseline_manifest["backlog_item"] == preflight_mod.BACKLOG_ITEM


def test_comparison_populates_fixed_sample_dynamic_range_for_baseline_without_eval_field(
    tmp_path,
):
    """Backfill the baseline collapse-detection signal from saved arrays.

    Older baselines predate the eval-split ``output_dynamic_range`` field, so
    the eval block is ``null`` on the baseline side. The comparison must still
    surface a populated like-for-like ``fixed_sample`` block computed from each
    bundle's ``figures/source_arrays/sample_*_<row_id>_q_pred.npy`` files,
    plus the per-component ``final_loss_breakdown`` deltas, so callers can
    diagnose collapse without needing to re-run the baseline.
    """
    from scripts.studies.born_rytov_dt import comparison as comp_mod

    baseline_root = tmp_path / "baseline_root"
    ablation_root = tmp_path / "ablation_root"
    (baseline_root / "figures" / "source_arrays").mkdir(parents=True)
    (ablation_root / "figures" / "source_arrays").mkdir(parents=True)

    rng = np.random.default_rng(0)
    # Baseline saved fixed-sample predictions: wide dynamic range.
    for sid in (1, 2):
        np.save(
            baseline_root / "figures" / "source_arrays"
            / f"sample_{sid:04d}_unet_q_pred.npy",
            rng.normal(0.0, 0.05, size=(8, 8)).astype(np.float32),
        )
    # Ablation saved fixed-sample predictions: narrower (collapse-like).
    for sid in (1, 2):
        np.save(
            ablation_root / "figures" / "source_arrays"
            / f"sample_{sid:04d}_unet_q_pred.npy",
            np.full((8, 8), 1e-4, dtype=np.float32),
        )

    baseline_metrics = {
        "rows": [
            {
                "row_id": "unet",
                "paper_label": "U-Net",
                "architecture": "unet",
                "image": {
                    "image_mae_phys": 0.001,
                    "image_rmse_phys": 0.003,
                    "image_relative_l2_phys": 0.7,
                },
                "measurement": {
                    "meas_mae": 0.002,
                    "meas_rmse": 0.004,
                    "meas_relative_l2": 0.5,
                },
                "supporting": {"psnr_phys": 25.0, "ssim_phys": 0.7},
                "runtime": {
                    "parameter_count": 18465,
                    "wall_time_train_s": 60.0,
                    "wall_time_eval_s": 0.5,
                    "epochs": 1,
                    "batch_size": 1,
                    "learning_rate": 2e-4,
                    "extras": {
                        "final_loss_breakdown": {
                            "image": 0.4,
                            "physics": 0.003,
                            "relative_physics": 0.7,
                            "tv": 1e-4,
                            "positivity": 1e-9,
                            "total": 0.5,
                        },
                    },
                },
            }
        ],
    }
    ablation_metrics = {
        "rows": [
            {
                "row_id": "unet",
                "paper_label": "U-Net",
                "architecture": "unet",
                "image": {
                    "image_mae_phys": 0.002,
                    "image_rmse_phys": 0.004,
                    "image_relative_l2_phys": 0.85,
                },
                "measurement": {
                    "meas_mae": 0.0025,
                    "meas_rmse": 0.005,
                    "meas_relative_l2": 0.6,
                },
                "supporting": {"psnr_phys": 21.0, "ssim_phys": 0.6},
                "runtime": {
                    "parameter_count": 18465,
                    "wall_time_train_s": 70.0,
                    "wall_time_eval_s": 0.6,
                    "epochs": 1,
                    "batch_size": 1,
                    "learning_rate": 2e-4,
                    "extras": {
                        "final_loss_breakdown": {
                            "image": 0.5,
                            "physics": 0.003,
                            "relative_physics": 0.6,
                            "tv": 2e-4,
                            "positivity": 1e-7,
                            "total": 0.6,
                        },
                        "output_dynamic_range": {
                            "physical_q_min": -0.001,
                            "physical_q_max": 0.001,
                            "physical_q_mean": 0.0,
                            "physical_q_std": 0.0005,
                            "physical_q_ptp": 0.002,
                        },
                    },
                },
            }
        ],
    }

    payload = comp_mod.build_comparison(
        baseline_metrics=baseline_metrics,
        ablation_metrics=ablation_metrics,
        selected_row_ids=["unet"],
        baseline_root=str(baseline_root),
        ablation_root=str(ablation_root),
        baseline_root_path=baseline_root,
        ablation_root_path=ablation_root,
    )

    row = payload["rows"][0]
    odr = row["output_dynamic_range"]
    # Eval-block: baseline is null because baseline metrics predate the field.
    assert odr["baseline"] is None
    assert odr["ablation"] is not None
    # Fixed-sample block: BOTH sides populated, deltas computed.
    fixed = odr["fixed_sample"]
    assert fixed["baseline"] is not None
    assert fixed["ablation"] is not None
    for key in (
        "physical_q_min",
        "physical_q_max",
        "physical_q_mean",
        "physical_q_std",
        "physical_q_ptp",
    ):
        assert key in fixed["baseline"]
        assert key in fixed["ablation"]
        assert fixed["delta"][key] == pytest.approx(
            fixed["ablation"][key] - fixed["baseline"][key]
        )
    # The ablation collapsed (constant predictions), so its ptp is much
    # smaller than the baseline — delta must be negative.
    assert fixed["delta"]["physical_q_ptp"] < 0
    assert fixed["baseline"]["n_samples"] == 2
    assert fixed["ablation"]["n_samples"] == 2

    # final_loss_breakdown now carries per-component deltas.
    flb = row["final_loss_breakdown"]
    assert flb["delta"]["total"] == pytest.approx(0.6 - 0.5)
    assert flb["delta"]["relative_physics"] == pytest.approx(0.6 - 0.7)

    # CSV must surface the loss-breakdown and dynamic-range columns that
    # were previously dropped.
    csv_path = tmp_path / "out.csv"
    comp_mod.write_comparison_csv(payload, csv_path)
    header_line = csv_path.read_text().splitlines()[0]
    for expected in (
        "final_loss_total_baseline",
        "final_loss_total_ablation",
        "final_loss_total_delta",
        "output_dynamic_range_eval_physical_q_ptp_ablation",
        "output_dynamic_range_fixed_physical_q_ptp_baseline",
        "output_dynamic_range_fixed_physical_q_ptp_ablation",
        "output_dynamic_range_fixed_physical_q_ptp_delta",
    ):
        assert expected in header_line, f"missing column: {expected}"


# ----------------------------------------------------------------------
# Invocation provenance: backlog identity + parser-valid row replay
# ----------------------------------------------------------------------
def test_row_replay_argv_replaces_existing_rows_selector():
    """When the parent argv carries ``--rows ...``, the per-row replay must
    narrow the selector to a single row rather than appending an unparsable
    flag."""
    parent = [
        "--manifest", "m.json",
        "--output-root", "out",
        "--objective-preset", "relative_physics_only",
        "--rows", "unet,fno_vanilla,hybrid_resnet",
    ]
    out = preflight_mod._row_replay_argv(parent, "fno_vanilla")
    # Single-row form is present.
    assert out.count("--rows") == 1
    idx = out.index("--rows")
    assert out[idx + 1] == "fno_vanilla"
    # The original multi-row value is no longer in the argv.
    assert "unet,fno_vanilla,hybrid_resnet" not in out
    # The invalid `--row-id` flag from the prior implementation is absent.
    assert "--row-id" not in out
    # The replay argv must round-trip through the live parser.
    parser = preflight_mod._build_parser()
    parsed = parser.parse_args(out)
    assert parsed.rows == "fno_vanilla"


def test_row_replay_argv_handles_equals_form_and_missing_selector():
    parser = preflight_mod._build_parser()
    # `--rows=...` form is rewritten in place.
    eq_form = ["--manifest", "m.json", "--output-root", "out", "--rows=unet,fno_vanilla"]
    eq_out = preflight_mod._row_replay_argv(eq_form, "unet")
    assert "--rows=unet" in eq_out
    parser.parse_args(eq_out)
    # Parents that did not carry `--rows` get the single-row selector appended.
    no_sel = ["--manifest", "m.json", "--output-root", "out"]
    appended = preflight_mod._row_replay_argv(no_sel, "hybrid_resnet")
    assert appended[-2:] == ["--rows", "hybrid_resnet"]
    parser.parse_args(appended)


def test_run_preflight_ablation_invocation_artifacts_carry_physics_only_backlog_item(
    tmp_path,
):
    """Top-level + per-row invocation provenance must record the physics-only
    backlog item when ``--objective-preset relative_physics_only`` is used,
    and per-row replay argv must only contain parser-valid flags."""
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "physics_only_invocation"
    _run_preflight(
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(preflight_root),
        "--dry-run",
        "--objective-preset",
        "relative_physics_only",
        "--rows",
        "unet,fno_vanilla,hybrid_resnet",
    )
    # Top-level invocation.json must record the actual backlog item that
    # owns this artifact root, not the supervised+Born baseline item.
    top = json.loads((preflight_root / "invocation.json").read_text())
    assert (
        top["extra"]["backlog_item"]
        == preflight_mod.PHYSICS_ONLY_BACKLOG_ITEM
    )

    # Run the full live ablation to also exercise the per-row provenance.
    # The parent argv mirrors what the CLI would pass so the per-row replay
    # argv is parser-valid in the same way an end-to-end run produces.
    live_root = tmp_path / "physics_only_invocation_live"
    live_kwargs = dict(_physics_only_kwargs(manifest_path, live_root))
    live_parent_argv = [
        "--manifest", str(manifest_path),
        "--output-root", str(live_root),
        "--objective-preset", "relative_physics_only",
        "--rows", "unet,fno_vanilla,hybrid_resnet",
    ]
    live_kwargs["parent_argv"] = live_parent_argv
    preflight_mod.run_preflight(**live_kwargs)
    parser = preflight_mod._build_parser()
    for row_id in ("unet", "fno_vanilla", "hybrid_resnet"):
        per_row = json.loads(
            (live_root / "rows" / row_id / "invocation.json").read_text()
        )
        # Backlog identity matches the ablation root.
        assert (
            per_row["extra"]["backlog_item"]
            == preflight_mod.PHYSICS_ONLY_BACKLOG_ITEM
        )
        # Replay argv: only parser-valid flags, scoped to this row.
        argv = per_row["argv"]
        assert "--row-id" not in argv
        parsed = parser.parse_args(argv)
        assert parsed.rows == row_id


# ----------------------------------------------------------------------
# BRDT FFNO row extension (2026-05-04-brdt-ffno-row-extension)
# ----------------------------------------------------------------------
def _make_synthetic_baseline_bundle(tmp_path: Path) -> Path:
    """Stand up a baseline-shaped bundle for FFNO-extension validator tests.

    Mirrors the fields the extension validator inspects without running a
    real four-row preflight. The synthetic bundle is structurally valid
    (passes ``validate_baseline_bundle``) but carries placeholder metrics
    so combined-bundle helpers can be exercised cheaply.
    """
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle

    baseline_root = tmp_path / "synthetic_baseline"
    (baseline_root / "figures" / "source_arrays").mkdir(parents=True)
    manifest = {
        "schema_version": "brdt_preflight_v1",
        "backlog_item": ext_bundle.BASELINE_BACKLOG_ITEM,
        "claim_boundary": "decision_support_preflight_only",
        "next_backlog_item": "downstream",
        "output_root": str(baseline_root),
        "dataset": {
            "dataset_id": dc.DECISION_SUPPORT_DATASET_NAME,
            "tier": "decision_support",
            "manifest_path": str(baseline_root / "dataset_manifest.json"),
            "split_counts": {"train": 2048, "val": 256, "test": 256},
            "normalization": {
                "q_max_train": 0.028,
                "q_mean_train": 0.001,
                "q_min_train": -0.027,
                "q_std_train": 0.005,
            },
            "claim_boundary": "decision_support",
        },
        "operator": {
            "validation_artifact": "/tmp/operator_validation.json",
            "validation_report": "docs/validation.md",
            "geometry": {
                "grid_size": 128,
                "detector_size": 128,
                "angle_count": 64,
                "wavelength_px": 8.0,
                "medium_ri": 1.333,
                "mode": "born",
                "normalize": "unitary_fft",
            },
        },
        "input_contract": {
            "input_mode": "born_init_image",
            "in_channels": 1,
        },
        "training_contract": {
            "epochs": 20,
            "batch_size": 16,
            "learning_rate": 2e-4,
            "optimizer": "Adam",
            "loss_weights": {
                "image": 1.0,
                "physics": 0.1,
                "relative_physics": 0.1,
                "tv": 1e-5,
                "positivity": 1e-4,
            },
            "seed": 42,
        },
        "fixed_sample_seed": 17,
        "fixed_sample_ids": [145, 83, 255, 126],
        "rows": [
            {
                "row_id": "classical_born_backprop",
                "model": "classical_born_backprop",
                "training": "none",
                "input_mode": "born_init_image",
                "dataset_id": dc.DECISION_SUPPORT_DATASET_NAME,
                "operator_version": "/tmp/operator_validation.json",
                "row_status": "completed",
                "paper_label": "Model-based Born inverse",
            },
            {
                "row_id": "unet",
                "model": "unet",
                "training": "supervised + Born consistency",
                "input_mode": "born_init_image",
                "dataset_id": dc.DECISION_SUPPORT_DATASET_NAME,
                "operator_version": "/tmp/operator_validation.json",
                "row_status": "completed",
                "paper_label": "U-Net",
            },
            {
                "row_id": "fno_vanilla",
                "model": "fno_vanilla",
                "training": "supervised + Born consistency",
                "input_mode": "born_init_image",
                "dataset_id": dc.DECISION_SUPPORT_DATASET_NAME,
                "operator_version": "/tmp/operator_validation.json",
                "row_status": "completed",
                "paper_label": "FNO vanilla",
            },
            {
                "row_id": "hybrid_resnet",
                "model": "hybrid_resnet",
                "training": "supervised + Born consistency",
                "input_mode": "born_init_image",
                "dataset_id": dc.DECISION_SUPPORT_DATASET_NAME,
                "operator_version": "/tmp/operator_validation.json",
                "row_status": "completed",
                "paper_label": "Hybrid ResNet",
            },
        ],
    }
    metric_rows = []
    for row in manifest["rows"]:
        if row["row_status"] == "blocked":
            metric_rows.append(
                {
                    "row_id": row["row_id"],
                    "paper_label": row["paper_label"],
                    "architecture": row["model"],
                    "row_status": "blocked",
                    "blocker_reason": "odtbrain_unavailable",
                    "image": {},
                    "measurement": {},
                    "supporting": {},
                    "runtime": {
                        "device": "cpu",
                        "device_name": "cpu",
                        "epochs": 0,
                        "batch_size": 16,
                        "learning_rate": 2e-4,
                        "parameter_count": 0,
                        "wall_time_train_s": 0.0,
                        "wall_time_eval_s": 0.0,
                        "row_status": "blocked",
                    },
                }
            )
            continue
        metric_rows.append(
            {
                "row_id": row["row_id"],
                "paper_label": row["paper_label"],
                "architecture": row["model"],
                "row_status": "completed",
                "image": {
                    "image_mae_phys": 0.001,
                    "image_rmse_phys": 0.002,
                    "image_relative_l2_phys": 0.5,
                },
                "measurement": {
                    "meas_mae": 0.001,
                    "meas_rmse": 0.002,
                    "meas_relative_l2": 0.3,
                },
                "supporting": {"psnr_phys": 25.0, "ssim_phys": 0.7},
                "runtime": {
                    "device": "cpu",
                    "device_name": "cpu",
                    "epochs": 20,
                    "batch_size": 16,
                    "learning_rate": 2e-4,
                    "parameter_count": 100,
                    "wall_time_train_s": 1.0,
                    "wall_time_eval_s": 0.1,
                    "row_status": "completed",
                },
            }
        )
    metrics = {
        "schema_version": "brdt_preflight_metrics_v1",
        "claim_boundary": "decision_support_preflight_only",
        "rows": metric_rows,
    }
    (baseline_root / "preflight_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (baseline_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return baseline_root


def test_extension_bundle_validate_baseline_bundle_rejects_missing_metrics(tmp_path):
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle

    baseline_root = tmp_path / "incomplete_baseline"
    baseline_root.mkdir()
    (baseline_root / "preflight_manifest.json").write_text(
        json.dumps({"backlog_item": ext_bundle.BASELINE_BACKLOG_ITEM})
    )
    with pytest.raises(ext_bundle.BaselineContractMismatchError, match="metrics.json"):
        ext_bundle.validate_baseline_bundle(baseline_root)


def test_extension_bundle_validate_baseline_bundle_rejects_wrong_backlog_item(tmp_path):
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle

    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    manifest_path = baseline_root / "preflight_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["backlog_item"] = "some-other-backlog"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(
        ext_bundle.BaselineContractMismatchError, match="backlog_item"
    ):
        ext_bundle.validate_baseline_bundle(baseline_root)


def test_extension_bundle_validate_baseline_bundle_consumes_rows_as_they_exist(
    tmp_path,
):
    """Plan-binding regression: the FFNO extension consumes the baseline
    ``preflight_manifest.json`` and ``metrics.json`` as they exist on disk
    (`docs/plans/.../execution_plan.md` Lines 25-39, 78). The validator must
    therefore accept whatever row roster and per-row status the baseline
    bundle currently records — divergence from any external summary
    authority is not the FFNO extension's enforcement surface.
    """
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle
    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    # Mutate the baseline rows: drop one row, demote one neural row, promote
    # the classical row. The validator must still accept the bundle because
    # row contents are not part of the FFNO extension's contract surface.
    metrics_path = baseline_root / "metrics.json"
    metrics_payload = json.loads(metrics_path.read_text())
    metrics_payload["rows"] = [
        row for row in metrics_payload["rows"] if row["row_id"] != "fno_vanilla"
    ]
    for row in metrics_payload["rows"]:
        if row["row_id"] == "classical_born_backprop":
            row["row_status"] = "completed"
            row["paper_label"] = "Model-based Born inverse"
        if row["row_id"] == "hybrid_resnet":
            row["row_status"] = "blocked"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))
    manifest_path = baseline_root / "preflight_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text())
    manifest_payload["rows"] = [
        row for row in manifest_payload["rows"] if row["row_id"] != "fno_vanilla"
    ]
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True))
    # Should not raise: structural fields are still intact.
    parsed = ext_bundle.validate_baseline_bundle(baseline_root)
    assert [r["row_id"] for r in parsed["rows"]] == [
        "classical_born_backprop",
        "unet",
        "hybrid_resnet",
    ]


def test_extension_bundle_assert_extension_inherits_rejects_dataset_drift(tmp_path):
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle

    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    baseline_manifest = ext_bundle.validate_baseline_bundle(baseline_root)
    base_training = baseline_manifest["training_contract"]
    base_pointer = baseline_manifest["operator"]["validation_artifact"]
    with pytest.raises(
        ext_bundle.BaselineContractMismatchError, match="dataset_id"
    ):
        ext_bundle.assert_extension_inherits_baseline(
            baseline_manifest=baseline_manifest,
            extension_dataset_id="some-other-dataset",
            extension_input_mode="born_init_image",
            extension_in_channels=1,
            extension_training_contract=base_training,
            extension_fixed_sample_ids=baseline_manifest["fixed_sample_ids"],
            extension_operator_pointer=base_pointer,
            extension_claim_boundary=ext_bundle.APPEND_ONLY_CLAIM_BOUNDARY,
        )


def test_extension_bundle_assert_extension_inherits_rejects_claim_boundary_drift(
    tmp_path,
):
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle

    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    baseline_manifest = ext_bundle.validate_baseline_bundle(baseline_root)
    base_training = baseline_manifest["training_contract"]
    base_pointer = baseline_manifest["operator"]["validation_artifact"]
    with pytest.raises(
        ext_bundle.BaselineContractMismatchError, match="claim_boundary"
    ):
        ext_bundle.assert_extension_inherits_baseline(
            baseline_manifest=baseline_manifest,
            extension_dataset_id=baseline_manifest["dataset"]["dataset_id"],
            extension_input_mode="born_init_image",
            extension_in_channels=1,
            extension_training_contract=base_training,
            extension_fixed_sample_ids=baseline_manifest["fixed_sample_ids"],
            extension_operator_pointer=base_pointer,
            extension_claim_boundary="decision_support_preflight_only",
        )


def test_extension_bundle_assert_extension_inherits_rejects_split_counts_drift(
    tmp_path,
):
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle

    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    baseline_manifest = ext_bundle.validate_baseline_bundle(baseline_root)
    base_training = baseline_manifest["training_contract"]
    base_pointer = baseline_manifest["operator"]["validation_artifact"]
    with pytest.raises(
        ext_bundle.BaselineContractMismatchError, match="split_counts.train"
    ):
        ext_bundle.assert_extension_inherits_baseline(
            baseline_manifest=baseline_manifest,
            extension_dataset_id=baseline_manifest["dataset"]["dataset_id"],
            extension_input_mode="born_init_image",
            extension_in_channels=1,
            extension_training_contract=base_training,
            extension_fixed_sample_ids=baseline_manifest["fixed_sample_ids"],
            extension_operator_pointer=base_pointer,
            extension_claim_boundary=ext_bundle.APPEND_ONLY_CLAIM_BOUNDARY,
            extension_split_counts={"train": 4096, "val": 256, "test": 256},
        )


def test_extension_bundle_assert_extension_inherits_rejects_normalization_drift(
    tmp_path,
):
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle

    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    baseline_manifest = ext_bundle.validate_baseline_bundle(baseline_root)
    base_training = baseline_manifest["training_contract"]
    base_pointer = baseline_manifest["operator"]["validation_artifact"]
    drifted_norm = dict(baseline_manifest["dataset"]["normalization"])
    drifted_norm["q_max_train"] = float(drifted_norm["q_max_train"]) + 1.0
    with pytest.raises(
        ext_bundle.BaselineContractMismatchError,
        match="normalization.q_max_train",
    ):
        ext_bundle.assert_extension_inherits_baseline(
            baseline_manifest=baseline_manifest,
            extension_dataset_id=baseline_manifest["dataset"]["dataset_id"],
            extension_input_mode="born_init_image",
            extension_in_channels=1,
            extension_training_contract=base_training,
            extension_fixed_sample_ids=baseline_manifest["fixed_sample_ids"],
            extension_operator_pointer=base_pointer,
            extension_claim_boundary=ext_bundle.APPEND_ONLY_CLAIM_BOUNDARY,
            extension_normalization=drifted_norm,
        )


def test_extension_bundle_assert_extension_inherits_rejects_operator_geometry_drift(
    tmp_path,
):
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle

    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    baseline_manifest = ext_bundle.validate_baseline_bundle(baseline_root)
    base_training = baseline_manifest["training_contract"]
    base_pointer = baseline_manifest["operator"]["validation_artifact"]
    drifted_geom = dict(baseline_manifest["operator"]["geometry"])
    drifted_geom["angle_count"] = int(drifted_geom["angle_count"]) + 1
    with pytest.raises(
        ext_bundle.BaselineContractMismatchError,
        match="operator.geometry.angle_count",
    ):
        ext_bundle.assert_extension_inherits_baseline(
            baseline_manifest=baseline_manifest,
            extension_dataset_id=baseline_manifest["dataset"]["dataset_id"],
            extension_input_mode="born_init_image",
            extension_in_channels=1,
            extension_training_contract=base_training,
            extension_fixed_sample_ids=baseline_manifest["fixed_sample_ids"],
            extension_operator_pointer=base_pointer,
            extension_claim_boundary=ext_bundle.APPEND_ONLY_CLAIM_BOUNDARY,
            extension_operator_geometry=drifted_geom,
        )


def test_extension_bundle_assert_extension_inherits_rejects_fixed_sample_seed_drift(
    tmp_path,
):
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle

    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    baseline_manifest = ext_bundle.validate_baseline_bundle(baseline_root)
    base_training = baseline_manifest["training_contract"]
    base_pointer = baseline_manifest["operator"]["validation_artifact"]
    base_seed = int(baseline_manifest["fixed_sample_seed"])
    with pytest.raises(
        ext_bundle.BaselineContractMismatchError,
        match="fixed_sample_seed",
    ):
        ext_bundle.assert_extension_inherits_baseline(
            baseline_manifest=baseline_manifest,
            extension_dataset_id=baseline_manifest["dataset"]["dataset_id"],
            extension_input_mode="born_init_image",
            extension_in_channels=1,
            extension_training_contract=base_training,
            extension_fixed_sample_ids=baseline_manifest["fixed_sample_ids"],
            extension_operator_pointer=base_pointer,
            extension_claim_boundary=ext_bundle.APPEND_ONLY_CLAIM_BOUNDARY,
            extension_fixed_sample_seed=base_seed + 1,
        )


def test_extension_bundle_classical_backend_authority_handoff_records_both_surfaces(
    tmp_path,
):
    """Authority handoff names both baseline surfaces and the extension's choice.

    The plan binds the FFNO extension to the baseline JSON files as they
    exist; the baseline's manifest-level ``input_contract.classical_backend``
    diverges from its per-row neural ``invocation.json`` value. The handoff
    block records both surfaces explicitly so reviewers can see the
    extension's choice is not silent.
    """
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle

    baseline_manifest = {
        "input_contract": {
            "classical_backend": {
                "name": "odtbrain",
                "claim_boundary": "external_oracle",
                "reason": "odtbrain_import_succeeded",
            }
        }
    }
    extension_backend = {
        "name": "local_adjoint",
        "claim_boundary": "initialization_only",
        "reason": "born_init_image_uses_locked_local_adjoint",
    }
    handoff = ext_bundle.build_classical_backend_authority_handoff(
        baseline_manifest=baseline_manifest,
        extension_classical_backend=extension_backend,
    )
    assert handoff["extension_classical_backend"] == extension_backend
    assert (
        handoff["baseline_manifest_classical_backend"]["name"] == "odtbrain"
    )
    assert (
        handoff["baseline_neural_row_classical_backend"]["name"] == "local_adjoint"
    )
    assert "rationale" in handoff and "diverge" in handoff["rationale"]
    assert handoff["approved_by"] == "implementation_review_high_1_response"


def test_extension_bundle_combined_metrics_preserves_baseline_and_appends_ffno(
    tmp_path,
):
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle

    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    extension_root = tmp_path / "extension_root"
    extension_root.mkdir()
    # Synthetic extension manifest + metrics carrying only the FFNO row.
    extension_manifest = {
        "schema_version": "brdt_preflight_v1",
        "backlog_item": ext_bundle.EXTENSION_BACKLOG_ITEM,
        "claim_boundary": ext_bundle.APPEND_ONLY_CLAIM_BOUNDARY,
        "rows": [
            {
                "row_id": "ffno",
                "model": "ffno",
                "training": "supervised + Born consistency",
                "input_mode": "born_init_image",
                "dataset_id": dc.DECISION_SUPPORT_DATASET_NAME,
                "operator_version": "/tmp/operator_validation.json",
                "row_status": "completed",
                "paper_label": "FFNO",
            }
        ],
    }
    extension_metrics = {
        "schema_version": "brdt_preflight_metrics_v1",
        "claim_boundary": ext_bundle.APPEND_ONLY_CLAIM_BOUNDARY,
        "rows": [
            {
                "row_id": "ffno",
                "paper_label": "FFNO",
                "architecture": "ffno",
                "row_status": "completed",
                "image": {
                    "image_mae_phys": 0.0009,
                    "image_rmse_phys": 0.0019,
                    "image_relative_l2_phys": 0.45,
                },
                "measurement": {
                    "meas_mae": 0.0009,
                    "meas_rmse": 0.0018,
                    "meas_relative_l2": 0.27,
                },
                "supporting": {"psnr_phys": 27.0, "ssim_phys": 0.72},
                "runtime": {
                    "device": "cpu",
                    "device_name": "cpu",
                    "epochs": 20,
                    "batch_size": 16,
                    "learning_rate": 2e-4,
                    "parameter_count": 36674,
                    "wall_time_train_s": 5.0,
                    "wall_time_eval_s": 0.2,
                    "row_status": "completed",
                },
            }
        ],
    }
    (extension_root / "preflight_manifest.json").write_text(
        json.dumps(extension_manifest, indent=2, sort_keys=True) + "\n"
    )
    (extension_root / "metrics.json").write_text(
        json.dumps(extension_metrics, indent=2, sort_keys=True) + "\n"
    )

    paths = ext_bundle.emit_combined_bundle(
        baseline_root=baseline_root,
        extension_root=extension_root,
    )
    combined = json.loads(paths["combined_metrics_json"].read_text())
    # Five rows: original four baseline rows + one FFNO row.
    assert len(combined["rows"]) == 5
    row_ids = [row["row_id"] for row in combined["rows"]]
    assert row_ids == [
        "classical_born_backprop",
        "unet",
        "fno_vanilla",
        "hybrid_resnet",
        "ffno",
    ]
    sources = [row["source"] for row in combined["rows"]]
    assert sources[:4] == ["baseline_lineage"] * 4
    assert sources[-1] == "extension"
    # Combined manifest agrees on backlog identity + lineage pointers.
    manifest_payload = json.loads(paths["combined_manifest_json"].read_text())
    assert manifest_payload["backlog_item"] == ext_bundle.EXTENSION_BACKLOG_ITEM
    assert manifest_payload["claim_boundary"] == ext_bundle.APPEND_ONLY_CLAIM_BOUNDARY
    assert (
        manifest_payload["baseline"]["backlog_item"]
        == ext_bundle.BASELINE_BACKLOG_ITEM
    )
    # Baseline rows are preserved by lineage with their visible identity
    # intact: paper_label, architecture, and row_status are echoed verbatim
    # from the baseline bundle (the FFNO extension consumes the baseline
    # JSON files as they exist per the plan, not against any external
    # summary authority).
    baseline_combined_rows = [r for r in combined["rows"] if r["source"] == "baseline_lineage"]
    paper_labels = {r["row_id"]: r["paper_label"] for r in baseline_combined_rows}
    statuses = {r["row_id"]: r["row_status"] for r in baseline_combined_rows}
    assert paper_labels["classical_born_backprop"] == "Model-based Born inverse"
    assert paper_labels["hybrid_resnet"] == "Hybrid ResNet"
    assert statuses["classical_born_backprop"] == "completed"
    assert statuses["unet"] == "completed"
    assert statuses["fno_vanilla"] == "completed"
    assert statuses["hybrid_resnet"] == "completed"
    # Combined CSV header includes the source column.
    csv_text = paths["combined_metrics_csv"].read_text().splitlines()
    assert csv_text[0].startswith("source,row_id,paper_label,architecture")


def test_extension_bundle_emit_does_not_overwrite_baseline_artifacts(tmp_path):
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle

    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    baseline_manifest_before = (
        baseline_root / "preflight_manifest.json"
    ).read_bytes()
    baseline_metrics_before = (baseline_root / "metrics.json").read_bytes()
    extension_root = tmp_path / "extension_no_overwrite"
    extension_root.mkdir()
    (extension_root / "preflight_manifest.json").write_text(
        json.dumps({"rows": []})
    )
    (extension_root / "metrics.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row_id": "ffno",
                        "paper_label": "FFNO",
                        "architecture": "ffno",
                        "row_status": "completed",
                        "image": {},
                        "measurement": {},
                        "supporting": {},
                        "runtime": {},
                    }
                ]
            }
        )
    )
    ext_bundle.emit_combined_bundle(
        baseline_root=baseline_root,
        extension_root=extension_root,
    )
    assert (
        baseline_root / "preflight_manifest.json"
    ).read_bytes() == baseline_manifest_before
    assert (baseline_root / "metrics.json").read_bytes() == baseline_metrics_before


def test_run_ffno_extension_dry_run_writes_manifest_under_extension_root(tmp_path):
    """Dry-run path emits the FFNO-only manifest with append-only claim boundary."""
    from scripts.studies.born_rytov_dt import run_ffno_extension as ffno_mod

    manifest_path = _make_live_decision_support_dataset(tmp_path)
    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    # Patch the synthetic baseline so its dataset_id, in_channels, training
    # contract, split counts, and normalization match what the extension
    # would inherit from this CPU-friendly test fixture (epochs=1,
    # batch_size=1, lr=2e-4) so the tightened contract validator accepts it.
    base_manifest_path = baseline_root / "preflight_manifest.json"
    baseline = json.loads(base_manifest_path.read_text())
    live_manifest = json.loads(Path(manifest_path).read_text())
    op_pointer = live_manifest["operator"]["validation_artifact"]
    baseline["dataset"]["dataset_id"] = dc.DECISION_SUPPORT_DATASET_NAME
    baseline["dataset"]["split_counts"] = dict(live_manifest["split"]["counts"])
    baseline["dataset"]["normalization"] = dict(
        live_manifest.get("normalization") or {}
    )
    baseline["operator"]["validation_artifact"] = op_pointer
    baseline["training_contract"] = {
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 2e-4,
        "optimizer": "Adam",
        "loss_weights": run_config.LossWeights().as_dict(),
        "seed": 42,
    }
    base_manifest_path.write_text(json.dumps(baseline, indent=2, sort_keys=True))

    extension_root = tmp_path / "ffno_extension_dry"
    contract = preflight_mod.TrainingContract(
        epochs=1, batch_size=1, learning_rate=2e-4
    )
    result = ffno_mod.run_ffno_extension(
        baseline_root=baseline_root,
        manifest_path=manifest_path,
        output_root=extension_root,
        contract=contract,
        in_channels=1,
        device_choice="cpu",
        dry_run=True,
        parent_argv=["--dry-run"],
    )
    assert result.get("dry_run") is True
    manifest = json.loads(
        (extension_root / "preflight_manifest.json").read_text()
    )
    assert manifest["backlog_item"] == ffno_mod.BACKLOG_ITEM
    assert manifest["claim_boundary"] == ffno_mod.CLAIM_BOUNDARY
    # Only FFNO is listed under the extension's row roster.
    assert [r["row_id"] for r in manifest["rows"]] == ["ffno"]
    schema = json.loads((extension_root / "metric_schema.json").read_text())
    assert schema["claim_boundary"] == ffno_mod.CLAIM_BOUNDARY
    # The dry-run manifest must record the explicit classical_backend
    # authority handoff so the divergence between the baseline's
    # manifest-level surface (odtbrain external_oracle) and per-row neural
    # surface (local_adjoint) is documented, not silent.
    handoff = manifest["input_contract"]["classical_backend_authority_handoff"]
    assert handoff["extension_classical_backend"]["name"] == "local_adjoint"
    assert (
        handoff["baseline_neural_row_classical_backend"]["name"] == "local_adjoint"
    )
    assert handoff["approved_by"] == "implementation_review_high_1_response"


def test_run_ffno_extension_refuses_when_baseline_root_missing(tmp_path):
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle
    from scripts.studies.born_rytov_dt import run_ffno_extension as ffno_mod

    manifest_path = _make_live_decision_support_dataset(tmp_path)
    bogus_baseline = tmp_path / "no_such_baseline"
    bogus_baseline.mkdir()
    extension_root = tmp_path / "ffno_extension_invalid"
    contract = preflight_mod.TrainingContract(
        epochs=1, batch_size=1, learning_rate=2e-4
    )
    with pytest.raises(ext_bundle.BaselineContractMismatchError):
        ffno_mod.run_ffno_extension(
            baseline_root=bogus_baseline,
            manifest_path=manifest_path,
            output_root=extension_root,
            contract=contract,
            in_channels=1,
            device_choice="cpu",
            dry_run=True,
            parent_argv=["--dry-run"],
        )


def test_run_ffno_extension_live_emits_combined_bundle(tmp_path):
    """Live FFNO row run emits FFNO-only metrics + read-only-lineage combined bundle."""
    from scripts.studies.born_rytov_dt import run_ffno_extension as ffno_mod

    manifest_path = _make_live_decision_support_dataset(tmp_path)
    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    base_manifest_path = baseline_root / "preflight_manifest.json"
    baseline = json.loads(base_manifest_path.read_text())
    live_manifest = json.loads(Path(manifest_path).read_text())
    op_pointer = live_manifest["operator"]["validation_artifact"]
    baseline["dataset"]["dataset_id"] = dc.DECISION_SUPPORT_DATASET_NAME
    baseline["dataset"]["split_counts"] = dict(live_manifest["split"]["counts"])
    baseline["dataset"]["normalization"] = dict(
        live_manifest.get("normalization") or {}
    )
    baseline["operator"]["validation_artifact"] = op_pointer
    baseline["training_contract"] = {
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 2e-4,
        "optimizer": "Adam",
        "loss_weights": run_config.LossWeights().as_dict(),
        "seed": 42,
    }
    base_manifest_path.write_text(json.dumps(baseline, indent=2, sort_keys=True))

    extension_root = tmp_path / "ffno_extension_live"
    contract = preflight_mod.TrainingContract(
        epochs=1, batch_size=1, learning_rate=2e-4
    )
    result = ffno_mod.run_ffno_extension(
        baseline_root=baseline_root,
        manifest_path=manifest_path,
        output_root=extension_root,
        contract=contract,
        in_channels=1,
        device_choice="cpu",
        dry_run=False,
        parent_argv=["--manifest", str(manifest_path)],
    )
    # Extension-root artifacts present and append-only.
    metrics = json.loads((extension_root / "metrics.json").read_text())
    assert metrics["claim_boundary"] == ffno_mod.CLAIM_BOUNDARY
    assert [r["row_id"] for r in metrics["rows"]] == ["ffno"]
    assert metrics["rows"][0]["row_status"] == "completed"
    # Combined view preserves baseline rows (read-only lineage) + FFNO.
    combined = json.loads(
        (extension_root / "combined_metrics.json").read_text()
    )
    row_ids = [row["row_id"] for row in combined["rows"]]
    assert row_ids == [
        "classical_born_backprop",
        "unet",
        "fno_vanilla",
        "hybrid_resnet",
        "ffno",
    ]
    # Baseline bundle is left untouched.
    assert (baseline_root / "preflight_manifest.json").exists()
    base_payload = json.loads(
        (baseline_root / "preflight_manifest.json").read_text()
    )
    assert base_payload["backlog_item"] == "2026-04-29-brdt-four-row-preflight"
    # Per-row provenance + state under rows/ffno/.
    ffno_summary = json.loads(
        (extension_root / "rows" / "ffno" / "row_summary.json").read_text()
    )
    assert ffno_summary["row_id"] == "ffno"
    assert ffno_summary["row_status"] == "completed"
    assert "model_state_path" in ffno_summary
    # Returned paths surface the combined-bundle artifacts.
    assert "combined_metrics_json_path" in result
    assert "combined_manifest_json_path" in result


def test_run_preflight_default_invocation_artifacts_keep_baseline_backlog_item(
    tmp_path,
):
    """Default (supervised+Born) runs must keep the original backlog item
    in invocation provenance so prior bundles are not silently relabeled."""
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    preflight_root = tmp_path / "default_invocation"
    _run_preflight(
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(preflight_root),
        "--dry-run",
    )
    top = json.loads((preflight_root / "invocation.json").read_text())
    assert top["extra"]["backlog_item"] == preflight_mod.BACKLOG_ITEM
    assert top["extra"]["backlog_item"] != preflight_mod.PHYSICS_ONLY_BACKLOG_ITEM


# ----------------------------------------------------------------------
# BRDT 40-epoch paper-evidence follow-up (2026-05-05)
# ----------------------------------------------------------------------
def _make_synthetic_ffno_extension_bundle(tmp_path: Path, baseline_root: Path) -> Path:
    from scripts.studies.born_rytov_dt import extension_bundle as ext_bundle

    extension_root = tmp_path / "synthetic_ffno_extension"
    extension_root.mkdir()
    baseline_manifest = json.loads((baseline_root / "preflight_manifest.json").read_text())
    baseline_metrics = json.loads((baseline_root / "metrics.json").read_text())
    baseline_rows = {row["row_id"]: row for row in baseline_metrics["rows"]}

    manifest = {
        "schema_version": "brdt_preflight_v1",
        "backlog_item": ext_bundle.EXTENSION_BACKLOG_ITEM,
        "claim_boundary": ext_bundle.APPEND_ONLY_CLAIM_BOUNDARY,
        "baseline_lineage": {
            "baseline_root": str(baseline_root),
            "baseline_backlog_item": ext_bundle.BASELINE_BACKLOG_ITEM,
            "baseline_preflight_manifest": str(
                baseline_root / "preflight_manifest.json"
            ),
            "baseline_metrics_json": str(baseline_root / "metrics.json"),
        },
        "dataset": dict(baseline_manifest["dataset"]),
        "operator": dict(baseline_manifest["operator"]),
        "input_contract": {
            "input_mode": "born_init_image",
            "in_channels": 1,
            "classical_backend": {
                "name": "local_adjoint",
                "reason": "born_init_image_uses_locked_local_adjoint",
                "claim_boundary": "initialization_only",
            },
        },
        "training_contract": dict(baseline_manifest["training_contract"]),
        "fixed_sample_seed": int(baseline_manifest["fixed_sample_seed"]),
        "fixed_sample_ids": list(baseline_manifest["fixed_sample_ids"]),
        "rows": [
            {
                "row_id": "ffno",
                "model": "ffno",
                "training": "supervised + Born consistency",
                "input_mode": "born_init_image",
                "dataset_id": dc.DECISION_SUPPORT_DATASET_NAME,
                "operator_version": baseline_manifest["operator"][
                    "validation_artifact"
                ],
                "row_status": "completed",
                "paper_label": "FFNO",
            }
        ],
    }
    metrics = {
        "schema_version": "brdt_preflight_metrics_v1",
        "claim_boundary": ext_bundle.APPEND_ONLY_CLAIM_BOUNDARY,
        "rows": [
            {
                "row_id": "ffno",
                "paper_label": "FFNO",
                "architecture": "ffno",
                "row_status": "completed",
                "image": {
                    "image_mae_phys": 0.0008,
                    "image_rmse_phys": 0.0018,
                    "image_relative_l2_phys": 0.34,
                },
                "measurement": {
                    "meas_mae": 0.0008,
                    "meas_rmse": 0.0017,
                    "meas_relative_l2": 0.19,
                },
                "supporting": {"psnr_phys": 29.1, "ssim_phys": 0.94},
                "runtime": {
                    "device": "cpu",
                    "device_name": "cpu",
                    "epochs": 20,
                    "batch_size": 16,
                    "learning_rate": 2e-4,
                    "parameter_count": 36674,
                    "wall_time_train_s": 5.0,
                    "wall_time_eval_s": 0.2,
                    "row_status": "completed",
                },
            }
        ],
    }
    combined = {
        "schema_version": "brdt_ffno_extension_combined_v1",
        "claim_boundary": ext_bundle.APPEND_ONLY_CLAIM_BOUNDARY,
        "rows": [
            {**baseline_rows["classical_born_backprop"], "source": "baseline_lineage"},
            {**baseline_rows["unet"], "source": "baseline_lineage"},
            {**baseline_rows["fno_vanilla"], "source": "baseline_lineage"},
            {**baseline_rows["hybrid_resnet"], "source": "baseline_lineage"},
            {**metrics["rows"][0], "source": "extension"},
        ],
    }
    (extension_root / "preflight_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (extension_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    (extension_root / "combined_metrics.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n"
    )
    return extension_root


def test_train_neural_row_writes_history_with_scheduler_fields(tmp_path):
    manifest_path = _make_live_decision_support_dataset(tmp_path)
    authority = preflight_mod.load_dataset_authority(manifest_path)
    row = preflight_mod.resolve_row_roster(
        manifest_path=manifest_path,
        hybrid_label="hybrid_resnet",
        selected_row_ids=["hybrid_resnet"],
    )[0]
    operator = preflight_mod._build_operator(torch.device("cpu"))
    backend = preflight_mod._born_init_backend()
    contract = preflight_mod.TrainingContract(
        epochs=2,
        batch_size=1,
        learning_rate=2e-4,
        scheduler="reduce_on_plateau",
        plateau_factor=0.5,
        plateau_patience=2,
        plateau_threshold=0.0,
        plateau_min_lr=1e-5,
    )

    module, runtime_meta, _elapsed = preflight_mod._train_neural_row(
        row=row,
        authority=authority,
        operator=operator,
        backend=backend,
        device=torch.device("cpu"),
        contract=contract,
        in_channels=1,
        output_dir=tmp_path / "row_history",
    )

    assert module is not None
    history_path = Path(runtime_meta["history_json_path"])
    history_csv_path = Path(runtime_meta["history_csv_path"])
    assert history_path.exists()
    assert history_csv_path.exists()
    history = json.loads(history_path.read_text())
    assert history["scheduler"]["name"] == "reduce_on_plateau"
    assert len(history["epochs"]) == 2
    epoch0 = history["epochs"][0]
    for key in (
        "epoch",
        "train_total_loss",
        "train_loss_components",
        "learning_rate",
        "scheduler_metric",
        "lr_reduced",
    ):
        assert key in epoch0
    assert history_csv_path.read_text().splitlines()[0].startswith(
        "epoch,train_total_loss"
    )


def test_paper_evidence_gate_requires_40_history_and_separate_promotion_status():
    from scripts.studies.born_rytov_dt import convergence as conv_mod

    gate = conv_mod.build_paper_evidence_gate(
        backlog_item="2026-05-05-brdt-supervised-born-40ep-paper-evidence",
        expected_epochs=40,
        rows={
            "hybrid_resnet": {
                "row_status": "completed",
                "history_records": 40,
                "scheduler_matches_contract": True,
            },
            "ffno": {
                "row_status": "completed",
                "history_records": 39,
                "scheduler_matches_contract": True,
            },
        },
        provenance_checks={
            "runtime_provenance": True,
            "dataset_identity": True,
            "split_manifest": True,
            "sample_255_visual_bundle": True,
            "exit_code_proof": True,
            "evidence_surfaces_prepared": True,
            "same_contract_lineage": True,
        },
    )

    assert gate["claim_boundary"] == "decision_support_convergence_followup"
    assert gate["promotion_status"] == "failed"
    assert gate["row_status"] == "completed"
    assert "ffno.history_records" in gate["failed_gate_checks"]


def test_run_brdt_40ep_paper_evidence_dry_run_writes_locked_manifest(tmp_path):
    from scripts.studies.born_rytov_dt import (
        run_brdt_40ep_paper_evidence as paper_mod,
    )

    manifest_path = _make_live_decision_support_dataset(tmp_path)
    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    ffno_extension_root = _make_synthetic_ffno_extension_bundle(
        tmp_path, baseline_root
    )

    result = paper_mod.run_paper_evidence(
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
        manifest_path=manifest_path,
        output_root=tmp_path / "paper_evidence_dry",
        device_choice="cpu",
        dry_run=True,
        fixed_sample_ids=[145, 83, 255, 126],
        required_paper_sample=255,
        parent_argv=["--dry-run"],
    )

    assert result["dry_run"] is True
    manifest = json.loads(
        (tmp_path / "paper_evidence_dry" / "preflight_manifest.json").read_text()
    )
    assert manifest["backlog_item"] == paper_mod.BACKLOG_ITEM
    assert manifest["training_contract"]["epochs"] == 40
    assert manifest["training_contract"]["scheduler"] == "reduce_on_plateau"
    assert manifest["required_paper_sample"] == 255
    assert manifest["fixed_sample_ids"] == [145, 83, 255, 126]
    assert [row["row_id"] for row in manifest["rows"]] == [
        "hybrid_resnet",
        "ffno",
    ]
    assert manifest["claim_boundary"] == "decision_support_convergence_followup"
    assert manifest["promotion_status"] == "pending"


def test_run_brdt_40ep_paper_evidence_live_writes_histories_audit_and_gate(tmp_path):
    from scripts.studies.born_rytov_dt import (
        run_brdt_40ep_paper_evidence as paper_mod,
    )

    manifest_path = _make_live_decision_support_dataset(tmp_path)
    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    ffno_extension_root = _make_synthetic_ffno_extension_bundle(
        tmp_path, baseline_root
    )
    output_root = tmp_path / "paper_evidence_live"
    contract = preflight_mod.TrainingContract(
        epochs=1,
        batch_size=1,
        learning_rate=2e-4,
        scheduler="reduce_on_plateau",
        plateau_factor=0.5,
        plateau_patience=2,
        plateau_threshold=0.0,
        plateau_min_lr=1e-5,
    )

    result = paper_mod.run_paper_evidence(
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
        manifest_path=manifest_path,
        output_root=output_root,
        contract=contract,
        device_choice="cpu",
        dry_run=False,
        fixed_sample_ids=[0],
        required_paper_sample=0,
        parent_argv=["--manifest", str(manifest_path)],
    )

    assert "convergence_audit_json_path" in result
    assert "paper_evidence_gate_json_path" in result
    for row_id in ("hybrid_resnet", "ffno"):
        history_json = output_root / "rows" / row_id / "history.json"
        history_csv = output_root / "rows" / row_id / "history.csv"
        assert history_json.exists()
        assert history_csv.exists()
    gate = json.loads((output_root / "paper_evidence_gate.json").read_text())
    assert gate["promotion_status"] == "failed"
    assert gate["claim_boundary"] == "decision_support_convergence_followup"
    audit = json.loads((output_root / "convergence_audit.json").read_text())
    assert {row["row_id"] for row in audit["rows"]} == {"hybrid_resnet", "ffno"}

    # The runner must re-seed the top-level manifest with the gate's final
    # claim boundary and promotion status so the bundle cannot present a
    # passing additive label without the gate actually passing.
    manifest = json.loads((output_root / "preflight_manifest.json").read_text())
    assert manifest["claim_boundary"] == gate["claim_boundary"]
    assert manifest["promotion_status"] == gate["promotion_status"]
    assert manifest["paper_evidence_gate_path"] == str(
        output_root / "paper_evidence_gate.json"
    )

    # Provenance contract: stronger gate-side checks must be present in the
    # gate payload's provenance_checks dict.
    pc = gate["provenance_checks"]
    for required_key in (
        "git_provenance",
        "host_provenance",
        "model_profiles",
        "run_log_present",
        "evidence_surfaces_prepared",
    ):
        assert required_key in pc

    runtime_provenance = json.loads(
        (output_root / "runtime_provenance.json").read_text()
    )
    for required_field in (
        "git_sha",
        "git_dirty",
        "hostname",
        "launch_timestamp_utc",
        "gpu_count",
        "tracked_pid",
    ):
        assert required_field in runtime_provenance
    exit_status = json.loads((output_root / "run_exit_status.json").read_text())
    assert "tracked_pid" in exit_status
    assert "log_path" in exit_status  # may be None when no logs/ exists


def test_run_brdt_40ep_paper_evidence_refuses_duplicate_writer(tmp_path):
    from scripts.studies.born_rytov_dt import (
        run_brdt_40ep_paper_evidence as paper_mod,
    )

    manifest_path = _make_live_decision_support_dataset(tmp_path)
    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    ffno_extension_root = _make_synthetic_ffno_extension_bundle(
        tmp_path, baseline_root
    )
    output_root = tmp_path / "writer_lock_root"
    output_root.mkdir()
    # Simulate a live writer holding the lock with a real PID.
    (output_root / paper_mod.WRITER_LOCK_NAME).write_text(
        json.dumps(
            {"pid": int(os.getpid()), "host": "test", "acquired_utc": "1970-01-01T00:00:00+00:00"}
        )
    )
    # Use a separate fake PID to verify the refusal path: monkey-patch _pid_alive.
    other_pid = 999_999_999
    (output_root / paper_mod.WRITER_LOCK_NAME).write_text(
        json.dumps({"pid": other_pid, "host": "test", "acquired_utc": "1970-01-01T00:00:00+00:00"})
    )
    original = paper_mod._pid_alive
    paper_mod._pid_alive = lambda pid: pid == other_pid
    try:
        with pytest.raises(paper_mod.WriterConflictError):
            paper_mod.run_paper_evidence(
                baseline_root=baseline_root,
                ffno_extension_root=ffno_extension_root,
                manifest_path=manifest_path,
                output_root=output_root,
                contract=preflight_mod.TrainingContract(
                    epochs=1, batch_size=1, learning_rate=2e-4
                ),
                device_choice="cpu",
                dry_run=False,
                fixed_sample_ids=[0],
                required_paper_sample=0,
                parent_argv=[],
            )
    finally:
        paper_mod._pid_alive = original


def test_run_brdt_40ep_paper_evidence_refuses_completed_root(tmp_path):
    from scripts.studies.born_rytov_dt import (
        run_brdt_40ep_paper_evidence as paper_mod,
    )

    manifest_path = _make_live_decision_support_dataset(tmp_path)
    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    ffno_extension_root = _make_synthetic_ffno_extension_bundle(
        tmp_path, baseline_root
    )
    output_root = tmp_path / "completed_root"
    output_root.mkdir()
    # Simulate a previously completed bundle.
    (output_root / "paper_evidence_gate.json").write_text("{}")
    (output_root / "run_exit_status.json").write_text("{}")
    with pytest.raises(paper_mod.WriterConflictError):
        paper_mod.run_paper_evidence(
            baseline_root=baseline_root,
            ffno_extension_root=ffno_extension_root,
            manifest_path=manifest_path,
            output_root=output_root,
            contract=preflight_mod.TrainingContract(
                epochs=1, batch_size=1, learning_rate=2e-4
            ),
            device_choice="cpu",
            dry_run=False,
            fixed_sample_ids=[0],
            required_paper_sample=0,
            parent_argv=[],
        )


def test_rebuild_meta_only_refreshes_manifest_provenance_and_gate(tmp_path):
    """rebuild_meta_only must rebuild the top-level manifest, provenance,
    audit, and gate payloads from existing per-row outputs without retraining."""
    from scripts.studies.born_rytov_dt import (
        run_brdt_40ep_paper_evidence as paper_mod,
    )

    manifest_path = _make_live_decision_support_dataset(tmp_path)
    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    ffno_extension_root = _make_synthetic_ffno_extension_bundle(
        tmp_path, baseline_root
    )
    output_root = tmp_path / "rebuild_meta_root"
    contract = preflight_mod.TrainingContract(
        epochs=1,
        batch_size=1,
        learning_rate=2e-4,
        scheduler="reduce_on_plateau",
        plateau_factor=0.5,
        plateau_patience=2,
        plateau_threshold=0.0,
        plateau_min_lr=1e-5,
    )

    # First, do a real live run so the per-row outputs exist.
    paper_mod.run_paper_evidence(
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
        manifest_path=manifest_path,
        output_root=output_root,
        contract=contract,
        device_choice="cpu",
        dry_run=False,
        fixed_sample_ids=[0],
        required_paper_sample=0,
        parent_argv=[],
    )

    # Corrupt the top-level manifest to simulate a stale meta payload.
    manifest_path_top = output_root / "preflight_manifest.json"
    payload = json.loads(manifest_path_top.read_text())
    payload["claim_boundary"] = "paper_evidence_brdt_additive"
    payload["promotion_status"] = "passed"
    manifest_path_top.write_text(json.dumps(payload, indent=2, sort_keys=True))

    # Rebuild meta only: must recompute gate honestly and re-seed manifest.
    result = paper_mod.rebuild_meta_only(
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
        manifest_path=manifest_path,
        output_root=output_root,
        contract=contract,
        fixed_sample_ids=[0],
        required_paper_sample=0,
    )
    assert result["rebuild_meta_only"] is True
    gate = json.loads((output_root / "paper_evidence_gate.json").read_text())
    refreshed = json.loads(manifest_path_top.read_text())
    assert refreshed["claim_boundary"] == gate["claim_boundary"]
    assert refreshed["promotion_status"] == gate["promotion_status"]
    # Synthetic single-epoch contract still fails the 40-epoch gate.
    assert gate["promotion_status"] == "failed"

    # The rebuild must keep runtime_provenance.tracked_pid and
    # run_exit_status.tracked_pid in agreement so the bundle still identifies
    # one authoritative tracked run.
    runtime_provenance = json.loads(
        (output_root / "runtime_provenance.json").read_text()
    )
    exit_status = json.loads((output_root / "run_exit_status.json").read_text())
    assert runtime_provenance["tracked_pid"] == exit_status["tracked_pid"]
    # Rebuild metadata must be recorded separately from the original tracked run.
    assert "meta_rebuild" in runtime_provenance
    assert "rebuild_pid" in runtime_provenance["meta_rebuild"]
    # exit_code_proof must be True because the tracked PIDs match.
    assert gate["provenance_checks"]["exit_code_proof"] is True


def test_rebuild_meta_only_refuses_when_run_exit_status_missing(tmp_path):
    """Reviewer-blocked defect: ``rebuild_meta_only`` previously defaulted
    ``exit_code=0`` and ``status="completed"`` whenever ``run_exit_status.json``
    was missing or unreadable, then rewrote the file with that fabricated
    record. After the fix, the rebuild path must refuse to fabricate
    completion evidence; if the original exit-status proof is gone, the
    bundle has to be retrained, not silently regenerated."""
    from scripts.studies.born_rytov_dt import (
        run_brdt_40ep_paper_evidence as paper_mod,
    )

    manifest_path = _make_live_decision_support_dataset(tmp_path)
    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    ffno_extension_root = _make_synthetic_ffno_extension_bundle(
        tmp_path, baseline_root
    )
    output_root = tmp_path / "rebuild_no_exit_status_root"
    contract = preflight_mod.TrainingContract(
        epochs=1,
        batch_size=1,
        learning_rate=2e-4,
        scheduler="reduce_on_plateau",
        plateau_factor=0.5,
        plateau_patience=2,
        plateau_threshold=0.0,
        plateau_min_lr=1e-5,
    )

    # Live-run once so per-row outputs exist on disk.
    paper_mod.run_paper_evidence(
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
        manifest_path=manifest_path,
        output_root=output_root,
        contract=contract,
        device_choice="cpu",
        dry_run=False,
        fixed_sample_ids=[0],
        required_paper_sample=0,
        parent_argv=[],
    )

    exit_status_path = output_root / "run_exit_status.json"
    assert exit_status_path.exists()
    exit_status_path.unlink()

    # Missing record: refuse to rebuild rather than fabricate completion.
    with pytest.raises(FileNotFoundError):
        paper_mod.rebuild_meta_only(
            baseline_root=baseline_root,
            ffno_extension_root=ffno_extension_root,
            manifest_path=manifest_path,
            output_root=output_root,
            contract=contract,
            fixed_sample_ids=[0],
            required_paper_sample=0,
        )
    assert not exit_status_path.exists()

    # Unparseable record: refuse rather than coerce to defaults.
    exit_status_path.write_text("{not-json")
    with pytest.raises(RuntimeError):
        paper_mod.rebuild_meta_only(
            baseline_root=baseline_root,
            ffno_extension_root=ffno_extension_root,
            manifest_path=manifest_path,
            output_root=output_root,
            contract=contract,
            fixed_sample_ids=[0],
            required_paper_sample=0,
        )

    # Parseable but missing tracked_pid/exit_code/status: refuse.
    exit_status_path.write_text(json.dumps({"unrelated": "value"}))
    with pytest.raises(RuntimeError):
        paper_mod.rebuild_meta_only(
            baseline_root=baseline_root,
            ffno_extension_root=ffno_extension_root,
            manifest_path=manifest_path,
            output_root=output_root,
            contract=contract,
            fixed_sample_ids=[0],
            required_paper_sample=0,
        )


def test_reseed_metrics_with_gate_aligns_claim_boundary(tmp_path):
    """Reviewer-blocked defect: the live runner stamped ``metrics.json`` /
    ``combined_metrics.json`` / ``metric_schema.json`` with the pre-gate
    boundary before the gate ran, so a promoted bundle ended up with the gate
    advertising ``paper_evidence_brdt_additive`` while the metric tables
    still advertised ``decision_support_convergence_followup``. The reseed
    helper must rewrite all three files to match the gate's final boundary."""
    from scripts.studies.born_rytov_dt import (
        run_brdt_40ep_paper_evidence as paper_mod,
    )

    output_root = tmp_path / "reseed_metrics_root"
    output_root.mkdir()
    pre_gate = paper_mod.PRE_GATE_CLAIM_BOUNDARY
    promoted = paper_mod.PASSED_CLAIM_BOUNDARY

    metrics_path = output_root / "metrics.json"
    combined_path = output_root / "combined_metrics.json"
    schema_path = output_root / "metric_schema.json"
    metrics_path.write_text(json.dumps({"claim_boundary": pre_gate, "rows": []}))
    combined_path.write_text(json.dumps({"claim_boundary": pre_gate, "rows": []}))
    schema_path.write_text(json.dumps({"claim_boundary": pre_gate}))

    paper_mod._reseed_metrics_with_gate(
        output_root=output_root,
        gate_payload={"claim_boundary": promoted},
    )

    assert json.loads(metrics_path.read_text())["claim_boundary"] == promoted
    assert json.loads(combined_path.read_text())["claim_boundary"] == promoted
    assert json.loads(schema_path.read_text())["claim_boundary"] == promoted

    # Failed-gate case: a failed verdict must propagate the pre-gate label
    # back through the metric tables so the bundle never carries a stale
    # promoted label after a failed rerun.
    paper_mod._reseed_metrics_with_gate(
        output_root=output_root,
        gate_payload={"claim_boundary": pre_gate},
    )
    assert json.loads(metrics_path.read_text())["claim_boundary"] == pre_gate
    assert json.loads(combined_path.read_text())["claim_boundary"] == pre_gate
    assert json.loads(schema_path.read_text())["claim_boundary"] == pre_gate


def test_rebuild_meta_only_refuses_active_writer(tmp_path):
    """rebuild_meta_only must refuse to run when another writer holds the
    output-root lock, the same protection the live training path applies."""
    from scripts.studies.born_rytov_dt import (
        run_brdt_40ep_paper_evidence as paper_mod,
    )

    manifest_path = _make_live_decision_support_dataset(tmp_path)
    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    ffno_extension_root = _make_synthetic_ffno_extension_bundle(
        tmp_path, baseline_root
    )
    output_root = tmp_path / "rebuild_meta_lock_root"
    output_root.mkdir()
    other_pid = 999_999_999
    (output_root / paper_mod.WRITER_LOCK_NAME).write_text(
        json.dumps(
            {
                "pid": other_pid,
                "host": "test",
                "acquired_utc": "1970-01-01T00:00:00+00:00",
            }
        )
    )
    original = paper_mod._pid_alive
    paper_mod._pid_alive = lambda pid: pid == other_pid
    try:
        with pytest.raises(paper_mod.WriterConflictError):
            paper_mod.rebuild_meta_only(
                baseline_root=baseline_root,
                ffno_extension_root=ffno_extension_root,
                manifest_path=manifest_path,
                output_root=output_root,
                fixed_sample_ids=[0],
                required_paper_sample=0,
            )
    finally:
        paper_mod._pid_alive = original


def test_evidence_surfaces_consistency_check_requires_all_surfaces(tmp_path):
    """The evidence_surfaces_prepared gate check must require the durable
    summary, the paper-evidence index, the paper-evidence manifest, AND the
    repo-wide ``docs/index.md`` discoverability surface to all reference this
    backlog item AND the canonical artifact root AND the manifest's
    authoritative claim-boundary string."""
    from scripts.studies.born_rytov_dt import (
        run_brdt_40ep_paper_evidence as paper_mod,
    )

    fake_repo = tmp_path / "fake_repo"
    (fake_repo / "docs" / "plans" / "NEURIPS-HYBRID-RESNET-2026").mkdir(parents=True)
    (fake_repo / "docs").mkdir(exist_ok=True)

    summary_path = fake_repo / paper_mod.DURABLE_SUMMARY_PATH
    index_path = fake_repo / paper_mod.PAPER_EVIDENCE_INDEX_PATH
    manifest_path = fake_repo / paper_mod.PAPER_EVIDENCE_MANIFEST_PATH
    docs_index_path = fake_repo / paper_mod.DOCS_INDEX_PATH

    canonical_root = paper_mod.CANONICAL_ARTIFACT_ROOT
    boundary = paper_mod.PASSED_CLAIM_BOUNDARY
    consistent_marker = (
        f"{paper_mod.BACKLOG_ITEM} {canonical_root} {boundary}\n"
    )
    consistent_manifest_payload = {
        "row_registry": [
            {
                "claim_boundary": boundary,
                "source_root": f"{canonical_root}/",
            }
        ]
    }

    # Only the durable summary references the backlog item: must fail.
    summary_path.write_text(consistent_marker)
    assert (
        paper_mod._check_evidence_surfaces_consistent(repo_root=fake_repo) is False
    )

    # Add the paper-evidence index but not the manifest: must still fail.
    index_path.write_text(consistent_marker)
    assert (
        paper_mod._check_evidence_surfaces_consistent(repo_root=fake_repo) is False
    )

    # Add the manifest but not docs/index.md: must still fail because the
    # repo-wide docs index discoverability surface is required by the plan.
    manifest_path.write_text(json.dumps(consistent_manifest_payload))
    assert (
        paper_mod._check_evidence_surfaces_consistent(repo_root=fake_repo) is False
    )

    # All four surfaces present and consistent (backlog id, canonical artifact
    # root, and matching claim boundary): must pass.
    docs_index_path.write_text(consistent_marker)
    assert (
        paper_mod._check_evidence_surfaces_consistent(repo_root=fake_repo) is True
    )

    # Drift the durable summary's claim-boundary while the manifest still
    # advertises the promoted boundary: must fail because the surfaces no
    # longer agree on a single claim boundary.
    summary_path.write_text(
        f"{paper_mod.BACKLOG_ITEM} {canonical_root} "
        f"{paper_mod.PRE_GATE_CLAIM_BOUNDARY}\n"
    )
    assert (
        paper_mod._check_evidence_surfaces_consistent(repo_root=fake_repo) is False
    )

    # Drift the artifact root in one surface: must fail.
    summary_path.write_text(consistent_marker)
    docs_index_path.write_text(
        f"{paper_mod.BACKLOG_ITEM} .artifacts/wrong/path {boundary}\n"
    )
    assert (
        paper_mod._check_evidence_surfaces_consistent(repo_root=fake_repo) is False
    )

    # Manifest entry missing claim_boundary: must fail.
    docs_index_path.write_text(consistent_marker)
    manifest_path.write_text(
        json.dumps(
            {"row_registry": [{"source_root": f"{canonical_root}/"}]}
        )
    )
    assert (
        paper_mod._check_evidence_surfaces_consistent(repo_root=fake_repo) is False
    )


def test_evidence_surfaces_consistency_check_detects_manifest_boundary_drift(
    tmp_path,
):
    """When the manifest's authoritative claim_boundary entry drifts from the
    boundary advertised by the other discoverability surfaces, the check must
    fail. This is the precise drift scenario flagged by the reviewer:
    ``paper_evidence_manifest.json`` could reference one boundary while
    ``paper_evidence_index.md``/``docs/index.md`` referenced another."""
    from scripts.studies.born_rytov_dt import (
        run_brdt_40ep_paper_evidence as paper_mod,
    )

    fake_repo = tmp_path / "fake_repo"
    (fake_repo / "docs" / "plans" / "NEURIPS-HYBRID-RESNET-2026").mkdir(parents=True)
    (fake_repo / "docs").mkdir(exist_ok=True)

    summary_path = fake_repo / paper_mod.DURABLE_SUMMARY_PATH
    index_path = fake_repo / paper_mod.PAPER_EVIDENCE_INDEX_PATH
    manifest_path = fake_repo / paper_mod.PAPER_EVIDENCE_MANIFEST_PATH
    docs_index_path = fake_repo / paper_mod.DOCS_INDEX_PATH

    canonical_root = paper_mod.CANONICAL_ARTIFACT_ROOT
    promoted = paper_mod.PASSED_CLAIM_BOUNDARY
    pre_gate = paper_mod.PRE_GATE_CLAIM_BOUNDARY

    # Index, summary, and docs/index advertise the promoted boundary.
    consistent_marker = (
        f"{paper_mod.BACKLOG_ITEM} {canonical_root} {promoted}\n"
    )
    summary_path.write_text(consistent_marker)
    index_path.write_text(consistent_marker)
    docs_index_path.write_text(consistent_marker)

    # The manifest's authoritative entry drifts to the pre-gate boundary.
    manifest_path.write_text(
        json.dumps(
            {
                "row_registry": [
                    {
                        "claim_boundary": pre_gate,
                        "source_root": f"{canonical_root}/",
                    }
                ]
            }
        )
        + f"\n{pre_gate}\n"
    )
    assert (
        paper_mod._check_evidence_surfaces_consistent(repo_root=fake_repo) is False
    )


def test_scheduler_matches_contract_rejects_plateau_drift():
    """Reviewer-blocked defect: ``scheduler_matches_contract`` previously
    accepted any bundle whose scheduler name matched, even when plateau
    factor/patience/threshold/min_lr drifted. The strengthened helper must
    fail every plan-bound scheduler/optimizer field check."""
    from scripts.studies.born_rytov_dt import (
        run_brdt_40ep_paper_evidence as paper_mod,
    )

    contract = preflight_mod.TrainingContract(
        epochs=40,
        batch_size=16,
        learning_rate=2e-4,
        scheduler="reduce_on_plateau",
        plateau_factor=0.5,
        plateau_patience=2,
        plateau_threshold=0.0,
        plateau_min_lr=1e-5,
    ).as_dict()

    matching_summary = {
        "scheduler": {
            "name": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 2,
            "threshold": 0.0,
            "min_lr": 1e-5,
        },
        "runtime": {"epochs": 40, "batch_size": 16, "learning_rate": 2e-4},
    }
    assert paper_mod._scheduler_matches_contract(
        row_summary=matching_summary, contract_dict=contract
    )

    # Drifted plateau_factor: must fail despite name matching.
    drift_summary = json.loads(json.dumps(matching_summary))
    drift_summary["scheduler"]["factor"] = 0.25
    assert not paper_mod._scheduler_matches_contract(
        row_summary=drift_summary, contract_dict=contract
    )

    # Drifted plateau_patience: must fail.
    drift_summary = json.loads(json.dumps(matching_summary))
    drift_summary["scheduler"]["patience"] = 5
    assert not paper_mod._scheduler_matches_contract(
        row_summary=drift_summary, contract_dict=contract
    )

    # Drifted plateau_min_lr: must fail.
    drift_summary = json.loads(json.dumps(matching_summary))
    drift_summary["scheduler"]["min_lr"] = 1e-3
    assert not paper_mod._scheduler_matches_contract(
        row_summary=drift_summary, contract_dict=contract
    )

    # Drifted batch size in optimizer recipe: must fail.
    drift_summary = json.loads(json.dumps(matching_summary))
    drift_summary["runtime"]["batch_size"] = 32
    assert not paper_mod._scheduler_matches_contract(
        row_summary=drift_summary, contract_dict=contract
    )


def test_sample_visual_source_arrays_check_requires_each_required_file(tmp_path):
    """Reviewer-blocked defect: ``sample_255_visual_bundle`` previously only
    checked ``classical_present`` plus a non-empty figure list, so the
    durable per-row source arrays could disappear without the gate
    failing. The strengthened helper must require every per-sample,
    per-row source-array file."""
    from scripts.studies.born_rytov_dt import (
        run_brdt_40ep_paper_evidence as paper_mod,
    )

    output_root = tmp_path / "fake_bundle"
    arrays_dir = output_root / "figures" / "source_arrays"
    arrays_dir.mkdir(parents=True)
    sid = 255
    required_names = [
        f"sample_{sid:04d}_q_target.npy",
        f"sample_{sid:04d}_sino_obs.npy",
        f"sample_{sid:04d}_classical_born_backprop_q_pred.npy",
        f"sample_{sid:04d}_classical_born_backprop_sino_pred.npy",
        f"sample_{sid:04d}_hybrid_resnet_q_pred.npy",
        f"sample_{sid:04d}_hybrid_resnet_sino_pred.npy",
        f"sample_{sid:04d}_ffno_q_pred.npy",
        f"sample_{sid:04d}_ffno_sino_pred.npy",
    ]
    # Missing every file: must fail.
    assert not paper_mod._check_sample_visual_source_arrays(
        output_root=output_root, sample_id=sid
    )
    # Populate all but one file: must still fail.
    for name in required_names[:-1]:
        (arrays_dir / name).write_bytes(b"\x00")
    assert not paper_mod._check_sample_visual_source_arrays(
        output_root=output_root, sample_id=sid
    )
    # Populate the last file: must pass.
    (arrays_dir / required_names[-1]).write_bytes(b"\x00")
    assert paper_mod._check_sample_visual_source_arrays(
        output_root=output_root, sample_id=sid
    )


def test_exit_code_proof_requires_completed_status_and_zero_exit_code(tmp_path):
    """Reviewer-blocked defect: ``exit_code_proof`` previously only required
    ``run_exit_status.json`` to exist and the tracked PID to match
    ``runtime_provenance.json``. A bundle whose run actually failed (with
    ``status != 'completed'`` or ``exit_code != 0``) would still pass.
    The strengthened check must require both."""
    from scripts.studies.born_rytov_dt import (
        run_brdt_40ep_paper_evidence as paper_mod,
    )

    output_root = tmp_path / "exit_code_proof_root"
    output_root.mkdir()
    (output_root / "rows" / "hybrid_resnet").mkdir(parents=True)
    (output_root / "rows" / "ffno").mkdir(parents=True)
    (output_root / "rows" / "hybrid_resnet" / "model_profile.json").write_text("{}")
    (output_root / "rows" / "ffno" / "model_profile.json").write_text("{}")

    # Minimal runtime/dataset/split provenance files so the surrounding
    # checks do not collapse the test focus.
    runtime_payload = {
        "tracked_pid": 4242,
        "git_sha": "abc",
        "git_dirty": False,
        "hostname": "test-host",
        "gpu_count": 0,
    }
    (output_root / "runtime_provenance.json").write_text(json.dumps(runtime_payload))
    (output_root / "dataset_identity_manifest.json").write_text("{}")
    (output_root / "split_manifest.json").write_text("{}")
    (output_root / "preflight_manifest.json").write_text(
        json.dumps({"required_paper_sample": 0})
    )
    log_path = output_root / "logs" / "run.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("ok")

    visual_status = {
        "classical_present": True,
        "figures": ["a.png"],
        "required_paper_sample": 0,
    }
    rows = {"hybrid_resnet": {}, "ffno": {}}
    provenance_paths = {
        "runtime_provenance_path": str(output_root / "runtime_provenance.json"),
        "dataset_identity_manifest_path": str(
            output_root / "dataset_identity_manifest.json"
        ),
        "split_manifest_path": str(output_root / "split_manifest.json"),
    }

    # Failed status: must fail exit_code_proof even with PID match.
    (output_root / "run_exit_status.json").write_text(
        json.dumps(
            {"tracked_pid": 4242, "exit_code": 1, "status": "failed"}
        )
    )
    checks = paper_mod._build_provenance_checks(
        output_root=output_root,
        provenance_paths=provenance_paths,
        visual_status=visual_status,
        rows=rows,
        log_path=log_path,
    )
    assert checks["exit_code_proof"] is False

    # Completed status but non-zero exit_code: must still fail.
    (output_root / "run_exit_status.json").write_text(
        json.dumps(
            {"tracked_pid": 4242, "exit_code": 2, "status": "completed"}
        )
    )
    checks = paper_mod._build_provenance_checks(
        output_root=output_root,
        provenance_paths=provenance_paths,
        visual_status=visual_status,
        rows=rows,
        log_path=log_path,
    )
    assert checks["exit_code_proof"] is False

    # Completed + exit_code 0 + matching PID: must pass.
    (output_root / "run_exit_status.json").write_text(
        json.dumps(
            {"tracked_pid": 4242, "exit_code": 0, "status": "completed"}
        )
    )
    checks = paper_mod._build_provenance_checks(
        output_root=output_root,
        provenance_paths=provenance_paths,
        visual_status=visual_status,
        rows=rows,
        log_path=log_path,
    )
    assert checks["exit_code_proof"] is True


def test_same_contract_lineage_check_detects_split_count_drift(tmp_path):
    """Reviewer-blocked defect: ``same_contract_lineage`` previously only
    re-checked lineage-root pointers and the dataset id. The strengthened
    helper must reject a current bundle whose split counts, fixed-sample
    roster, operator geometry, normalization, or training-contract fields
    drift away from the frozen baseline lineage."""
    from scripts.studies.born_rytov_dt import (
        run_brdt_40ep_paper_evidence as paper_mod,
    )

    baseline_root = _make_synthetic_baseline_bundle(tmp_path)
    ffno_extension_root = _make_synthetic_ffno_extension_bundle(
        tmp_path, baseline_root
    )

    output_root = tmp_path / "current_bundle"
    output_root.mkdir()
    baseline_manifest = json.loads(
        (baseline_root / "preflight_manifest.json").read_text()
    )
    current_manifest = {
        "backlog_item": paper_mod.BACKLOG_ITEM,
        "baseline_lineage": {
            "baseline_root": str(baseline_root),
            "ffno_extension_root": str(ffno_extension_root),
        },
        "dataset": dict(baseline_manifest["dataset"]),
        "operator": dict(baseline_manifest["operator"]),
        "fixed_sample_ids": list(baseline_manifest["fixed_sample_ids"]),
        "training_contract": dict(baseline_manifest["training_contract"]),
        "rows": [
            {"row_id": "hybrid_resnet", "input_mode": "born_init_image"},
            {"row_id": "ffno", "input_mode": "born_init_image"},
        ],
    }
    manifest_path_top = output_root / "preflight_manifest.json"
    manifest_path_top.write_text(json.dumps(current_manifest))
    assert paper_mod._check_same_contract_lineage(
        output_root=output_root,
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
    )

    # Drift split counts: must fail.
    drifted = json.loads(manifest_path_top.read_text())
    drifted["dataset"] = dict(drifted["dataset"])
    drifted["dataset"]["split_counts"] = {"train": 1024, "val": 128, "test": 128}
    manifest_path_top.write_text(json.dumps(drifted))
    assert not paper_mod._check_same_contract_lineage(
        output_root=output_root,
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
    )

    # Drift fixed_sample_ids: must fail.
    drifted = json.loads(json.dumps(current_manifest))
    drifted["fixed_sample_ids"] = [1, 2, 3, 4]
    manifest_path_top.write_text(json.dumps(drifted))
    assert not paper_mod._check_same_contract_lineage(
        output_root=output_root,
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
    )

    # Drift operator geometry: must fail.
    drifted = json.loads(json.dumps(current_manifest))
    drifted["operator"] = dict(drifted["operator"])
    drifted["operator"]["geometry"] = dict(drifted["operator"]["geometry"])
    drifted["operator"]["geometry"]["angle_count"] = 32
    manifest_path_top.write_text(json.dumps(drifted))
    assert not paper_mod._check_same_contract_lineage(
        output_root=output_root,
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
    )

    # Drift training-contract loss weights: must fail.
    drifted = json.loads(json.dumps(current_manifest))
    drifted["training_contract"] = dict(drifted["training_contract"])
    drifted["training_contract"]["loss_weights"] = dict(
        drifted["training_contract"]["loss_weights"]
    )
    drifted["training_contract"]["loss_weights"]["physics"] = 9.99
    manifest_path_top.write_text(json.dumps(drifted))
    assert not paper_mod._check_same_contract_lineage(
        output_root=output_root,
        baseline_root=baseline_root,
        ffno_extension_root=ffno_extension_root,
    )
