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
