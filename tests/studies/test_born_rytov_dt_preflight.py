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
