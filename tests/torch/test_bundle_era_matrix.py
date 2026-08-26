"""Bundle-era x load-path compatibility matrix.

Every (era x production load path) cell from the Phase 0 contract is asserted
here: a cell either load-succeeds with a sane decoded identity (``gridsize`` and
``N`` present) or raises the typed schema/migration error. The matrix is the
regression harness for R1 (channel-identity faithfulness), R2 (era acceptance
— v1/v2 load on the checkpoint-decode and migrator paths, while the
``bundle_io`` runtime path rejects them naming the migrator), and R3 (migrator
re-sealing pre-v3 identities to v4 without retired config fields).

Load-path levels exercised per cell:

- **P1** (bundle_io runtime path): full ``load_inference_bundle_torch`` for v4
  and ci-entrypoints-v1 (real weights exist); metadata-decode level
  (``_decode_bundle_metadata`` on the archive's sealed payload) for v3, and for
  synthesized v1/v2/unfaithful (rejected by the runtime gate naming the
  migrator, which carry no model weights); full manifest read for the dill-era
  bundle (rejected before any weights are touched).
- **P2** (checkpoint decode): ``decode_checkpoint_hparams`` on the dual-written
  old-era sections (v1/v2/unfaithful) or the sealed ``artifact_identity``
  (v3/v4).
- **P3** (migrator): ``scripts.migrate_legacy_bundle.migrate_bundle``, with the
  migrated output re-decoded to prove it is a valid v4 bundle.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import era_fixtures as ef


def _checkpoint_hparams(payload: dict) -> dict:
    """Reconstruct the old-era dual-write hyper_parameters dict from a payload."""
    model_spec = payload["model_spec"]
    return {
        "model_spec": model_spec,
        "model_config": model_spec["model_config"],
        "data_config": payload["data_config"],
        "training_config": payload["training_config"],
        "inference_config": payload["inference_config"],
        "parity_scale_mode": model_spec["parity_scale_mode"],
        "parity_fixed_delta": model_spec["parity_fixed_delta"],
        "parity_init_scheme": model_spec["parity_init_scheme"],
    }


# ---------------------------------------------------------------------------
# Path runners: each returns (gridsize, N) on success or raises the typed error.
# ---------------------------------------------------------------------------

def _run_p1(era: str, tmp_path: Path):
    from ptycho_torch.workflows.bundle_io import _decode_bundle_metadata
    from ptycho_torch.workflows.components import load_inference_bundle_torch

    if era == "metadata-free-legacy":
        # Full manifest read; the dill-era bundle is rejected by naming the
        # migration script before any model reconstruction.
        load_inference_bundle_torch(ef.metadata_free_bundle(tmp_path))
        raise AssertionError("metadata-free bundle unexpectedly loaded")
    if era in ("ci-entrypoints-v1", "ci-entrypoints-v1-legacy-shaped"):
        # Full load: real weights + transitional metadata upgrade in-place.
        bundle_builder = (
            ef.ci_entrypoints_bundle
            if era == "ci-entrypoints-v1"
            else ef.ci_entrypoints_legacy_shaped_bundle
        )
        models, _ = load_inference_bundle_torch(bundle_builder(tmp_path))
        model = models["diffraction_to_obj"]
        return model.data_config.gridsize, model.data_config.N
    if era == "v5":
        # Full load: real save + sealed v5 identity.
        models, _ = load_inference_bundle_torch(ef.v5_bundle(tmp_path))
        model = models["diffraction_to_obj"]
        return model.data_config.gridsize, model.data_config.N
    if era == "unfaithful-unversioned":
        # Full load: legacy-shaped transitional metadata with disagreeing C.
        load_inference_bundle_torch(ef.unfaithful_unversioned_bundle(tmp_path))
        raise AssertionError("unfaithful unversioned bundle unexpectedly loaded")
    if era in ("v1", "v2", "unfaithful-legacy"):
        # Runtime gate: pre-v3 schemas are rejected naming the migrator before
        # any channel-faithfulness check runs.
        bundle_builder = {
            "v1": ef.v1_bundle,
            "v2": ef.v2_bundle,
            "unfaithful-legacy": ef.unfaithful_bundle,
        }[era]
        _decode_bundle_metadata(ef.read_bundle_metadata(bundle_builder(tmp_path)))
        raise AssertionError(f"{era} bundle unexpectedly decoded by the runtime path")
    if era in ("v3-c4", "v4-c4"):
        # C>1 pre-v5 bundles fail the centered-nearest gate at runtime decode.
        bundle_builder = {"v3-c4": ef.v3_c4_bundle, "v4-c4": ef.v4_c4_bundle}[era]
        _decode_bundle_metadata(ef.read_bundle_metadata(bundle_builder(tmp_path)))
        raise AssertionError(f"{era} C>1 bundle unexpectedly decoded")
    # v3/v4: metadata-decode level (no weights); they upgrade to v5.
    bundle_builder = {"v3": ef.v3_bundle, "v4": ef.v4_bundle}[era]
    identity = _decode_bundle_metadata(ef.read_bundle_metadata(bundle_builder(tmp_path)))
    return identity.data_config.gridsize, identity.data_config.N


def _run_p2(era: str, tmp_path: Path):
    from ptycho_torch.checkpoint_decode import decode_checkpoint_hparams

    if era in ("v3", "v4", "v5", "v3-c4", "v4-c4"):
        payload = {
            "v3": ef.v3_payload,
            "v4": ef.v4_payload,
            "v5": ef.v5_payload,
            "v3-c4": ef.v3_c4_payload,
            "v4-c4": ef.v4_c4_payload,
        }[era]()
        decode_checkpoint_hparams({"artifact_identity": payload})
        return payload["data_config"]["gridsize"], payload["data_config"]["N"]
    if era == "unfaithful-legacy":
        decode_checkpoint_hparams(_checkpoint_hparams(ef.unfaithful_v1_payload()))
        raise AssertionError("unfaithful checkpoint unexpectedly decoded")
    # Old-era checkpoints: single-channel variants promote in memory;
    # multi-channel (C=4 frozen) payloads fail the centered-nearest gate.
    payload = {
        "v1": ef.v1_payload,
        "v2": ef.v2_payload,
        "v1-c1": ef.v1_c1_payload,
        "v2-c1": ef.v2_c1_payload,
    }[era]()
    decode_checkpoint_hparams(_checkpoint_hparams(payload))
    return payload["data_config"]["grid_size"][0], payload["data_config"]["N"]


def _run_p3(era: str, tmp_path: Path):
    from ptycho_torch.workflows.bundle_io import _decode_bundle_metadata
    from scripts.migrate_legacy_bundle import migrate_bundle

    bundle = _P3_BUNDLES[era](tmp_path)
    out_dir = tmp_path / f"migrated-{era}"
    migrate_bundle(bundle, out_dir)
    identity = _decode_bundle_metadata(ef.read_bundle_metadata(out_dir))
    return identity.data_config.gridsize, identity.data_config.N


_P3_BUNDLES = {
    "metadata-free-legacy": ef.metadata_free_bundle,
    "ci-entrypoints-v1": ef.ci_entrypoints_bundle,
    "ci-entrypoints-v1-legacy-shaped": ef.ci_entrypoints_legacy_shaped_bundle,
    "unfaithful-unversioned": ef.unfaithful_unversioned_bundle,
    "v1": ef.v1_bundle,
    "v2": ef.v2_bundle,
    "v1-c1": ef.v1_c1_bundle,
    "v2-c1": ef.v2_c1_bundle,
    "v3": ef.v3_bundle,
    "v4": ef.v4_bundle,
    "v3-c4": ef.v3_c4_bundle,
    "v4-c4": ef.v4_c4_bundle,
    "v5": ef.v5_bundle,
    "unfaithful-legacy": ef.unfaithful_bundle,
}

_RUNNERS = {"P1": _run_p1, "P2": _run_p2, "P3": _run_p3}

# (era, path, expect, match) — the full contract matrix.
_CELLS = [
    ("metadata-free-legacy", "P1", "reject", r"migrate_bundle"),
    ("metadata-free-legacy", "P3", "accept", None),
    ("ci-entrypoints-v1", "P1", "accept", None),
    ("ci-entrypoints-v1", "P3", "accept", None),
    ("ci-entrypoints-v1-legacy-shaped", "P1", "accept", None),
    ("ci-entrypoints-v1-legacy-shaped", "P3", "accept", None),
    ("v1", "P1", "reject", r"migrate_bundle"),
    ("v1", "P2", "reject", r"supports exactly one derived channel"),
    ("v1", "P3", "reject", r"supports exactly one derived channel"),
    ("v1-c1", "P2", "accept", None),
    ("v1-c1", "P3", "accept", None),
    ("v2", "P1", "reject", r"migrate_bundle"),
    ("v2", "P2", "reject", r"supports exactly one derived channel"),
    ("v2", "P3", "reject", r"supports exactly one derived channel"),
    ("v2-c1", "P2", "accept", None),
    ("v2-c1", "P3", "accept", None),
    ("v3", "P1", "accept", None),
    ("v3", "P2", "accept", None),
    ("v3", "P3", "accept", None),
    ("v4", "P1", "accept", None),
    ("v4", "P2", "accept", None),
    ("v4", "P3", "accept", None),
    ("v3-c4", "P1", "reject", r"supports exactly one derived channel"),
    ("v3-c4", "P2", "reject", r"supports exactly one derived channel"),
    ("v3-c4", "P3", "reject", r"supports exactly one derived channel"),
    ("v4-c4", "P1", "reject", r"supports exactly one derived channel"),
    ("v4-c4", "P2", "reject", r"supports exactly one derived channel"),
    ("v4-c4", "P3", "reject", r"supports exactly one derived channel"),
    ("v5", "P1", "accept", None),
    ("v5", "P2", "accept", None),
    ("v5", "P3", "accept", None),
    ("unfaithful-legacy", "P1", "reject", r"migrate_bundle"),
    ("unfaithful-legacy", "P2", "reject", r"gridsize\*\*2"),
    # Raw channel faithfulness is validated before the centered-nearest C1
    # gate, so the migrator's decode reports the stored-C disagreement.
    ("unfaithful-legacy", "P3", "reject", r"gridsize\*\*2"),
    ("unfaithful-unversioned", "P1", "reject", r"unversioned.*gridsize\*\*2"),
    ("unfaithful-unversioned", "P3", "reject", r"unversioned.*gridsize\*\*2"),
]


@pytest.mark.parametrize(
    "era,path,expect,match",
    _CELLS,
    ids=[f"{era} x {path}" for era, path, _, _ in _CELLS],
)
def test_bundle_era_matrix(era, path, expect, match, tmp_path):
    runner = _RUNNERS[path]
    if expect == "reject":
        with pytest.raises(ValueError, match=match):
            runner(era, tmp_path)
    else:
        gridsize, N = runner(era, tmp_path)
        assert isinstance(gridsize, int) and gridsize > 0
        assert isinstance(N, int) and N > 0


def test_metadata_free_p3_migrates_to_current_era(tmp_path):
    """P3 on a dill-era bundle promotes the manifest to torch-artifact-v5."""
    from scripts.migrate_legacy_bundle import migrate_bundle

    out_dir = tmp_path / "migrated"
    migrate_bundle(ef.metadata_free_bundle(tmp_path), out_dir)
    with zipfile.ZipFile(out_dir / "wts.h5.zip", "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["artifact_schema_version"] == "torch-artifact-v5"
    assert "manifest.dill" not in zipfile.ZipFile(out_dir / "wts.h5.zip").namelist()


def test_v3_bundle_upgrades_to_v5_fields(tmp_path):
    """A v3 bundle decodes under v5 CURRENT with the renamed field spellings."""
    from ptycho_torch.artifact_schema import CENTERED_NEAREST_GROUPING_CONTRACT
    from ptycho_torch.workflows.bundle_io import _decode_bundle_metadata

    identity = _decode_bundle_metadata(
        ef.read_bundle_metadata(ef.v3_bundle(tmp_path))
    )
    assert identity.grouping_contract == CENTERED_NEAREST_GROUPING_CONTRACT
    assert identity.data_config.neighbor_count == 7
    assert not hasattr(identity.data_config, "K")
    assert not hasattr(identity.data_config, "groups_per_center")
    assert identity.data_config.group_padding_step == 3.0
    assert identity.training_config.training_groups == 4
    assert not hasattr(identity.training_config, "n_groups")


def test_v2_manifest_with_v1_metadata_rejected_by_agreement_gate(tmp_path):
    """The manifest/metadata agreement gate fires for ALL versioned eras.

    Regression pin: the gate's local era set omitted v2, so a v2 manifest
    paired with v1 metadata skipped the agreement check entirely.
    """
    from ptycho_torch.workflows.components import load_inference_bundle_torch

    with pytest.raises(ValueError, match="disagree"):
        load_inference_bundle_torch(ef.mismatched_v2_bundle(tmp_path))
