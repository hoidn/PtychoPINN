"""Era-detecting offline migration for PyTorch ``wts.h5.zip`` bundles.

``python -m ptycho_torch.migrate_bundle SRC OUT`` migrates one bundle directory:

- dill-era / unsealed bundles (``manifest.dill`` or a JSON manifest without an
  ``artifact_schema_version``) are delegated to the heavy legacy migrator in
  ``scripts/migrate_legacy_bundle.py``; when that migrator preserves a
  historical sealed identity, the produced bundle is re-encoded at the current
  era in the same pass.
- versioned ``torch-artifact-portable-v1..v3`` bundles are strict-loaded and
  re-encoded at the current ``torch-artifact-portable-v4`` era through
  ``bundle_io``/``artifact_schema``.
- an already-current ``torch-artifact-portable-v4`` bundle raises: there is
  nothing to migrate.

The heavy machinery (torch, the artifact schema, and the legacy migrator) is
lazy-imported only once an actual migration is dispatched.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

_DILL_MANIFEST_MEMBER = "manifest.dill"
_JSON_MANIFEST_MEMBER = "manifest.json"
_SCALING_METADATA_MEMBER = "torch_scaling_metadata.pt"


def _era_of(manifest: dict) -> str:
    """Return the manifest's artifact era, or ``"legacy"`` when unsealed."""
    schema = manifest.get("artifact_schema_version")
    backend = manifest.get("backend")
    if backend is None or schema is None:
        return "legacy"
    return schema


def migrate_bundle(source_dir: Path, out_dir: Path) -> Path:
    """Migrate one bundle directory; return the migrated ``wts.h5.zip`` path."""
    source_zip = Path(source_dir) / "wts.h5.zip"
    if not source_zip.is_file():
        raise FileNotFoundError(f"No wts.h5.zip in {source_dir}")

    out_dir = Path(out_dir)

    with zipfile.ZipFile(source_zip, "r") as archive:
        names = set(archive.namelist())

    if _DILL_MANIFEST_MEMBER in names or _JSON_MANIFEST_MEMBER not in names:
        return _migrate_legacy_then_reencode(source_dir, out_dir)

    with zipfile.ZipFile(source_zip, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}

    manifest = json.loads(members[_JSON_MANIFEST_MEMBER])
    if _era_of(manifest) == "legacy":
        return _migrate_legacy_then_reencode(source_dir, out_dir)

    return _reencode_to_current(source_zip, out_dir)


def _reencode_to_current(source_zip: Path, out_dir: Path) -> Path:
    """Re-encode a versioned v1..v3 bundle at the current v4 era."""
    from ptycho_torch.artifact_schema import (
        CURRENT_ARTIFACT_SCHEMA_VERSION,
        SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
        decode_artifact_identity,
        encode_artifact_identity,
    )

    with zipfile.ZipFile(source_zip, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}

    manifest = json.loads(members[_JSON_MANIFEST_MEMBER])
    era = _era_of(manifest)

    if era == CURRENT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"{source_zip} is already at {era}; nothing to migrate.")
    if era not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported wts.h5.zip artifact schema {era!r}")

    if _SCALING_METADATA_MEMBER not in members:
        raise ValueError(
            f"{source_zip} carries no sealed {_SCALING_METADATA_MEMBER} to re-encode."
        )

    import torch

    metadata = torch.load(
        io.BytesIO(members[_SCALING_METADATA_MEMBER]),
        map_location="cpu",
        weights_only=True,
    )
    decoded = decode_artifact_identity(metadata)
    payload = encode_artifact_identity(
        decoded.model_spec,
        decoded.data_config,
        decoded.training_config,
        decoded.inference_config,
        ci_statistics=decoded.ci_statistics,
    )

    new_manifest = dict(manifest)
    new_manifest["artifact_schema_version"] = CURRENT_ARTIFACT_SCHEMA_VERSION

    buffer = io.BytesIO()
    torch.save(payload, buffer)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip = out_dir / "wts.h5.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_JSON_MANIFEST_MEMBER, json.dumps(new_manifest))
        archive.writestr(_SCALING_METADATA_MEMBER, buffer.getvalue())
        for name, content in members.items():
            if name in (_JSON_MANIFEST_MEMBER, _SCALING_METADATA_MEMBER):
                continue
            archive.writestr(name, content)
    return out_zip


def _migrate_legacy_then_reencode(source_dir: Path, out_dir: Path) -> Path:
    """Delegate to the legacy migrator, then re-encode any historical era."""
    out_zip = _migrate_legacy(source_dir, out_dir)

    from ptycho_torch.artifact_schema import CURRENT_ARTIFACT_SCHEMA_VERSION

    with zipfile.ZipFile(out_zip, "r") as archive:
        manifest = json.loads(archive.read(_JSON_MANIFEST_MEMBER))
    if _era_of(manifest) == CURRENT_ARTIFACT_SCHEMA_VERSION:
        return out_zip  # legacy migrator already sealed a current identity
    return _reencode_to_current(out_zip, out_dir)


def _migrate_legacy(source_dir: Path, out_dir: Path) -> Path:
    """Delegate dill-era / unsealed bundles to the heavy legacy migrator."""
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.migrate_legacy_bundle import migrate_bundle as _legacy_migrate

    return _legacy_migrate(Path(source_dir), Path(out_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate a PyTorch wts.h5.zip bundle to the current "
        "torch-artifact-portable-v4 era."
    )
    parser.add_argument(
        "source_dir", type=Path, help="directory holding the source wts.h5.zip"
    )
    parser.add_argument(
        "out_dir", type=Path, help="directory receiving the migrated wts.h5.zip"
    )
    args = parser.parse_args()
    out_zip = migrate_bundle(args.source_dir, args.out_dir)
    print(f"migrated bundle written to {out_zip}")


if __name__ == "__main__":
    main()
