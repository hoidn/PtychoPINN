"""Deterministic generic-schema twins of grid-lines dictionary NPZ pairs.

Bridge-ladder materialization tool (plan Task 21, handoff step 1): re-express
a grid-lines dictionary-schema pair (``diffraction``/``coords_*``/``YY_*``)
in the generic mmap-loader schema (``diff3d``/``xcoords``/``ycoords``/
``probeGuess``/``objectGuess``). The mapping is purely mechanical:

- ``diff3d = diffraction[..., 0]`` — measurement VALUES pass through
  bit-identically (same dtype, no arithmetic), so the conversion can never
  hide a normalization difference. Works for both the normalized-amplitude
  pair and the aligned count twin (``make_aligned_count_twin.py`` output).
- global scan positions come from ``coords_offsets + coords_nominal`` with
  the grid-lines channel convention (index 0 = y = rows, index 1 = x = cols;
  see ``grid_lines_workflow.simulate_grid_data``), emitted as the loader's
  ``ycoords``/``xcoords``.
- ``objectGuess``: the test split carries ``YY_ground_truth`` squeezed to 2-D
  (the fixture metric frame); the train split carries the mmap loader's
  all-ones "no object" placeholder (``PtychoDataset`` requires the key in
  every file and treats sum == H*W as absent).

Output NPZs are byte-deterministic (fixed zip metadata, pickle-free), same
technique as ``ci_compat_materializer_lib.npz_bytes``.

Provenance always describes the stored/transformed dictionary probe and the
emitted generic ``probeGuess`` arrays with their exact shape and dtype plus the
repository canonical array hash. When ``--probe-archive`` is supplied, it also
fingerprints the archive bytes and its raw ``probeGuess`` array; the archive is
optional because the generic converter also supports dictionary pairs whose
probe transform source is unavailable or not archive-backed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.studies.ablation.dataset_provenance import (  # noqa: E402
    canonical_array_sha256,
)

GENERIC_SCHEMA_TWIN_TOOL_ID = "grid_lines_generic_schema_twin_v1"


class ConversionError(RuntimeError):
    """Raised when a source pair cannot be converted faithfully."""


def _probe_descriptor(npz_path: Path, *, label: str) -> dict[str, Any]:
    """Read one NPZ probeGuess and return its exact identity descriptor."""
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            probe = np.asarray(archive["probeGuess"])
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise ConversionError(
            f"{label} {npz_path}: cannot read probeGuess: {exc}"
        ) from exc
    return {
        "canonical_sha256": canonical_array_sha256(probe),
        "shape": [int(size) for size in probe.shape],
        "dtype": str(probe.dtype),
    }


def _probe_archive_provenance(probe_archive: Path | None) -> dict[str, str] | None:
    """Compute optional raw probe-archive lineage without trusted pins."""
    if probe_archive is None:
        return None
    path = Path(probe_archive)
    if not path.is_file():
        raise ConversionError(f"input probe archive is missing: {path}")
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "probe_key": "probeGuess",
        "raw_probe_array_canonical_sha256": _probe_descriptor(
            path, label="input probe archive"
        )["canonical_sha256"],
    }


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize an NPZ with fixed zip metadata for byte determinism."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for key, value in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload.getvalue())
    return buffer.getvalue()


def _global_positions(data: Any, source: Path) -> np.ndarray:
    """Return (M, 2) global (y, x) scan positions from the dictionary pair."""
    offsets = data.get("coords_offsets")
    if offsets is None:
        raise ConversionError(
            f"{source}: dictionary pair lacks coords_offsets (global scan "
            "positions); cannot derive xcoords/ycoords"
        )
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.ndim != 4 or offsets.shape[1] != 1 or offsets.shape[2] != 2:
        raise ConversionError(
            f"{source}: coords_offsets shape {offsets.shape} is not (M, 1, 2, C)"
        )
    if offsets.shape[3] != 1:
        raise ConversionError(
            f"{source}: coords_offsets carries {offsets.shape[3]} channels; the "
            "generic twin is materialized from per-position (gridsize=1) data"
        )
    positions = offsets[:, 0, :, 0]
    nominal = data.get("coords_nominal")
    if nominal is not None:
        nominal = np.asarray(nominal, dtype=np.float64)
        if nominal.shape != offsets.shape:
            raise ConversionError(
                f"{source}: coords_nominal shape {nominal.shape} does not match "
                f"coords_offsets {offsets.shape}"
            )
        positions = positions + nominal[:, 0, :, 0]
    return positions


def convert_split(source: Path, destination: Path, *, split: str) -> dict[str, Any]:
    """Convert one dictionary-schema NPZ into its generic-schema twin."""
    source = Path(source)
    # Pickle-free reads: every consumed key is a plain array; _metadata (the
    # only object-typed key) is deliberately never accessed.
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files if key != "_metadata"}
    diffraction = np.asarray(data.get("diffraction"))
    if diffraction.ndim != 4 or diffraction.shape[-1] != 1:
        raise ConversionError(
            f"{source}: diffraction must be (M, H, W, 1); got "
            f"{None if data.get('diffraction') is None else diffraction.shape}"
        )
    probe = data.get("probeGuess")
    if probe is None:
        raise ConversionError(f"{source}: dictionary pair lacks probeGuess")
    positions = _global_positions(data, source)
    if positions.shape[0] != diffraction.shape[0]:
        raise ConversionError(
            f"{source}: {positions.shape[0]} scan positions do not match "
            f"{diffraction.shape[0]} diffraction patterns"
        )
    if split == "test":
        truth = data.get("YY_ground_truth")
        if truth is None:
            truth = data.get("YY_full")
        if truth is None:
            raise ConversionError(
                f"{source}: test split must carry YY_ground_truth or YY_full"
            )
        object_guess = np.squeeze(np.asarray(truth))
        if object_guess.ndim != 2:
            raise ConversionError(
                f"{source}: ground truth must squeeze to 2-D; got "
                f"{object_guess.shape}"
            )
    else:
        # PtychoDataset requires objectGuess in every file; all-ones with
        # sum == H*W is its documented "no object" placeholder.
        object_guess = np.ones(diffraction.shape[1:3], dtype=np.complex64)
    arrays = {
        "diff3d": np.ascontiguousarray(diffraction[..., 0]),
        "xcoords": np.ascontiguousarray(positions[:, 1]),
        "ycoords": np.ascontiguousarray(positions[:, 0]),
        "probeGuess": np.asarray(probe),
        "objectGuess": np.ascontiguousarray(object_guess),
    }
    payload = _npz_bytes(arrays)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output": str(destination),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "n_positions": int(diffraction.shape[0]),
        "detector_shape": [int(diffraction.shape[1]), int(diffraction.shape[2])],
    }


def convert_pair(
    train_npz: Path,
    test_npz: Path,
    output_dir: Path,
    *,
    probe_archive: Path | None = None,
) -> dict[str, Any]:
    """Convert a train/test dictionary pair; write twins plus provenance."""
    train_npz = Path(train_npz)
    test_npz = Path(test_npz)
    output_dir = Path(output_dir)
    archive_provenance = _probe_archive_provenance(probe_archive)
    source_probe_descriptors = {
        "train": _probe_descriptor(train_npz, label="dictionary source train"),
        "test": _probe_descriptor(test_npz, label="dictionary source test"),
    }
    source_splits_equal = (
        source_probe_descriptors["train"] == source_probe_descriptors["test"]
    )
    if not source_splits_equal:
        raise ConversionError(
            "source train/test probeGuess mismatch: "
            f"{source_probe_descriptors['train']} != "
            f"{source_probe_descriptors['test']}"
        )

    provenance: dict[str, Any] = {"tool": GENERIC_SCHEMA_TWIN_TOOL_ID}
    provenance["train"] = convert_split(
        train_npz, output_dir / "train.npz", split="train"
    )
    provenance["test"] = convert_split(
        test_npz, output_dir / "test.npz", split="test"
    )
    output_probe_descriptors = {
        split: _probe_descriptor(
            output_dir / f"{split}.npz", label=f"generic output {split}"
        )
        for split in ("train", "test")
    }
    output_splits_equal = (
        output_probe_descriptors["train"] == output_probe_descriptors["test"]
    )
    output_equal = all(
        output_probe_descriptors[split] == source_probe_descriptors[split]
        for split in ("train", "test")
    )
    for split in ("train", "test"):
        if output_probe_descriptors[split] != source_probe_descriptors[split]:
            raise ConversionError(
                f"generic output {split} probeGuess mismatch: "
                f"{output_probe_descriptors[split]} != dictionary source "
                f"{source_probe_descriptors[split]}"
            )
    provenance["probe_lineage"] = {
        "input_probe_archive": archive_provenance,
        "dictionary_source_stored_transformed": {
            "probe_key": "probeGuess",
            "train": source_probe_descriptors["train"],
            "test": source_probe_descriptors["test"],
            "splits_equal": source_splits_equal,
        },
        "generic_output": {
            "probe_key": "probeGuess",
            "train": output_probe_descriptors["train"],
            "test": output_probe_descriptors["test"],
            "splits_equal": output_splits_equal,
            "output_equal": output_equal,
        },
    }
    (output_dir / "generic_twin_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", required=True, type=Path)
    parser.add_argument("--test-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--probe-archive",
        type=Path,
        help=(
            "optional raw probe NPZ; records computed archive-byte and "
            "probeGuess canonical hashes in provenance"
        ),
    )
    args = parser.parse_args(argv)
    try:
        provenance = convert_pair(
            args.train_npz,
            args.test_npz,
            args.output_dir,
            probe_archive=args.probe_archive,
        )
    except ConversionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for split in ("train", "test"):
        info = provenance[split]
        print(
            f"{split}: n={info['n_positions']} sha256={info['output_sha256']} "
            f"-> {info['output']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
