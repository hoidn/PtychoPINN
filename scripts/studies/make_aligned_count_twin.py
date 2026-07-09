"""Invert the grid-lines amplitude normalization into count-convention twins.

The grid-lines simulation draws Poisson counts at `nphotons` per pattern, then
divides amplitudes by an unsaved `intensity_scale` (`ptycho/diffsim.py`). That
scale is recoverable because it was chosen to satisfy the photon budget:

    S = sqrt(nphotons / mean_over_patterns(sum(diffraction**2)))

Multiplying the stored (normalized) amplitudes by the per-split S recovers the
physical count-amplitudes actually drawn. This tool writes two twin datasets:

    data_amp/{split}.npz       diffraction = (diffraction * S)      (sqrt-counts)
    data_intensity/{split}.npz diffraction = (diffraction * S) ** 2 (counts)

and a `provenance.json` recording the per-split S, nphotons, and source info.
All non-diffraction keys are copied verbatim.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptycho.metadata import MetadataManager

DEFAULT_NPHOTONS = 1.0e9
SPLITS = ("train", "test")
S_FORMULA = "S = sqrt(nphotons / mean_over_patterns(sum(diffraction**2)))"


def _read_nphotons(data: dict, src_path: Path) -> float:
    """Read nphotons from `_metadata`, defaulting to 1e9 with a warning."""
    if "_metadata" not in data:
        warnings.warn(f"{src_path}: no '_metadata' key; defaulting nphotons={DEFAULT_NPHOTONS:.0e}")
        return DEFAULT_NPHOTONS

    raw = data["_metadata"]
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, (bytes, str)):
        raw = json.loads(raw)

    nphotons = None
    if isinstance(raw, dict):
        nphotons = raw.get("nphotons")
        if nphotons is None:
            physics_params = raw.get("physics_parameters", {})
            if isinstance(physics_params, dict) and "nphotons" in physics_params:
                nphotons = MetadataManager.get_nphotons(raw, default=DEFAULT_NPHOTONS)

    if nphotons is None:
        warnings.warn(f"{src_path}: no nphotons in '_metadata'; defaulting nphotons={DEFAULT_NPHOTONS:.0e}")
        return DEFAULT_NPHOTONS
    return float(nphotons)


def _recover_scale(diffraction: np.ndarray, nphotons: float) -> float:
    """S = sqrt(nphotons / mean_over_patterns(sum(diffraction**2)))."""
    if diffraction.ndim != 4:
        raise ValueError(f"'diffraction' must be 4D (n,H,W,1), got shape {diffraction.shape}")
    per_pattern_total = np.sum(diffraction.astype(np.float64) ** 2, axis=(1, 2, 3))
    mean_total = per_pattern_total.mean()
    return float(np.sqrt(nphotons / mean_total))


def _convert_split(src_dir: Path, split: str, out_root: Path) -> dict[str, object]:
    src_path = src_dir / f"{split}.npz"
    with np.load(src_path, allow_pickle=True) as npz:
        data = {k: npz[k] for k in npz.files}

    if "diffraction" not in data:
        raise KeyError(f"{src_path}: missing 'diffraction' key")
    diffraction = data["diffraction"]

    nphotons = _read_nphotons(data, src_path)
    S = _recover_scale(diffraction, nphotons)
    scaled = diffraction.astype(np.float64) * S

    amp_data = dict(data)
    amp_data["diffraction"] = scaled.astype(np.float32)
    intensity_data = dict(data)
    intensity_data["diffraction"] = (scaled ** 2).astype(np.float32)

    amp_path = out_root / "data_amp" / f"{split}.npz"
    intensity_path = out_root / "data_intensity" / f"{split}.npz"
    amp_path.parent.mkdir(parents=True, exist_ok=True)
    intensity_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(amp_path, **amp_data)
    np.savez(intensity_path, **intensity_data)

    intensity_totals = np.sum(intensity_data["diffraction"].astype(np.float64), axis=(1, 2, 3))
    return {
        "S": S,
        "nphotons": nphotons,
        "source_path": str(src_path),
        "source_mtime": src_path.stat().st_mtime,
        "mean_per_pattern_total_intensity": float(intensity_totals.mean()),
        "max_intensity": float(intensity_data["diffraction"].max()),
        "formula": S_FORMULA,
    }


def build_twins(src_dir: Path, out_root: Path) -> dict:
    provenance: dict[str, object] = {"formula": S_FORMULA}
    for split in SPLITS:
        provenance[split] = _convert_split(src_dir, split, out_root)

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", required=True, type=Path,
                         help="Directory containing train.npz and test.npz")
    parser.add_argument("--out-root", required=True, type=Path,
                         help="Output root for data_amp/, data_intensity/, provenance.json")
    args = parser.parse_args()

    provenance = build_twins(args.src_dir, args.out_root)
    for split in SPLITS:
        info = provenance[split]
        print(f"{split}: S={info['S']:.6g} nphotons={info['nphotons']:.6g} "
              f"mean_total_intensity={info['mean_per_pattern_total_intensity']:.6g} "
              f"max_intensity={info['max_intensity']:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
