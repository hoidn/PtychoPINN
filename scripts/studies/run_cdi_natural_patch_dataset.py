"""CLI entrypoint for generating natural_patches128_fixedprobe_v1.

Examples
--------
Generate the full 8000/1000/1000 dataset under the canonical artifact root::

    python scripts/studies/run_cdi_natural_patch_dataset.py \\
        --dataset-root .artifacts/data/NEURIPS-HYBRID-RESNET-2026/natural_patches128_fixedprobe_v1

The default settings match the locked contract in the execution plan and the
durable summary at
``docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_fixedprobe_dataset_summary.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.studies.cdi_natural_patch_dataset import (
    DEFAULT_CROP_SEED,
    DEFAULT_DATASET_ID,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PROBE_SCALE_MODE,
    DEFAULT_PROBE_SMOOTHING_SIGMA,
    DEFAULT_PROBE_SOURCE,
    DEFAULT_SKIMAGE_SOURCE_NAMES,
    DEFAULT_SPLIT_COUNTS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_SPLIT_SOURCE_COUNTS,
    DEFAULT_TOTAL_CAP,
    build_dataset,
    load_skimage_corpus,
    post_audit,
    prepare_probe_from_run1084,
    render_contact_sheet,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build natural_patches128_fixedprobe_v1 dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Output directory (will be created). Should live under .artifacts/data/.",
    )
    parser.add_argument(
        "--probe-npz",
        type=Path,
        default=Path(DEFAULT_PROBE_SOURCE),
        help="Run1084 probe NPZ source.",
    )
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--total-cap", type=int, default=DEFAULT_TOTAL_CAP)
    parser.add_argument("--n-train", type=int, default=int(DEFAULT_SPLIT_COUNTS["train"]))
    parser.add_argument("--n-val", type=int, default=int(DEFAULT_SPLIT_COUNTS["val"]))
    parser.add_argument("--n-test", type=int, default=int(DEFAULT_SPLIT_COUNTS["test"]))
    parser.add_argument(
        "--source-train",
        type=int,
        default=int(DEFAULT_SPLIT_SOURCE_COUNTS["train"]),
        help="Number of source images allocated to the train split.",
    )
    parser.add_argument(
        "--source-val",
        type=int,
        default=int(DEFAULT_SPLIT_SOURCE_COUNTS["val"]),
        help="Number of source images allocated to the validation split.",
    )
    parser.add_argument(
        "--source-test",
        type=int,
        default=int(DEFAULT_SPLIT_SOURCE_COUNTS["test"]),
        help="Number of source images allocated to the test split.",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--crop-seed", type=int, default=DEFAULT_CROP_SEED)
    parser.add_argument("--dataset-id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--smoothing-sigma", type=float, default=DEFAULT_PROBE_SMOOTHING_SIGMA)
    parser.add_argument("--scale-mode", type=str, default=DEFAULT_PROBE_SCALE_MODE)
    parser.add_argument(
        "--source-names",
        type=str,
        default=",".join(DEFAULT_SKIMAGE_SOURCE_NAMES),
        help="Comma-separated scikit-image data names to use as source corpus.",
    )
    parser.add_argument(
        "--skip-contact-sheet",
        action="store_true",
        help="Skip contact-sheet rendering (e.g., during low-overhead retries).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    source_names = [name.strip() for name in args.source_names.split(",") if name.strip()]
    records = load_skimage_corpus(source_names, patch_size=int(args.patch_size))
    probe_bundle = prepare_probe_from_run1084(
        probe_npz_path=args.probe_npz,
        target_N=int(args.patch_size),
        smoothing_sigma=float(args.smoothing_sigma),
        scale_mode=str(args.scale_mode),
    )
    split_counts = {"train": int(args.n_train), "val": int(args.n_val), "test": int(args.n_test)}
    split_source_counts = {
        "train": int(args.source_train),
        "val": int(args.source_val),
        "test": int(args.source_test),
    }
    result = build_dataset(
        dataset_root=args.dataset_root,
        records=records,
        probe_bundle=probe_bundle,
        split_counts=split_counts,
        split_source_counts=split_source_counts,
        patch_size=int(args.patch_size),
        total_cap=int(args.total_cap),
        split_seed=int(args.split_seed),
        crop_seed=int(args.crop_seed),
        dataset_id=str(args.dataset_id),
    )
    if not args.skip_contact_sheet:
        render_contact_sheet(dataset_root=args.dataset_root)

    audit = post_audit(
        dataset_root=args.dataset_root,
        expected_split_counts=split_counts,
        total_cap=int(args.total_cap),
    )
    verification_dir = Path(args.dataset_root) / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    (verification_dir / "post_audit.json").write_text(json.dumps(audit, indent=2))

    print(json.dumps({"dataset_root": str(result.dataset_root), "audit": audit}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
