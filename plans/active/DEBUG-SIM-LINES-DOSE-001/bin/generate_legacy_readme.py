#!/usr/bin/env python
"""Generate a maintainer-ready README for the dose_experiments ground-truth bundle.

This CLI loads the Phase-A manifest JSON and baseline summary, then emits a README.md
documenting the simulate->train->infer flow so Maintainer <2> can rerun dose_experiments
without touching production code.

Spec references:
- specs/data_contracts.md: RawData NPZ key requirements
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md: Phase B checklist

Usage:
    python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/generate_legacy_readme.py \
        --manifest plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001018Z/ground_truth_manifest.json \
        --baseline-summary plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.json \
        --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T001931Z
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """Load and return JSON from path, raising argparse error if file is missing."""
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable size (MB with 2 decimals)."""
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def format_sha_short(sha256: str) -> str:
    """Return first 12 chars of SHA256 for table display."""
    return sha256[:12] + "..."


def extract_photon_dose(filename: str) -> str:
    """Extract photon dose label from dataset filename (e.g., data_p1e5.npz -> 1e5)."""
    # Pattern: data_p1e<N>.npz
    base = Path(filename).stem  # data_p1e5
    if base.startswith("data_p"):
        return base[6:]  # 1e5
    return "unknown"


def build_readme(
    manifest: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> str:
    """Build the README markdown content from manifest and baseline summary."""
    lines: list[str] = []

    # Header
    lines.append("# Dose Experiments Ground-Truth Bundle")
    lines.append("")
    lines.append(f"**Scenario ID:** {manifest.get('scenario_id', 'N/A')}")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"**Dataset Root:** `{manifest.get('dataset_root', 'N/A')}`")
    lines.append("")

    # Section 1: Overview
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("This bundle contains the artifacts from a legacy `dose_experiments` ")
    lines.append("simulate->train->infer run. It is provided so Maintainer <2> can ")
    lines.append("compare outputs against new implementations without re-running the ")
    lines.append("legacy TF/Keras 2.x pipeline.")
    lines.append("")
    lines.append("**Pipeline stages:**")
    lines.append("1. **Simulation:** Generate diffraction patterns at various photon doses")
    lines.append("2. **Training:** Train baseline model on simulated data")
    lines.append("3. **Inference:** Reconstruct amplitude/phase from trained model")
    lines.append("")

    # Key parameters from baseline
    key_params = baseline_summary.get("key_params", {})
    metrics = baseline_summary.get("metrics", {})

    lines.append("**Key Parameters:**")
    lines.append(f"- N (patch size): {key_params.get('N', 'N/A')}")
    lines.append(f"- gridsize: {key_params.get('gridsize', 'N/A')}")
    lines.append(f"- nepochs: {key_params.get('nepochs', 'N/A')}")
    lines.append(f"- batch_size: {key_params.get('batch_size', 'N/A')}")
    lines.append(f"- loss: NLL-only (nll_weight={key_params.get('nll_weight', 'N/A')}, "
                 f"mae_weight={key_params.get('mae_weight', 'N/A')})")
    lines.append(f"- probe.trainable: {key_params.get('probe.trainable', 'N/A')}")
    lines.append(f"- intensity_scale.trainable: {key_params.get('intensity_scale.trainable', 'N/A')}")
    lines.append(f"- intensity_scale_value (learned): {key_params.get('intensity_scale_value', 'N/A'):.2f}")
    lines.append("")

    lines.append("**Baseline Metrics (train / test):**")
    if metrics:
        ms_ssim = metrics.get("ms_ssim", [None, None])
        psnr = metrics.get("psnr", [None, None])
        lines.append(f"- MS-SSIM: {ms_ssim[0]:.4f} / {ms_ssim[1]:.4f}")
        lines.append(f"- PSNR: {psnr[0]:.2f} dB / {psnr[1]:.2f} dB")
    lines.append("")

    # Section 2: Environment Requirements
    lines.append("## 2. Environment Requirements")
    lines.append("")
    lines.append("**IMPORTANT:** This pipeline was generated under TensorFlow/Keras 2.x. ")
    lines.append("Running under Keras 3.x will produce errors such as:")
    lines.append("")
    lines.append("```")
    lines.append("KerasTensor cannot be used as input to a TensorFlow function")
    lines.append("```")
    lines.append("")
    lines.append("**Recommended environment:**")
    lines.append("- Python 3.9 or 3.10")
    lines.append("- TensorFlow 2.12 - 2.15")
    lines.append("- Keras 2.x (bundled with TF 2.x)")
    lines.append("- NumPy < 2.0")
    lines.append("")
    lines.append("If you cannot set up this environment, use the pre-computed artifacts ")
    lines.append("in this bundle instead of re-running the pipeline.")
    lines.append("")

    # Section 3: Simulation Commands
    lines.append("## 3. Simulation Commands")
    lines.append("")
    lines.append("The canonical entry point is `notebooks/dose_dependence.ipynb`, which ")
    lines.append("invokes `notebooks/dose.py`:")
    lines.append("")
    lines.append("```python")
    lines.append("# In dose_dependence.ipynb cell:")
    lines.append("from notebooks import dose")
    lines.append("dose.init(nphotons=1e5, loss_fn='nll')")
    lines.append("# Simulation runs through dose.run_experiment_with_photons() internally")
    lines.append("```")
    lines.append("")
    lines.append("The `dose.init()` function configures:")
    lines.append("```python")
    lines.append("cfg['data_source'] = 'lines'")
    lines.append("cfg['gridsize'] = 2  # Note: dose.py default; baseline used gs=1")
    lines.append("cfg['intensity_scale.trainable'] = True")
    lines.append("cfg['probe.trainable'] = False")
    lines.append("cfg['nepochs'] = 60")
    lines.append("```")
    lines.append("")
    lines.append("**Note:** The baseline run in this bundle used `gridsize=1` and ")
    lines.append("`nepochs=50`, overriding the dose.py defaults.")
    lines.append("")

    # Section 4: Training Commands
    lines.append("## 4. Training Commands")
    lines.append("")
    lines.append("From repo root:")
    lines.append("")
    lines.append("```bash")
    lines.append("cd ~/Documents/PtychoPINN")
    lines.append("")
    lines.append("python -m ptycho.train \\")
    lines.append("    --train_data_file photon_grid_study_20250826_152459/data_p1e5.npz \\")
    lines.append("    --output_dir photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run \\")
    lines.append("    --batch_size 16 \\")
    lines.append("    --nepochs 50 \\")
    lines.append("    --gridsize 1 \\")
    lines.append("    --intensity_scale_trainable True \\")
    lines.append("    --probe_trainable False")
    lines.append("```")
    lines.append("")
    lines.append("**Overrides from dose.py defaults:**")
    lines.append("- `gridsize=1` (baseline; dose.py defaults to 2)")
    lines.append("- `batch_size=16`")
    lines.append("- `nepochs=50` (baseline; dose.py defaults to 60)")
    lines.append("")

    # Section 5: Inference Commands
    lines.append("## 5. Inference Commands")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m ptycho.inference \\")
    lines.append("    --model_path photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/08-26-2025-16.38.17_baseline_gs1/wts.h5.zip \\")
    lines.append("    --test_data photon_grid_study_20250826_152459/data_p1e5.npz")
    lines.append("```")
    lines.append("")

    # Section 6: Artifact Provenance Table
    lines.append("## 6. Artifact Provenance Table")
    lines.append("")
    lines.append("All sizes and SHA256 checksums are sourced from the Phase-A manifest ")
    lines.append("(`ground_truth_manifest.json`). NPZ files conform to ")
    lines.append("`specs/data_contracts.md` RawData NPZ requirements.")
    lines.append("")

    # Datasets table
    lines.append("### Datasets")
    lines.append("")
    lines.append("| File | Photon Dose | Size | SHA256 |")
    lines.append("|------|-------------|------|--------|")

    datasets = manifest.get("datasets", [])
    # Sort by photon dose (numeric extraction)
    def dose_key(d: dict) -> float:
        dose_str = extract_photon_dose(d.get("relative_path", ""))
        try:
            # Convert "1e5" to float for sorting
            return float(dose_str)
        except ValueError:
            return 0.0

    sorted_datasets = sorted(datasets, key=dose_key)

    for ds in sorted_datasets:
        rel_path = ds.get("relative_path", "N/A")
        filename = Path(rel_path).name
        dose = extract_photon_dose(filename)
        size = format_bytes(ds.get("size_bytes", 0))
        sha = format_sha_short(ds.get("sha256", "N/A"))
        lines.append(f"| `{filename}` | {dose} | {size} | `{sha}` |")

    lines.append("")

    # Baseline files table
    lines.append("### Baseline Artifacts")
    lines.append("")
    lines.append("| File | Type | Size | SHA256 |")
    lines.append("|------|------|------|--------|")

    # params.dill
    baseline_params = manifest.get("baseline_params", {})
    if baseline_params:
        params_path = Path(baseline_params.get("relative_path", "")).name or "params.dill"
        params_size = format_bytes(baseline_params.get("size_bytes", 0))
        params_sha = format_sha_short(baseline_params.get("sha256", "N/A"))
        lines.append(f"| `{params_path}` | params | {params_size} | `{params_sha}` |")

    # Other baseline files
    baseline_files = manifest.get("baseline_files", [])
    for bf in baseline_files:
        bf_path = Path(bf.get("relative_path", "")).name
        bf_type = bf.get("file_type", "output")
        bf_size = format_bytes(bf.get("size_bytes", 0))
        bf_sha = format_sha_short(bf.get("sha256", "N/A"))
        lines.append(f"| `{bf_path}` | {bf_type} | {bf_size} | `{bf_sha}` |")

    # PINN weights if present
    pinn_weights = manifest.get("pinn_weights", {})
    if pinn_weights:
        pinn_path = Path(pinn_weights.get("relative_path", "")).name or "wts.h5.zip"
        pinn_size = format_bytes(pinn_weights.get("size_bytes", 0))
        pinn_sha = format_sha_short(pinn_weights.get("sha256", "N/A"))
        lines.append(f"| `{pinn_path}` | pinn_weights | {pinn_size} | `{pinn_sha}` |")

    lines.append("")

    # Section 7: NPZ Key Requirements
    lines.append("## 7. NPZ Key Requirements")
    lines.append("")
    lines.append("Per `specs/data_contracts.md` RawData NPZ, each dataset NPZ must contain:")
    lines.append("")
    lines.append("**Required keys:**")
    lines.append("- `xcoords`, `ycoords` - scan positions (float64)")
    lines.append("- `xcoords_start`, `ycoords_start` - starting positions (float64, deprecated but present)")
    lines.append("- `diff3d` - diffraction patterns, shape (N_patterns, N, N), dtype float32")
    lines.append("- `probeGuess` - embedded probe, shape (N, N), dtype complex128")
    lines.append("- `scan_index` - scan indices (int64)")
    lines.append("")
    lines.append("**Optional keys:**")
    lines.append("- `objectGuess` - initial object guess")
    lines.append("- `ground_truth_patches` - ground truth for training/evaluation")
    lines.append("")

    # Array shapes from first dataset
    if sorted_datasets:
        first_ds = sorted_datasets[0]
        array_meta = first_ds.get("array_metadata", {})
        if array_meta:
            lines.append("**Array shapes (from first dataset):**")
            for key, meta in array_meta.items():
                shape = meta.get("shape", [])
                dtype = meta.get("dtype", "unknown")
                lines.append(f"- `{key}`: {shape} ({dtype})")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("**Manifest source:** `ground_truth_manifest.json`")
    lines.append(f"**Baseline summary:** `dose_baseline_summary.json`")
    lines.append("")
    lines.append("For questions, contact Maintainer <1> (dose_experiments branch).")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate README for dose_experiments ground-truth bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to ground_truth_manifest.json from Phase A",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        required=True,
        help="Path to dose_baseline_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for README.md",
    )

    args = parser.parse_args()

    # Validate input files exist (fail fast)
    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    if not args.baseline_summary.exists():
        print(f"ERROR: Baseline summary not found: {args.baseline_summary}", file=sys.stderr)
        return 1

    # Load inputs
    print(f"Loading manifest: {args.manifest}")
    manifest = load_json(args.manifest)

    print(f"Loading baseline summary: {args.baseline_summary}")
    baseline_summary = load_json(args.baseline_summary)

    # Build README content
    print("Generating README content...")
    readme_content = build_readme(manifest, baseline_summary)

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write README
    readme_path = args.output_dir / "README.md"
    print(f"Writing README to: {readme_path}")
    readme_path.write_text(readme_content, encoding="utf-8")

    # Summary
    print()
    print("=== README Generation Complete ===")
    print(f"Scenario ID: {manifest.get('scenario_id', 'N/A')}")
    print(f"Datasets: {len(manifest.get('datasets', []))}")
    print(f"Output: {readme_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
