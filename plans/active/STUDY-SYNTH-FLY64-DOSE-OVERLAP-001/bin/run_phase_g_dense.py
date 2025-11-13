#!/usr/bin/env python3
"""
Phase G Dense Execution Orchestrator for STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.

Runs the complete Phase C→G pipeline (dataset generation → overlap views → training →
reconstruction → comparison) for a single dose/view/splits configuration, capturing all
CLI outputs to per-phase log files under the provided hub directory.

Enforces TYPE-PATH-001 (Path normalization), propagates AUTHORITATIVE_CMDS_DOC, and
halts execution on any non-zero subprocess return code with blocker log notes.

Usage:
    export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
    python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \\
        --hub plans/active/.../reports/<timestamp>/phase_g_execution_real_runs \\
        --dose 1000 \\
        --view dense \\
        --splits train test

Outputs:
    - {HUB}/data/phase_c/: Phase C dataset generation outputs
    - {HUB}/data/phase_d/: Phase D overlap view outputs
    - {HUB}/data/phase_e/: Phase E training outputs (manifests, bundles)
    - {HUB}/data/phase_f/: Phase F reconstruction outputs (manifests, recon NPZs)
    - {HUB}/analysis/: Comparison plots, metrics, and summaries
    - {HUB}/cli/phase_{c,d,e,f,g}_{split}.log: Per-phase CLI transcripts

If any command fails, execution halts and a blocker note is written to:
    {HUB}/analysis/blocker.log

DEPRECATION NOTICE
- Prefer the neutral entrypoint under `scripts/study/run_dense_pipeline.py` for
  invoking this pipeline. That script forwards to this runner while providing a
  stable, plan‑agnostic path and an `--output-root` alias for `--hub`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from ptycho.metadata import MetadataManager
from contextlib import redirect_stdout, redirect_stderr
import io

# Interpreter selection (PYTHON-ENV-001):
# Use the active interpreter for all Python subprocess commands.
# Allow override via PYTHON_BIN for advanced scenarios.
PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)


def run_command(
    cmd: List[str],
    log_path: Path,
    env: dict | None = None,
    cwd: Path | None = None,
) -> None:
    """
    Execute a subprocess command, tee stdout/stderr to log_path, and halt on non-zero exit.

    Args:
        cmd: Command list (will be joined for shell execution)
        log_path: Path to write combined stdout/stderr
        env: Optional environment dict (merged with os.environ)
        cwd: Optional working directory (default: current)

    Raises:
        SystemExit: If subprocess returns non-zero, after writing blocker.log
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with current environment
    exec_env = os.environ.copy()
    if env:
        exec_env.update(env)

    cmd_str = " ".join(str(c) for c in cmd)
    print(f"[run_phase_g_dense] Executing: {cmd_str}", flush=True)
    print(f"[run_phase_g_dense] Logging to: {log_path}", flush=True)

    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            # Write command header to log
            log_handle.write(f"# Command: {cmd_str}\n")
            log_handle.write(f"# CWD: {cwd or Path.cwd()}\n")
            log_handle.write(f"# Log: {log_path}\n")
            log_handle.write("# " + "=" * 78 + "\n\n")
            log_handle.flush()

            proc = subprocess.Popen(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=exec_env,
                cwd=cwd,
                text=True,
                bufsize=1,
            )

            # Tee output to both log and stdout
            for line in proc.stdout:  # type: ignore
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()

            proc.wait()

            if proc.returncode != 0:
                error_msg = (
                    f"[run_phase_g_dense] ERROR: Command failed with return code {proc.returncode}\n"
                    f"Command: {cmd_str}\n"
                    f"Log: {log_path}\n"
                )
                print(error_msg, file=sys.stderr, flush=True)
                log_handle.write("\n" + error_msg)

                # Write blocker log
                hub_root = log_path.parent.parent  # Back up from cli/ to hub root
                blocker_path = hub_root / "analysis" / "blocker.log"
                blocker_path.parent.mkdir(parents=True, exist_ok=True)
                with blocker_path.open("w", encoding="utf-8") as blocker:
                    blocker.write(f"Blocked at: {cmd_str}\n")
                    blocker.write(f"Return code: {proc.returncode}\n")
                    blocker.write(f"Full log: {log_path}\n")

                sys.exit(proc.returncode)

    except Exception as e:
        error_msg = f"[run_phase_g_dense] ERROR: Exception during command execution: {e}\n"
        print(error_msg, file=sys.stderr, flush=True)

        # Write blocker log
        hub_root = log_path.parent.parent
        blocker_path = hub_root / "analysis" / "blocker.log"
        blocker_path.parent.mkdir(parents=True, exist_ok=True)
        with blocker_path.open("w", encoding="utf-8") as blocker:
            blocker.write(f"Blocked at: {cmd_str}\n")
            blocker.write(f"Exception: {e}\n")
            blocker.write(f"Log: {log_path}\n")

        raise


def prepare_hub(hub: Path, clobber: bool = False) -> None:
    """
    Prepare hub directory for Phase C→G pipeline execution.

    This helper normalizes the hub path, detects stale Phase C outputs, and either
    raises an error (when clobber=False) or removes/archives them (when clobber=True).

    Args:
        hub: Root directory for all phase artifacts
        clobber: If True, remove/archive stale outputs; if False (default), raise on stale outputs

    Raises:
        RuntimeError: If clobber=False and stale Phase C outputs exist, with actionable
                      error message mentioning the stale path and --clobber remedy

    Follows TYPE-PATH-001 (Path normalization).
    Default behavior is read-only (clobber=False) to prevent accidental data loss.
    """
    import shutil
    from datetime import datetime

    # TYPE-PATH-001: Normalize to Path
    hub = Path(hub).resolve()
    phase_c_root = hub / "data" / "phase_c"

    # Check if Phase C outputs already exist
    stale_outputs_exist = phase_c_root.exists() and any(phase_c_root.iterdir())

    if not stale_outputs_exist:
        # Clean hub, nothing to do
        print(f"[prepare_hub] Hub is clean: {hub}")
        return

    # Stale outputs detected
    if not clobber:
        # Read-only mode: raise error with actionable guidance
        raise RuntimeError(
            f"Hub contains stale Phase C outputs: {phase_c_root}\n"
            f"To remove previous outputs and proceed, re-run with --clobber flag.\n"
            f"Example: python {Path(__file__).name} --hub {hub} --dose <dose> --view <view> --splits <splits> --clobber"
        )

    # Clobber mode: archive or delete stale outputs
    # Archive strategy: move to timestamped archive directory to preserve evidence
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    archive_root = hub / "archive" / f"phase_c_{timestamp}"

    print(f"[prepare_hub] Archiving stale Phase C outputs to: {archive_root}")

    # Move phase_c_root to archive
    archive_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(phase_c_root), str(archive_root))

    print(f"[prepare_hub] Archived {phase_c_root} → {archive_root}")
    print(f"[prepare_hub] Hub is now clean and ready for new pipeline run")


def validate_phase_c_metadata(hub: Path) -> None:
    """
    Validate that Phase C dataset NPZ outputs contain required _metadata field.

    This guard ensures Phase C outputs were generated with metadata tracking enabled,
    which is required for downstream provenance and parameter validation. Checks both
    train and test splits and raises RuntimeError if metadata is missing.

    Args:
        hub: Root directory containing data/phase_c/ subdirectory

    Raises:
        RuntimeError: If metadata is missing from any Phase C NPZ file, with actionable
                      error message mentioning '_metadata' and the offending file path

    Follows TYPE-PATH-001 (Path normalization), DATA-001 (NPZ contract).
    Does not mutate or delete Phase C outputs (read-only validation).
    """
    from ptycho.metadata import MetadataManager

    # TYPE-PATH-001: Normalize to Path
    hub = Path(hub).resolve()
    phase_c_root = hub / "data" / "phase_c"

    if not phase_c_root.exists():
        raise RuntimeError(
            f"Phase C root directory not found: {phase_c_root}. "
            "Phase C generation may not have completed."
        )

    # Check both train and test splits
    # Phase C generation creates dose_* directories with patched_{train,test}.npz files
    # (modern layout as of PHASEC-METADATA-001)
    splits_to_check = ["train", "test"]

    # Find all dose directories (pattern: dose_*)
    dose_dirs = list(phase_c_root.glob("dose_*"))

    if not dose_dirs:
        raise RuntimeError(
            f"No Phase C dose directories found under {phase_c_root}. "
            "Expected pattern: dose_*/"
        )

    for dose_dir in dose_dirs:
        if not dose_dir.is_dir():
            continue

        for split in splits_to_check:
            # Modern layout: dose_*/patched_{train,test}.npz
            npz_path = dose_dir / f"patched_{split}.npz"

            if not npz_path.exists():
                raise RuntimeError(
                    f"Phase C {split} NPZ file not found: {npz_path}. "
                    f"Expected: patched_{split}.npz in {dose_dir}"
                )

            # Load NPZ and check for metadata
            data_dict, metadata = MetadataManager.load_with_metadata(str(npz_path))

            if metadata is None:
                raise RuntimeError(
                    f"Phase C NPZ file missing required _metadata field: {npz_path}. "
                    f"This file was likely generated before metadata tracking was enabled. "
                    f"Please regenerate Phase C outputs with metadata support."
                )

            # Require canonical transformation history (transpose_rename_convert)
            # This ensures Phase C outputs have gone through format canonicalization
            transformations = metadata.get("data_transformations", [])
            has_canonical_transform = any(
                t.get("tool") == "transpose_rename_convert"
                for t in transformations
            )

            if not has_canonical_transform:
                raise RuntimeError(
                    f"Phase C NPZ file missing required canonical transformation in _metadata: {npz_path}. "
                    f"Expected 'transpose_rename_convert' in data_transformations list. "
                    f"Found transformations: {[t.get('tool') for t in transformations]}. "
                        f"Please ensure Phase C pipeline includes transpose_rename_convert canonicalization."
                    )

                print(f"[validate_phase_c_metadata] ✓ {npz_path.name} contains _metadata with canonical transformation")

    print(f"[validate_phase_c_metadata] SUCCESS: All Phase C NPZ files contain required _metadata")


def generate_artifact_inventory(hub: Path) -> None:
    """
    Generate deterministic artifact_inventory.txt listing all files in the hub.

    This helper walks the hub directory tree and emits a sorted list of all artifacts
    (relative POSIX paths from hub root) to analysis/artifact_inventory.txt.
    The inventory is deterministic (sorted lexicographically) to enable diffs across runs.

    Args:
        hub: Root directory containing all phase artifacts

    Follows TYPE-PATH-001 (Path normalization), deterministic ordering.
    Output format: one relative POSIX path per line, sorted.
    """
    # TYPE-PATH-001: Normalize to Path
    hub = Path(hub).resolve()
    analysis = hub / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    inventory_path = analysis / "artifact_inventory.txt"

    # Walk hub directory and collect all file paths
    all_files = []
    for root, dirs, files in os.walk(hub):
        root_path = Path(root)
        for file in files:
            file_path = root_path / file
            # Compute relative path from hub root
            try:
                rel_path = file_path.relative_to(hub)
                # Convert to POSIX style for deterministic serialization (TYPE-PATH-001)
                all_files.append(rel_path.as_posix())
            except ValueError:
                # Skip files outside hub (should not happen)
                pass

    # Sort lexicographically for deterministic output
    all_files.sort()

    # Write inventory to file
    with inventory_path.open("w", encoding="utf-8") as f:
        for rel_path in all_files:
            f.write(f"{rel_path}\n")

    print(f"[generate_artifact_inventory] Wrote {len(all_files)} artifact paths to: {inventory_path.relative_to(hub)}")


def summarize_phase_g_outputs(hub: Path) -> None:
    """
    Parse comparison_manifest.json and per-job metrics CSVs to emit a metrics summary.

    This helper validates successful Phase G execution, extracts key metrics
    (MS-SSIM, MAE) from per-job comparison_metrics.csv files, and writes both
    JSON and Markdown summaries to {hub}/analysis/.

    Also validates Phase C metadata compliance and persists the result.

    Args:
        hub: Root directory containing analysis/ subdirectory with comparison_manifest.json

    Raises:
        RuntimeError: If manifest is missing, n_failed > 0, or required CSV files are absent

    Follows TYPE-PATH-001 (Path normalization).
    """
    # TYPE-PATH-001: Normalize to Path
    hub = Path(hub).resolve()
    analysis = hub / "analysis"

    # Validate Phase C metadata compliance (PHASEC-METADATA-001)
    # This checks that all Phase C NPZ files contain _metadata with canonical transformations
    phase_c_metadata_compliance = {}
    try:
        phase_c_root = hub / "data" / "phase_c"
        if phase_c_root.exists():
            # Extract dose directories
            dose_dirs = sorted([d for d in phase_c_root.iterdir() if d.is_dir() and d.name.startswith("dose_")])

            # Validate each dose × split combination
            for dose_dir in dose_dirs:
                dose_value = int(dose_dir.name.replace("dose_", ""))
                dose_key = f"dose_{dose_value}"
                phase_c_metadata_compliance[dose_key] = {}

                for split in ["train", "test"]:
                    npz_path = dose_dir / f"patched_{split}.npz"
                    if npz_path.exists():
                        try:
                            data_dict, metadata = MetadataManager.load_with_metadata(str(npz_path))
                            has_metadata = metadata is not None
                            has_canonical = False
                            if has_metadata:
                                transformations = metadata.get("data_transformations", [])
                                has_canonical = any(
                                    t.get("tool") == "transpose_rename_convert"
                                    for t in transformations
                                )
                            phase_c_metadata_compliance[dose_key][split] = {
                                "npz_path": str(npz_path.relative_to(hub)),
                                "has_metadata": has_metadata,
                                "has_canonical_transform": has_canonical,
                                "compliant": has_metadata and has_canonical,
                            }
                        except Exception as e:
                            phase_c_metadata_compliance[dose_key][split] = {
                                "npz_path": str(npz_path.relative_to(hub)),
                                "has_metadata": False,
                                "has_canonical_transform": False,
                                "compliant": False,
                                "error": str(e),
                            }
                    else:
                        phase_c_metadata_compliance[dose_key][split] = {
                            "npz_path": str(npz_path.relative_to(hub)),
                            "has_metadata": False,
                            "has_canonical_transform": False,
                            "compliant": False,
                            "error": "File not found",
                        }
        else:
            phase_c_metadata_compliance = {"error": f"Phase C root not found: {phase_c_root}"}
    except Exception as e:
        phase_c_metadata_compliance = {"error": f"Failed to validate Phase C metadata: {e}"}

    # Validate manifest exists
    manifest_path = analysis / "comparison_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"comparison_manifest.json not found at {manifest_path}. "
            "Phase G comparison may not have completed."
        )

    # Load manifest
    with manifest_path.open() as f:
        manifest = json.load(f)

    # Fail fast on execution failures
    n_failed = manifest.get('n_failed', 0)
    if n_failed > 0:
        raise RuntimeError(
            f"Manifest reports {n_failed} failed comparison jobs (n_failed > 0). "
            "Cannot summarize metrics until all jobs succeed."
        )

    n_jobs = manifest.get('n_jobs', 0)
    n_success = manifest.get('n_success', 0)
    execution_results = manifest.get('execution_results', [])

    # Build summary data structure
    summary_data = {
        'n_jobs': n_jobs,
        'n_success': n_success,
        'n_failed': n_failed,
        'jobs': [],
    }

    # Collect per-model metrics across all jobs for aggregate computation
    # Structure: {model_name: {metric_name: {'amplitude': [...], 'phase': [...]}}}
    model_metrics_collector = {}

    # Extract metrics from each successful job
    for result in execution_results:
        dose = result['dose']
        view = result['view']
        split = result['split']
        returncode = result.get('returncode', -1)

        if returncode != 0:
            continue  # Skip failed jobs

        # Construct path to comparison_metrics.csv
        # Per comparison.py:189-191, output_dir = artifact_root / f"dose_{dose}" / view / split
        job_dir = analysis / f"dose_{dose}" / view / split
        csv_path = job_dir / "comparison_metrics.csv"

        if not csv_path.exists():
            raise RuntimeError(
                f"comparison_metrics.csv not found for dose={dose}, view={view}, split={split}. "
                f"Expected at: {csv_path}"
            )

        # Parse CSV (tidy format: model, metric, amplitude, phase, value)
        job_metrics = []
        with csv_path.open(newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                model = row['model']
                metric = row['metric']
                amplitude_str = row.get('amplitude', '').strip()
                phase_str = row.get('phase', '').strip()
                value_str = row.get('value', '').strip()

                # Build metric entry
                metric_entry = {
                    'model': model,
                    'metric': metric,
                }

                if amplitude_str and phase_str:
                    # Tuple metric (amplitude, phase)
                    metric_entry['amplitude'] = float(amplitude_str)
                    metric_entry['phase'] = float(phase_str)

                    # Collect for aggregation (MS-SSIM and MAE only)
                    if metric in ('ms_ssim', 'mae'):
                        if model not in model_metrics_collector:
                            model_metrics_collector[model] = {}
                        if metric not in model_metrics_collector[model]:
                            model_metrics_collector[model][metric] = {'amplitude': [], 'phase': []}
                        model_metrics_collector[model][metric]['amplitude'].append(metric_entry['amplitude'])
                        model_metrics_collector[model][metric]['phase'].append(metric_entry['phase'])
                elif value_str:
                    # Scalar metric
                    metric_entry['value'] = float(value_str)

                job_metrics.append(metric_entry)

        # Append job summary
        job_summary = {
            'dose': dose,
            'view': view,
            'split': split,
            'metrics': job_metrics,
        }
        summary_data['jobs'].append(job_summary)

    # Compute aggregate metrics per model
    # Structure: {model_name: {metric_name: {'mean_amplitude': float, 'best_amplitude': float, ...}}}
    aggregate_metrics = {}
    for model_name, metrics_dict in sorted(model_metrics_collector.items()):
        aggregate_metrics[model_name] = {}

        for metric_name, values_dict in sorted(metrics_dict.items()):
            amp_values = values_dict.get('amplitude', [])
            phase_values = values_dict.get('phase', [])

            aggregate_entry = {}

            if amp_values:
                aggregate_entry['mean_amplitude'] = sum(amp_values) / len(amp_values)
                if metric_name == 'ms_ssim':
                    # Best MS-SSIM is maximum (higher is better)
                    aggregate_entry['best_amplitude'] = max(amp_values)
                # MAE: mean only (lower is better, "best" is ambiguous across jobs)

            if phase_values:
                aggregate_entry['mean_phase'] = sum(phase_values) / len(phase_values)
                if metric_name == 'ms_ssim':
                    # Best MS-SSIM is maximum (higher is better)
                    aggregate_entry['best_phase'] = max(phase_values)
                # MAE: mean only

            aggregate_metrics[model_name][metric_name] = aggregate_entry

    # Add aggregates to summary data
    summary_data['aggregate_metrics'] = aggregate_metrics

    # Add Phase C metadata compliance (PHASEC-METADATA-001)
    summary_data['phase_c_metadata_compliance'] = phase_c_metadata_compliance

    # Write JSON summary
    json_summary_path = analysis / "metrics_summary.json"
    with json_summary_path.open('w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"[summarize_phase_g_outputs] Wrote JSON summary: {json_summary_path}")

    # Write Markdown summary
    md_summary_path = analysis / "metrics_summary.md"
    with md_summary_path.open('w') as f:
        f.write("# Phase G Metrics Summary\n\n")
        f.write(f"**Total Jobs:** {n_jobs}  \n")
        f.write(f"**Successful:** {n_success}  \n")
        f.write(f"**Failed:** {n_failed}  \n\n")

        f.write("---\n\n")

        # Write aggregate metrics section
        f.write("## Aggregate Metrics\n\n")
        f.write("Summary statistics across all jobs per model.\n\n")

        for model_name in sorted(aggregate_metrics.keys()):
            model_aggs = aggregate_metrics[model_name]
            f.write(f"### {model_name}\n\n")

            # MS-SSIM table
            if 'ms_ssim' in model_aggs:
                ms_ssim = model_aggs['ms_ssim']
                f.write("**MS-SSIM:**\n\n")
                f.write("| Statistic | Amplitude | Phase |\n")
                f.write("|-----------|-----------|-------|\n")
                mean_amp = ms_ssim.get('mean_amplitude', '')
                mean_phase = ms_ssim.get('mean_phase', '')
                best_amp = ms_ssim.get('best_amplitude', '')
                best_phase = ms_ssim.get('best_phase', '')
                mean_amp_str = f"{mean_amp:.3f}" if mean_amp != '' else ''
                mean_phase_str = f"{mean_phase:.3f}" if mean_phase != '' else ''
                best_amp_str = f"{best_amp:.3f}" if best_amp != '' else ''
                best_phase_str = f"{best_phase:.3f}" if best_phase != '' else ''
                f.write(f"| Mean | {mean_amp_str} | {mean_phase_str} |\n")
                f.write(f"| Best | {best_amp_str} | {best_phase_str} |\n\n")

            # MAE table
            if 'mae' in model_aggs:
                mae = model_aggs['mae']
                f.write("**MAE:**\n\n")
                f.write("| Statistic | Amplitude | Phase |\n")
                f.write("|-----------|-----------|-------|\n")
                mean_amp = mae.get('mean_amplitude', '')
                mean_phase = mae.get('mean_phase', '')
                mean_amp_str = f"{mean_amp:.3f}" if mean_amp != '' else ''
                mean_phase_str = f"{mean_phase:.3f}" if mean_phase != '' else ''
                f.write(f"| Mean | {mean_amp_str} | {mean_phase_str} |\n\n")

        f.write("---\n\n")

        for job in summary_data['jobs']:
            dose = job['dose']
            view = job['view']
            split = job['split']

            f.write(f"## Job: {view}/{split} (dose={dose})\n\n")

            # Group metrics by model
            models = {}
            for m in job['metrics']:
                model_name = m['model']
                if model_name not in models:
                    models[model_name] = []
                models[model_name].append(m)

            for model_name, metrics in models.items():
                f.write(f"### {model_name}\n\n")
                f.write("| Metric | Amplitude | Phase | Value |\n")
                f.write("|--------|-----------|-------|-------|\n")

                for metric in metrics:
                    metric_name = metric['metric']
                    amp = metric.get('amplitude', '')
                    phase = metric.get('phase', '')
                    val = metric.get('value', '')

                    # Format floats
                    amp_str = f"{amp:.4f}" if amp != '' else ''
                    phase_str = f"{phase:.4f}" if phase != '' else ''
                    val_str = f"{val:.4f}" if val != '' else ''

                    f.write(f"| {metric_name} | {amp_str} | {phase_str} | {val_str} |\n")

                f.write("\n")

            f.write("---\n\n")

        # Write Phase C Metadata Compliance section (PHASEC-METADATA-001)
        f.write("## Phase C Metadata Compliance\n\n")
        f.write("Validation of Phase C NPZ files for _metadata and canonical transformations.\n\n")

        if "error" in phase_c_metadata_compliance:
            f.write(f"**Error:** {phase_c_metadata_compliance['error']}\n\n")
        else:
            # Build compliance table
            f.write("| Dose | Split | Compliant | Has Metadata | Has Canonical Transform | Path |\n")
            f.write("|------|-------|-----------|--------------|-------------------------|------|\n")

            for dose_key in sorted(phase_c_metadata_compliance.keys()):
                dose_data = phase_c_metadata_compliance[dose_key]
                for split in sorted(dose_data.keys()):
                    split_data = dose_data[split]
                    compliant = "✓" if split_data.get("compliant", False) else "✗"
                    has_meta = "✓" if split_data.get("has_metadata", False) else "✗"
                    has_canon = "✓" if split_data.get("has_canonical_transform", False) else "✗"
                    path = split_data.get("npz_path", "")
                    error_note = ""
                    if "error" in split_data:
                        error_note = f" ({split_data['error']})"

                    f.write(f"| {dose_key} | {split} | {compliant} | {has_meta} | {has_canon} | {path}{error_note} |\n")

            f.write("\n")

    print(f"[summarize_phase_g_outputs] Wrote Markdown summary: {md_summary_path}")


def persist_delta_highlights(
    aggregate_metrics: dict,
    output_dir: Path,
    hub: Path
) -> dict:
    """
    Compute and persist metric deltas (PtychoPINN vs Baseline/PtyChi) with correct precision.

    Creates two output files:
    1. metrics_delta_highlights.txt - Full highlights with amplitude+phase (4 lines)
    2. metrics_delta_highlights_preview.txt - Phase-only preview (4 lines)
    3. metrics_delta_summary.json - Structured numeric deltas

    Precision requirements:
    - MS-SSIM: ±0.000 (3 decimals)
    - MAE: ±0.000000 (6 decimals)

    Args:
        aggregate_metrics: Dict with PtychoPINN/Baseline/PtyChi metrics
        output_dir: Directory to write output files (TYPE-PATH-001: must be POSIX-relative to hub)
        hub: Hub root path for computing relative paths

    Returns:
        delta_summary: Dict with structure matching metrics_delta_summary.json spec
            {
                "generated_at": "2025-11-11T00:33:51Z",
                "source_metrics": "analysis/metrics_summary.json",
                "deltas": {
                    "vs_Baseline": {
                        "ms_ssim": {"amplitude": float, "phase": float},
                        "mae": {"amplitude": float, "phase": float}
                    },
                    "vs_PtyChi": {...}
                }
            }

    Follows TYPE-PATH-001 (POSIX paths), STUDY-001 (phase emphasis), TEST-CLI-001 (preview parity).
    """
    from datetime import datetime, timezone

    # Extract model metrics
    pinn = aggregate_metrics.get("PtychoPINN", {})
    baseline = aggregate_metrics.get("Baseline", {})
    ptychi = aggregate_metrics.get("PtyChi", {})

    pinn_ms = pinn.get("ms_ssim", {})
    base_ms = baseline.get("ms_ssim", {})
    pty_ms = ptychi.get("ms_ssim", {})

    pinn_mae = pinn.get("mae", {})
    base_mae = baseline.get("mae", {})
    pty_mae = ptychi.get("mae", {})

    # Helper to compute formatted delta with metric-specific precision
    def compute_delta_ms_ssim(pinn_val, other_val):
        """MS-SSIM delta with ±0.000 precision (3 decimals)."""
        if pinn_val is None or other_val is None:
            return "N/A"
        delta = pinn_val - other_val
        return f"{delta:+.3f}"

    def compute_delta_mae(pinn_val, other_val):
        """MAE delta with ±0.000000 precision (6 decimals)."""
        if pinn_val is None or other_val is None:
            return "N/A"
        delta = pinn_val - other_val
        return f"{delta:+.6f}"

    # Helper to get raw numeric delta (None if either value is missing)
    def compute_numeric_delta(pinn_val, other_val):
        if pinn_val is None or other_val is None:
            return None
        return pinn_val - other_val

    # Compute MS-SSIM deltas (higher is better → positive delta is good for PtychoPINN)
    delta_ms_base_amp = compute_delta_ms_ssim(pinn_ms.get("mean_amplitude"), base_ms.get("mean_amplitude"))
    delta_ms_base_phase = compute_delta_ms_ssim(pinn_ms.get("mean_phase"), base_ms.get("mean_phase"))
    delta_ms_pty_amp = compute_delta_ms_ssim(pinn_ms.get("mean_amplitude"), pty_ms.get("mean_amplitude"))
    delta_ms_pty_phase = compute_delta_ms_ssim(pinn_ms.get("mean_phase"), pty_ms.get("mean_phase"))

    # Compute MAE deltas (lower is better → negative delta is good for PtychoPINN)
    delta_mae_base_amp = compute_delta_mae(pinn_mae.get("mean_amplitude"), base_mae.get("mean_amplitude"))
    delta_mae_base_phase = compute_delta_mae(pinn_mae.get("mean_phase"), base_mae.get("mean_phase"))
    delta_mae_pty_amp = compute_delta_mae(pinn_mae.get("mean_amplitude"), pty_mae.get("mean_amplitude"))
    delta_mae_pty_phase = compute_delta_mae(pinn_mae.get("mean_phase"), pty_mae.get("mean_phase"))

    # Build full highlights lines (amplitude + phase, 4 lines)
    highlights_lines = [
        f"MS-SSIM Δ (PtychoPINN - Baseline)  : amplitude {delta_ms_base_amp}  phase {delta_ms_base_phase}",
        f"MS-SSIM Δ (PtychoPINN - PtyChi)    : amplitude {delta_ms_pty_amp}  phase {delta_ms_pty_phase}",
        f"MAE Δ (PtychoPINN - Baseline)      : amplitude {delta_mae_base_amp}  phase {delta_mae_base_phase}",
        f"MAE Δ (PtychoPINN - PtyChi)        : amplitude {delta_mae_pty_amp}  phase {delta_mae_pty_phase}",
    ]

    # Build preview lines (phase-only, 4 lines)
    preview_lines = [
        f"MS-SSIM Δ (PtychoPINN - Baseline): {delta_ms_base_phase}",
        f"MS-SSIM Δ (PtychoPINN - PtyChi): {delta_ms_pty_phase}",
        f"MAE Δ (PtychoPINN - Baseline): {delta_mae_base_phase}",
        f"MAE Δ (PtychoPINN - PtyChi): {delta_mae_pty_phase}",
    ]

    # Write highlights.txt (TYPE-PATH-001)
    highlights_txt_path = Path(output_dir) / "metrics_delta_highlights.txt"
    with highlights_txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(highlights_lines) + "\n")

    # Write preview.txt (TYPE-PATH-001)
    preview_txt_path = Path(output_dir) / "metrics_delta_highlights_preview.txt"
    with preview_txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(preview_lines) + "\n")

    # Build JSON structure with raw numeric deltas plus provenance metadata
    utc_now = datetime.now(timezone.utc)
    generated_at = utc_now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Compute relative path from hub to metrics_summary.json (TYPE-PATH-001: relative POSIX serialization)
    metrics_summary_path = Path(output_dir) / "metrics_summary.json"
    source_metrics_rel = metrics_summary_path.relative_to(hub).as_posix()

    delta_summary = {
        "generated_at": generated_at,
        "source_metrics": source_metrics_rel,
        "deltas": {
            "vs_Baseline": {
                "ms_ssim": {
                    "amplitude": compute_numeric_delta(
                        pinn_ms.get("mean_amplitude"), base_ms.get("mean_amplitude")
                    ),
                    "phase": compute_numeric_delta(
                        pinn_ms.get("mean_phase"), base_ms.get("mean_phase")
                    )
                },
                "mae": {
                    "amplitude": compute_numeric_delta(
                        pinn_mae.get("mean_amplitude"), base_mae.get("mean_amplitude")
                    ),
                    "phase": compute_numeric_delta(
                        pinn_mae.get("mean_phase"), base_mae.get("mean_phase")
                    )
                }
            },
            "vs_PtyChi": {
                "ms_ssim": {
                    "amplitude": compute_numeric_delta(
                        pinn_ms.get("mean_amplitude"), pty_ms.get("mean_amplitude")
                    ),
                    "phase": compute_numeric_delta(
                        pinn_ms.get("mean_phase"), pty_ms.get("mean_phase")
                    )
                },
                "mae": {
                    "amplitude": compute_numeric_delta(
                        pinn_mae.get("mean_amplitude"), pty_mae.get("mean_amplitude")
                    ),
                    "phase": compute_numeric_delta(
                        pinn_mae.get("mean_phase"), pty_mae.get("mean_phase")
                    )
                }
            }
        }
    }

    # Write JSON with indentation for readability (TYPE-PATH-001)
    delta_json_path = Path(output_dir) / "metrics_delta_summary.json"
    with delta_json_path.open("w", encoding="utf-8") as f:
        json.dump(delta_summary, f, indent=2)

    return delta_summary


def main() -> int:
    """CLI entry point for Phase C→G dense execution orchestrator."""
    parser = argparse.ArgumentParser(
        description="Phase G Dense Execution Orchestrator - run full Phase C→G pipeline"
    )
    parser.add_argument(
        "--hub",
        type=Path,
        required=True,
        help="Hub directory for all phase artifacts and logs",
    )
    parser.add_argument(
        "--dose",
        type=int,
        required=True,
        help="Dose value (e.g., 1000, 10000, 100000)",
    )
    parser.add_argument(
        "--view",
        type=str,
        required=True,
        choices=["dense", "sparse"],
        help="Overlap view type",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Data splits to process (default: train test)",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Verification mode: print planned commands without execution",
    )
    parser.add_argument(
        "--clobber",
        action="store_true",
        help="Remove/archive stale Phase C outputs before starting pipeline (default: error on stale outputs)",
    )
    parser.add_argument(
        "--skip-post-verify",
        action="store_true",
        help="Skip post-verify automation (verifier + highlights checker) for debugging",
    )
    parser.add_argument(
        "--post-verify-only",
        action="store_true",
        help="Skip Phase C→F execution, reuse existing hub outputs, run only SSIM grid + verifier + highlights checker",
    )

    args = parser.parse_args()

    # Guard: --post-verify-only is mutually exclusive with --clobber and --skip-post-verify
    if args.post_verify_only:
        if args.clobber:
            print(
                "[run_phase_g_dense] ERROR: --post-verify-only cannot be used with --clobber "
                "(verification-only runs must not delete existing outputs)",
                file=sys.stderr,
            )
            return 1
        if args.skip_post_verify:
            print(
                "[run_phase_g_dense] ERROR: --post-verify-only cannot be used with --skip-post-verify "
                "(verification-only runs must execute post-verify automation)",
                file=sys.stderr,
            )
            return 1

    # TYPE-PATH-001: Normalize to Path objects
    hub = Path(args.hub).resolve()
    dose = args.dose
    view = args.view
    splits = args.splits

    # Verify AUTHORITATIVE_CMDS_DOC is set
    if "AUTHORITATIVE_CMDS_DOC" not in os.environ:
        print(
            "[run_phase_g_dense] WARNING: AUTHORITATIVE_CMDS_DOC not set; "
            "setting to ./docs/TESTING_GUIDE.md",
            file=sys.stderr,
        )
        os.environ["AUTHORITATIVE_CMDS_DOC"] = "./docs/TESTING_GUIDE.md"

    # Define artifact directories (TYPE-PATH-001)
    phase_c_root = hub / "data" / "phase_c"
    phase_d_root = hub / "data" / "phase_d"
    phase_e_root = hub / "data" / "phase_e"
    phase_f_root = hub / "data" / "phase_f"
    phase_g_root = hub / "analysis"
    cli_log_dir = hub / "cli"

    # Base NPZ for Phase C generation (from generation.py default)
    base_npz = Path("tike_outputs/fly001_reconstructed_final_prepared/fly001_reconstructed_interp_smooth_both.npz")

    # Collect commands
    commands = []

    # Phase C: Dataset Generation
    phase_c_cmd = [
        PYTHON_BIN, "-m", "studies.fly64_dose_overlap.generation",
        "--base-npz", str(base_npz),
        "--output-root", str(phase_c_root),
        "--dose", str(dose),
    ]
    commands.append(("Phase C: Dataset Generation", phase_c_cmd, cli_log_dir / "phase_c_generation.log"))

    # Phase D: Overlap View Generation (programmatic, avoid fragile CLI arg passing)
    # Use defaults aligned with prior study choices for dense view.
    # Sentinel command tells the executor loop to invoke programmatic path.
    phase_d_log = cli_log_dir / f"phase_d_{view}.log"
    commands.append(("Phase D: Overlap View Generation", ["__PHASE_D_PROGRAMMATIC__"], phase_d_log))

    # Phase E: Training (baseline gs1 + view gs2)
    # Baseline (gs1)
    phase_e_baseline_cmd = [
        PYTHON_BIN, "-m", "studies.fly64_dose_overlap.training",
        "--phase-c-root", str(phase_c_root),
        "--phase-d-root", str(phase_d_root),
        "--artifact-root", str(phase_e_root),
        "--dose", str(dose),
        "--view", "baseline",
        "--gridsize", "1",
        "--backend", "tensorflow",
    ]
    commands.append(("Phase E: Training Baseline (gs1)", phase_e_baseline_cmd, cli_log_dir / f"phase_e_baseline_gs1_dose{dose}.log"))

    # Dense/sparse (gs2)
    phase_e_view_cmd = [
        PYTHON_BIN, "-m", "studies.fly64_dose_overlap.training",
        "--phase-c-root", str(phase_c_root),
        "--phase-d-root", str(phase_d_root),
        "--artifact-root", str(phase_e_root),
        "--dose", str(dose),
        "--view", view,
        "--gridsize", "2",
        "--backend", "tensorflow",
    ]
    commands.append((f"Phase E: Training {view.capitalize()} (gs2)", phase_e_view_cmd, cli_log_dir / f"phase_e_{view}_gs2_dose{dose}.log"))

    # Phase F: Reconstruction (one job per split)
    for split in splits:
        phase_f_cmd = [
            PYTHON_BIN, "-m", "studies.fly64_dose_overlap.reconstruction",
            "--phase-c-root", str(phase_c_root),
            "--phase-d-root", str(phase_d_root),
            "--artifact-root", str(phase_f_root),
            "--dose", str(dose),
            "--view", view,
            "--split", split,
            "--allow-missing-phase-d",
        ]
        commands.append((f"Phase F: Reconstruction {view}/{split}", phase_f_cmd, cli_log_dir / f"phase_f_{view}_{split}.log"))

    # Phase G: Comparison (one job per split)
    for split in splits:
        phase_g_cmd = [
            PYTHON_BIN, "-m", "studies.fly64_dose_overlap.comparison",
            "--phase-c-root", str(phase_c_root),
            "--phase-e-root", str(phase_e_root),
            "--phase-f-root", str(phase_f_root),
            "--artifact-root", str(phase_g_root),
            "--dose", str(dose),
            "--view", view,
            "--split", split,
        ]
        commands.append((f"Phase G: Comparison {view}/{split}", phase_g_cmd, cli_log_dir / f"phase_g_{view}_{split}.log"))

    # Reporting Helper: Generate aggregate metrics report
    # (executed after summarize_phase_g_outputs generates metrics_summary.json)
    metrics_summary_json = phase_g_root / "metrics_summary.json"
    aggregate_report_md = phase_g_root / "aggregate_report.md"
    aggregate_highlights_txt = phase_g_root / "aggregate_highlights.txt"
    report_helper_script = Path(__file__).parent / "report_phase_g_dense_metrics.py"
    report_helper_cmd = [
        PYTHON_BIN, str(report_helper_script),
        "--metrics", str(metrics_summary_json),
        "--output", str(aggregate_report_md),
        "--highlights", str(aggregate_highlights_txt),
    ]
    report_helper_log = cli_log_dir / "aggregate_report_cli.log"

    # Analyze Digest: Generate final metrics digest from summary + highlights
    # (executed after reporting helper generates aggregate_report.md and aggregate_highlights.txt)
    metrics_digest_md = phase_g_root / "metrics_digest.md"
    analyze_digest_script = Path(__file__).parent / "analyze_dense_metrics.py"
    analyze_digest_cmd = [
        PYTHON_BIN, str(analyze_digest_script),
        "--metrics", str(metrics_summary_json),
        "--highlights", str(aggregate_highlights_txt),
        "--output", str(metrics_digest_md),
    ]
    analyze_digest_log = cli_log_dir / "metrics_digest_cli.log"

    # SSIM Grid: Generate phase-only MS-SSIM/MAE delta table
    # (executed after delta summary generation, validates preview compliance)
    ssim_grid_script = Path(__file__).parent / "ssim_grid.py"
    ssim_grid_summary_md = phase_g_root / "ssim_grid_summary.md"
    ssim_grid_cmd = [
        PYTHON_BIN, str(ssim_grid_script),
        "--hub", str(hub),
    ]
    ssim_grid_log = cli_log_dir / "ssim_grid_cli.log"

    # Post-Verify Automation: Verify pipeline artifacts and check highlights match
    # (executed after ssim_grid generates the grid summary and preview metadata)
    verify_script = Path(__file__).parent / "verify_dense_pipeline_artifacts.py"
    verify_report_json = phase_g_root / "verification_report.json"
    verify_cmd = [
        PYTHON_BIN, str(verify_script),
        "--hub", str(hub),
        "--report", str(verify_report_json),
        "--dose", str(dose),
        "--view", view,
    ]
    verify_log = phase_g_root / "verify_dense_stdout.log"

    check_script = Path(__file__).parent / "check_dense_highlights_match.py"
    check_cmd = [
        PYTHON_BIN, str(check_script),
        "--hub", str(hub),
    ]
    check_log = phase_g_root / "check_dense_highlights.log"

    # Collect-only mode: print commands and exit
    if args.collect_only:
        print("[run_phase_g_dense] Collect-only mode: planned commands:")
        # --post-verify-only: print only SSIM grid + verification commands
        if args.post_verify_only:
            print("\n[run_phase_g_dense] Post-verify-only mode: skipping Phase C→F")
            print(f"\n1. Analysis: Generate SSIM grid summary (phase-only)")
            print(f"   Command: {' '.join(str(c) for c in ssim_grid_cmd)}")
            print(f"   Log: {ssim_grid_log.relative_to(hub)}")
            print(f"\n2. Post-Verify: Verify pipeline artifacts")
            print(f"   Command: {' '.join(str(c) for c in verify_cmd)}")
            print(f"   Log: {verify_log.relative_to(hub)}")
            print(f"\n3. Post-Verify: Check highlights match")
            print(f"   Command: {' '.join(str(c) for c in check_cmd)}")
            print(f"   Log: {check_log.relative_to(hub)}")
        else:
            # Print Phase C-G commands
            for i, (desc, cmd, log_path) in enumerate(commands, 1):
                print(f"\n{i}. {desc}")
                print(f"   Command: {' '.join(str(c) for c in cmd)}")
                print(f"   Log: {log_path}")
            # Print reporting helper command
            print(f"\n{len(commands) + 1}. Reporting: Aggregate metrics report")
            print(f"   Command: {' '.join(str(c) for c in report_helper_cmd)}")
            print(f"   Log: {report_helper_log}")
            # Print analyze digest command
            print(f"\n{len(commands) + 2}. Analysis: Generate metrics digest")
            print(f"   Command: {' '.join(str(c) for c in analyze_digest_cmd)}")
            print(f"   Log: {analyze_digest_log}")
            # Print ssim_grid command
            print(f"\n{len(commands) + 3}. Analysis: Generate SSIM grid summary (phase-only)")
            print(f"   Command: {' '.join(str(c) for c in ssim_grid_cmd)}")
            print(f"   Log: {ssim_grid_log}")
            # Print post-verify commands (conditional on --skip-post-verify)
            if not args.skip_post_verify:
                print(f"\n{len(commands) + 4}. Post-Verify: Verify pipeline artifacts")
                print(f"   Command: {' '.join(str(c) for c in verify_cmd)}")
                print(f"   Log: {verify_log}")
                print(f"\n{len(commands) + 5}. Post-Verify: Check highlights match")
                print(f"   Command: {' '.join(str(c) for c in check_cmd)}")
                print(f"   Log: {check_log}")
            else:
                print(f"\n[run_phase_g_dense] Post-verify automation skipped (--skip-post-verify)")
        return 0

    # --post-verify-only mode: skip prepare_hub and Phase C→F, run only verification commands
    if args.post_verify_only:
        print("\n" + "=" * 80)
        print("[run_phase_g_dense] Post-verify-only mode enabled")
        print("=" * 80)
        print("\nSkipping Phase C→F execution (reusing existing hub outputs)")
        print(f"Hub: {hub}")
        print(f"Dose: {dose}, View: {view}\n")

        # Ensure hub directories exist
        cli_log_dir.mkdir(parents=True, exist_ok=True)
        phase_g_root.mkdir(parents=True, exist_ok=True)

        # Run SSIM grid generation
        print("\n" + "=" * 80)
        print("[run_phase_g_dense] Generating SSIM grid summary (phase-only)...")
        print("=" * 80 + "\n")
        run_command(ssim_grid_cmd, ssim_grid_log)

        # Run post-verify automation (verify + check)
        print("\n" + "=" * 80)
        print("[run_phase_g_dense] Running post-verify automation...")
        print("=" * 80 + "\n")

        print("[run_phase_g_dense] Verifying pipeline artifacts...")
        run_command(verify_cmd, verify_log)

        print("[run_phase_g_dense] Checking highlights match...")
        run_command(check_cmd, check_log)

        print("\n[run_phase_g_dense] Post-verify automation completed successfully")

        # Re-generate artifact inventory
        print("\n" + "=" * 80)
        print("[run_phase_g_dense] Generating artifact inventory...")
        print("=" * 80 + "\n")
        generate_artifact_inventory(hub)

        # Validate artifact inventory file exists and announce its location (TYPE-PATH-001, DATA-001)
        artifact_inventory_path = Path(phase_g_root) / "artifact_inventory.txt"
        if not artifact_inventory_path.exists():
            raise RuntimeError(
                f"Artifact inventory file not found: {artifact_inventory_path}\n"
                f"generate_artifact_inventory() should have created this file.\n"
                f"This indicates a critical failure in artifact tracking."
            )

        print("\n" + "=" * 80)
        print("[run_phase_g_dense] SUCCESS: Post-verify-only mode completed")
        print("=" * 80)

        # Print artifact paths
        print(f"\nArtifacts saved to: {hub}")
        print(f"Artifact inventory: {artifact_inventory_path.relative_to(hub)}")
        print(f"CLI logs: {cli_log_dir.relative_to(hub)}")
        print(f"Analysis outputs: {phase_g_root.relative_to(hub)}")

        # Add SSIM grid summary to success banner
        ssim_grid_summary_path = Path(phase_g_root) / "ssim_grid_summary.md"
        if ssim_grid_summary_path.exists():
            print(f"SSIM Grid Summary (phase-only): {ssim_grid_summary_path.relative_to(hub)}")

        ssim_grid_log_path = Path(cli_log_dir) / "ssim_grid_cli.log"
        if ssim_grid_log_path.exists():
            print(f"SSIM Grid log: {ssim_grid_log_path.relative_to(hub)}")

        # Add post-verify artifacts to success banner
        verify_report_path = Path(phase_g_root) / "verification_report.json"
        if verify_report_path.exists():
            print(f"Verification report: {verify_report_path.relative_to(hub)}")

        verify_log_path = Path(phase_g_root) / "verify_dense_stdout.log"
        if verify_log_path.exists():
            print(f"Verification log: {verify_log_path.relative_to(hub)}")

        check_log_path = Path(phase_g_root) / "check_dense_highlights.log"
        if check_log_path.exists():
            print(f"Highlights check log: {check_log_path.relative_to(hub)}")

        return 0

    # Prepare hub: detect and handle stale outputs
    print("\n" + "=" * 80)
    print("[run_phase_g_dense] Preparing hub...")
    print("=" * 80 + "\n")
    try:
        prepare_hub(hub, clobber=args.clobber)
    except RuntimeError as e:
        print(f"[run_phase_g_dense] ERROR during hub preparation: {e}", file=sys.stderr)
        # Write blocker log
        blocker_path = phase_g_root / "blocker.log"
        blocker_path.parent.mkdir(parents=True, exist_ok=True)
        with blocker_path.open("w", encoding="utf-8") as blocker:
            blocker.write("Blocked during hub preparation\n")
            blocker.write(f"Exception: {e}\n")
        return 1

    # Execute all commands in sequence
    print(f"\n[run_phase_g_dense] Starting Phase C→G pipeline for dose={dose}, view={view}, splits={splits}")
    print(f"[run_phase_g_dense] Hub: {hub}")
    print(f"[run_phase_g_dense] Total commands: {len(commands)}\n")

    for i, (desc, cmd, log_path) in enumerate(commands, 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(commands)}] {desc}")
        print(f"{'=' * 80}\n")
        # Special-case Phase D to call Python API directly (no CLI args dependency)
        if isinstance(cmd, list) and len(cmd) == 1 and cmd[0] == "__PHASE_D_PROGRAMMATIC__":
            log_path.parent.mkdir(parents=True, exist_ok=True)
            buf = io.StringIO()
            with open(log_path, "w", encoding="utf-8") as flog, redirect_stdout(flog), redirect_stderr(flog):
                print(f"# Programmatic Phase D overlap generation")
                print(f"# Hub: {hub}")
                print(f"# Phase C root: {phase_c_root}")
                print(f"# Phase D root: {phase_d_root}")
                # Derive inputs/outputs
                dose_dir = Path(phase_c_root) / f"dose_{int(dose)}"
                train_npz = dose_dir / "patched_train.npz"
                test_npz = dose_dir / "patched_test.npz"
                out_dir = Path(phase_d_root) / f"dose_{int(dose)}" / f"{view}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # Parameter defaults (prior study defaults)
                if view == "dense":
                    gridsize = 2
                    s_img = 0.8
                    n_groups = 512
                else:
                    gridsize = 1
                    s_img = 1.0
                    n_groups = 512
                neighbor_count = 25
                probe_diameter_px = 38.4
                rng_seed_subsample = 456

                print(f"Using parameters: gridsize={gridsize}, s_img={s_img}, n_groups={n_groups}, K={neighbor_count}, D={probe_diameter_px}, seed={rng_seed_subsample}")
                # Import and execute
                from studies.fly64_dose_overlap.overlap import generate_overlap_views
                try:
                    generate_overlap_views(
                        train_path=train_npz,
                        test_path=test_npz,
                        output_dir=out_dir,
                        gridsize=gridsize,
                        s_img=s_img,
                        n_groups=n_groups,
                        neighbor_count=neighbor_count,
                        probe_diameter_px=probe_diameter_px,
                        rng_seed_subsample=rng_seed_subsample,
                    )
                    print("Programmatic Phase D complete.")
                except Exception as e:
                    print(f"[run_phase_g_dense] ERROR during programmatic Phase D: {e}")
                    # Write blocker log
                    blocker_path = Path(phase_g_root) / "blocker.log"
                    blocker_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(blocker_path, "w", encoding="utf-8") as blocker:
                        blocker.write("Blocked during Phase D overlap generation (programmatic)\n")
                        blocker.write(f"Exception: {e}\n")
                        blocker.write(f"Log: {log_path}\n")
                    return 1
        else:
            run_command(cmd, log_path)

        # After Phase C completes, clean up unwanted dose directories and validate metadata
        # (Skip this guard when --collect-only to keep dry runs fast)
        if not args.collect_only and "Phase C:" in desc:
            # Clean up unwanted dose directories (generation module creates all doses)
            # TYPE-PATH-001: Normalize to Path
            import shutil
            phase_c_root_path = Path(phase_c_root).resolve()
            requested_dose_dir = phase_c_root_path / f"dose_{int(dose)}"

            print("\n" + "=" * 80)
            print(f"[run_phase_g_dense] Cleaning up unwanted dose directories (keeping dose_{int(dose)} only)...")
            print("=" * 80 + "\n")

            if phase_c_root_path.exists():
                for dose_dir in phase_c_root_path.iterdir():
                    if dose_dir.is_dir() and dose_dir.name.startswith("dose_") and dose_dir != requested_dose_dir:
                        print(f"[run_phase_g_dense] Removing unwanted dose directory: {dose_dir.name}")
                        shutil.rmtree(dose_dir)

            print("\n" + "=" * 80)
            print("[run_phase_g_dense] Validating Phase C metadata...")
            print("=" * 80 + "\n")
            try:
                validate_phase_c_metadata(hub)
            except RuntimeError as e:
                print(f"[run_phase_g_dense] ERROR during Phase C metadata validation: {e}", file=sys.stderr)
                # Write blocker log
                blocker_path = phase_g_root / "blocker.log"
                blocker_path.parent.mkdir(parents=True, exist_ok=True)
                with blocker_path.open("w", encoding="utf-8") as blocker:
                    blocker.write("Blocked during Phase C metadata validation\n")
                    blocker.write(f"Exception: {e}\n")
                return 1

    print("\n" + "=" * 80)
    print("[run_phase_g_dense] SUCCESS: All phases completed")
    print("=" * 80)

    # Summarize metrics from Phase G comparison results
    print("\n" + "=" * 80)
    print("[run_phase_g_dense] Summarizing Phase G metrics...")
    print("=" * 80 + "\n")
    try:
        summarize_phase_g_outputs(hub)
    except Exception as e:
        print(f"[run_phase_g_dense] ERROR during metrics summarization: {e}", file=sys.stderr)
        # Write blocker log
        blocker_path = phase_g_root / "blocker.log"
        blocker_path.parent.mkdir(parents=True, exist_ok=True)
        with blocker_path.open("w", encoding="utf-8") as blocker:
            blocker.write("Blocked during metrics summarization\n")
            blocker.write(f"Exception: {e}\n")
        return 1

    # Generate aggregate metrics report using the reporting helper
    print("\n" + "=" * 80)
    print("[run_phase_g_dense] Generating aggregate metrics report...")
    print("=" * 80 + "\n")
    run_command(report_helper_cmd, report_helper_log)

    # Print highlights preview to stdout for quick sanity-check
    print("\n" + "=" * 80)
    print("[run_phase_g_dense] Aggregate highlights preview")
    print("=" * 80 + "\n")

    # Read and display highlights file (TYPE-PATH-001 compliance: use Path)
    highlights_path = Path(aggregate_highlights_txt)
    if not highlights_path.exists():
        raise RuntimeError(
            f"Highlights file not found at: {highlights_path}\n"
            f"The reporting helper should have created this file.\n"
            f"Check the reporting helper log at: {report_helper_log}"
        )

    highlights_content = highlights_path.read_text(encoding="utf-8")
    if not highlights_content.strip():
        raise RuntimeError(
            f"Highlights file is empty: {highlights_path}\n"
            f"The reporting helper should have written highlights content.\n"
            f"Check the reporting helper log at: {report_helper_log}"
        )

    # Print highlights content to stdout
    print(highlights_content)
    print("=" * 80 + "\n")

    # Generate final metrics digest
    print("\n" + "=" * 80)
    print("[run_phase_g_dense] Generating metrics digest...")
    print("=" * 80 + "\n")
    run_command(analyze_digest_cmd, analyze_digest_log)

    # Print key delta summary to stdout (sourced from metrics_summary.json)
    print("\n" + "=" * 80)
    print("[run_phase_g_dense] Key Metrics Deltas (PtychoPINN vs Baselines)")
    print("=" * 80 + "\n")

    # Load metrics_summary.json to compute deltas using the helper
    metrics_summary_path = Path(phase_g_root) / "metrics_summary.json"
    if not metrics_summary_path.exists():
        print(f"Warning: metrics_summary.json not found at {metrics_summary_path}, skipping delta summary")
    else:
        try:
            import json
            with metrics_summary_path.open("r", encoding="utf-8") as f:
                summary_data = json.load(f)

            # Extract aggregate_metrics for delta computation
            agg = summary_data.get("aggregate_metrics", {})

            # Call helper to persist delta highlights + preview + JSON (returns delta_summary)
            delta_summary = persist_delta_highlights(
                aggregate_metrics=agg,
                output_dir=Path(phase_g_root),
                hub=hub
            )

            # Read back the highlights file to print to stdout
            highlights_txt_path = Path(phase_g_root) / "metrics_delta_highlights.txt"
            with highlights_txt_path.open("r", encoding="utf-8") as f:
                highlights_content = f.read()

            # Print delta block to stdout (4 lines read from file)
            print(highlights_content, end="")
            print("\nNote: For MS-SSIM, positive Δ indicates PtychoPINN is better (higher similarity).")
            print("      For MAE, negative Δ indicates PtychoPINN is better (lower error).")
            print("=" * 80 + "\n")

            print(f"Delta highlights saved to: {highlights_txt_path.relative_to(hub)}")

            # Announce preview file (TYPE-PATH-001)
            preview_txt_path = Path(phase_g_root) / "metrics_delta_highlights_preview.txt"
            if preview_txt_path.exists():
                print(f"Delta highlights preview (phase-only) saved to: {preview_txt_path.relative_to(hub)}")

            # Announce JSON file (TYPE-PATH-001)
            delta_json_path = Path(phase_g_root) / "metrics_delta_summary.json"
            print(f"Delta metrics saved to: {delta_json_path.relative_to(hub)}")

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Failed to parse metrics_summary.json for delta computation: {e}")

    # Generate SSIM grid summary (phase-only preview validation)
    print("\n" + "=" * 80)
    print("[run_phase_g_dense] Generating SSIM grid summary (phase-only)...")
    print("=" * 80 + "\n")
    run_command(ssim_grid_cmd, ssim_grid_log)

    # Post-Verify Automation: Verify pipeline artifacts and check highlights match
    if not args.skip_post_verify:
        print("\n" + "=" * 80)
        print("[run_phase_g_dense] Running post-verify automation...")
        print("=" * 80 + "\n")

        # Step 1: Verify pipeline artifacts
        print("[run_phase_g_dense] Verifying pipeline artifacts...")
        run_command(verify_cmd, verify_log)

        # Step 2: Check highlights match
        print("[run_phase_g_dense] Checking highlights match...")
        run_command(check_cmd, check_log)

        print("\n[run_phase_g_dense] Post-verify automation completed successfully")
    else:
        print("\n[run_phase_g_dense] Post-verify automation skipped (--skip-post-verify)")

    # Re-generate artifact inventory after post-verify automation
    print("\n" + "=" * 80)
    print("[run_phase_g_dense] Generating artifact inventory...")
    print("=" * 80 + "\n")
    generate_artifact_inventory(hub)

    # Validate artifact inventory file exists and announce its location (TYPE-PATH-001, DATA-001)
    artifact_inventory_path = Path(phase_g_root) / "artifact_inventory.txt"
    if not artifact_inventory_path.exists():
        raise RuntimeError(
            f"Artifact inventory file not found: {artifact_inventory_path}\n"
            f"generate_artifact_inventory() should have created this file.\n"
            f"This indicates a critical failure in artifact tracking."
        )

    print(f"\nArtifacts saved to: {hub}")
    print(f"Artifact inventory: {artifact_inventory_path.relative_to(hub)}")
    print(f"CLI logs: {cli_log_dir.relative_to(hub)}")
    print(f"Analysis outputs: {phase_g_root.relative_to(hub)}")
    print(f"Aggregate report: {aggregate_report_md.relative_to(hub)}")
    print(f"Highlights: {aggregate_highlights_txt.relative_to(hub)}")
    print(f"Metrics digest: {metrics_digest_md.relative_to(hub)}")
    print(f"Metrics digest log: {analyze_digest_log.relative_to(hub)}")

    # Add delta JSON, highlights, and preview to success banner (TYPE-PATH-001)
    delta_json_path = Path(phase_g_root) / "metrics_delta_summary.json"
    if delta_json_path.exists():
        print(f"Delta metrics (JSON): {delta_json_path.relative_to(hub)}")

    metrics_delta_highlights_path = Path(phase_g_root) / "metrics_delta_highlights.txt"
    if metrics_delta_highlights_path.exists():
        print(f"Delta highlights (TXT): {metrics_delta_highlights_path.relative_to(hub)}")

    metrics_delta_preview_path = Path(phase_g_root) / "metrics_delta_highlights_preview.txt"
    if metrics_delta_preview_path.exists():
        print(f"Delta highlights preview (phase-only, TXT): {metrics_delta_preview_path.relative_to(hub)}")

    # Add ssim_grid summary to success banner (TYPE-PATH-001)
    ssim_grid_summary_path = Path(phase_g_root) / "ssim_grid_summary.md"
    if ssim_grid_summary_path.exists():
        print(f"SSIM Grid Summary (phase-only): {ssim_grid_summary_path.relative_to(hub)}")

    ssim_grid_log_path = Path(cli_log_dir) / "ssim_grid_cli.log"
    if ssim_grid_log_path.exists():
        print(f"SSIM Grid log: {ssim_grid_log_path.relative_to(hub)}")

    # Add post-verify artifacts to success banner (TYPE-PATH-001)
    if not args.skip_post_verify:
        verify_report_path = Path(phase_g_root) / "verification_report.json"
        if verify_report_path.exists():
            print(f"Verification report: {verify_report_path.relative_to(hub)}")

        verify_log_path = Path(phase_g_root) / "verify_dense_stdout.log"
        if verify_log_path.exists():
            print(f"Verification log: {verify_log_path.relative_to(hub)}")

        check_log_path = Path(phase_g_root) / "check_dense_highlights.log"
        if check_log_path.exists():
            print(f"Highlights check log: {check_log_path.relative_to(hub)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
