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
    # Phase C generation creates dose_<dose>_<split> directories with
    # fly64_<split>_simulated.npz files (see studies/fly64_dose_overlap/generation.py)
    splits_to_check = ["train", "test"]

    for split in splits_to_check:
        # Find the split directory (pattern: dose_*_<split>)
        split_dirs = list(phase_c_root.glob(f"dose_*_{split}"))

        if not split_dirs:
            raise RuntimeError(
                f"Phase C {split} split directory not found under {phase_c_root}. "
                f"Expected pattern: dose_*_{split}/"
            )

        # Should only be one per split, but check all if multiple exist
        for split_dir in split_dirs:
            # Find NPZ file (pattern: fly64_<split>_simulated.npz)
            npz_files = list(split_dir.glob(f"fly64_{split}_simulated.npz"))

            if not npz_files:
                raise RuntimeError(
                    f"Phase C NPZ file not found in {split_dir}. "
                    f"Expected: fly64_{split}_simulated.npz"
                )

            for npz_path in npz_files:
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


def summarize_phase_g_outputs(hub: Path) -> None:
    """
    Parse comparison_manifest.json and per-job metrics CSVs to emit a metrics summary.

    This helper validates successful Phase G execution, extracts key metrics
    (MS-SSIM, MAE) from per-job comparison_metrics.csv files, and writes both
    JSON and Markdown summaries to {hub}/analysis/.

    Args:
        hub: Root directory containing analysis/ subdirectory with comparison_manifest.json

    Raises:
        RuntimeError: If manifest is missing, n_failed > 0, or required CSV files are absent

    Follows TYPE-PATH-001 (Path normalization).
    """
    # TYPE-PATH-001: Normalize to Path
    hub = Path(hub).resolve()
    analysis = hub / "analysis"

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

    print(f"[summarize_phase_g_outputs] Wrote Markdown summary: {md_summary_path}")


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

    args = parser.parse_args()

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
        "python", "-m", "studies.fly64_dose_overlap.generation",
        "--base-npz", str(base_npz),
        "--output-root", str(phase_c_root),
    ]
    commands.append(("Phase C: Dataset Generation", phase_c_cmd, cli_log_dir / "phase_c_generation.log"))

    # Phase D: Overlap View Generation
    phase_d_cmd = [
        "python", "-m", "studies.fly64_dose_overlap.overlap",
        "--phase-c-root", str(phase_c_root),
        "--output-root", str(phase_d_root),
        "--doses", str(dose),
        "--views", view,
        "--artifact-root", str(phase_g_root),
    ]
    commands.append(("Phase D: Overlap View Generation", phase_d_cmd, cli_log_dir / f"phase_d_{view}.log"))

    # Phase E: Training (baseline gs1 + view gs2)
    # Baseline (gs1)
    phase_e_baseline_cmd = [
        "python", "-m", "studies.fly64_dose_overlap.training",
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
        "python", "-m", "studies.fly64_dose_overlap.training",
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
            "python", "-m", "studies.fly64_dose_overlap.reconstruction",
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
            "python", "-m", "studies.fly64_dose_overlap.comparison",
            "--phase-c-root", str(phase_c_root),
            "--phase-e-root", str(phase_e_root),
            "--phase-f-root", str(phase_f_root),
            "--artifact-root", str(phase_g_root),
            "--dose", str(dose),
            "--view", view,
            "--split", split,
        ]
        commands.append((f"Phase G: Comparison {view}/{split}", phase_g_cmd, cli_log_dir / f"phase_g_{view}_{split}.log"))

    # Collect-only mode: print commands and exit
    if args.collect_only:
        print("[run_phase_g_dense] Collect-only mode: planned commands:")
        for i, (desc, cmd, log_path) in enumerate(commands, 1):
            print(f"\n{i}. {desc}")
            print(f"   Command: {' '.join(str(c) for c in cmd)}")
            print(f"   Log: {log_path}")
        return 0

    # Execute all commands in sequence
    print(f"[run_phase_g_dense] Starting Phase C→G pipeline for dose={dose}, view={view}, splits={splits}")
    print(f"[run_phase_g_dense] Hub: {hub}")
    print(f"[run_phase_g_dense] Total commands: {len(commands)}\n")

    for i, (desc, cmd, log_path) in enumerate(commands, 1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(commands)}] {desc}")
        print(f"{'=' * 80}\n")
        run_command(cmd, log_path)

        # After Phase C completes, validate metadata
        # (Skip this guard when --collect-only to keep dry runs fast)
        if not args.collect_only and "Phase C:" in desc:
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

    print(f"\nArtifacts saved to: {hub}")
    print(f"CLI logs: {cli_log_dir}")
    print(f"Analysis outputs: {phase_g_root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
