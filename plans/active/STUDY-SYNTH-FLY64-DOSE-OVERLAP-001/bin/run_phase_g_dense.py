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

    print("\n" + "=" * 80)
    print("[run_phase_g_dense] SUCCESS: All phases completed")
    print("=" * 80)
    print(f"\nArtifacts saved to: {hub}")
    print(f"CLI logs: {cli_log_dir}")
    print(f"Analysis outputs: {phase_g_root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
