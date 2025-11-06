#!/usr/bin/env python3
"""
Phase G Dense Pipeline Artifact Verification Script

Validates that all expected Phase G outputs exist after orchestrator execution,
including metrics bundle, metadata compliance results, and analysis artifacts.

Exit codes:
  0 - Success (all required artifacts present and valid)
  1 - Validation failures (missing files or invalid structure)
  2 - Invalid arguments or I/O errors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def validate_file_exists(path: Path, description: str) -> dict[str, Any]:
    """
    Validate that a file exists and return validation result.

    Args:
        path: Path to check
        description: Human-readable description for error messages

    Returns:
        dict with 'valid' (bool), 'path' (str), 'description' (str),
        and optional 'error' (str) keys
    """
    result = {
        'valid': path.exists() and path.is_file(),
        'path': str(path),
        'description': description,
    }
    if not result['valid']:
        if not path.exists():
            result['error'] = 'File not found'
        elif not path.is_file():
            result['error'] = 'Path is not a file'
    return result


def validate_json_file(path: Path, description: str, required_keys: list[str] | None = None) -> dict[str, Any]:
    """
    Validate that a JSON file exists and contains required keys.

    Args:
        path: Path to JSON file
        description: Human-readable description
        required_keys: Optional list of top-level keys to validate

    Returns:
        dict with validation result and optional data preview
    """
    result = validate_file_exists(path, description)
    if not result['valid']:
        return result

    try:
        with path.open('r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result['valid'] = False
        result['error'] = f'Invalid JSON: {e}'
        return result
    except Exception as e:
        result['valid'] = False
        result['error'] = f'Failed to read file: {e}'
        return result

    if required_keys:
        missing = [k for k in required_keys if k not in data]
        if missing:
            result['valid'] = False
            result['error'] = f'Missing required keys: {", ".join(missing)}'
            result['missing_keys'] = missing
        else:
            result['found_keys'] = required_keys

    # Add data preview (top-level keys)
    result['data_keys'] = list(data.keys())

    return result


def validate_phase_c_metadata_compliance(metrics_summary_path: Path) -> dict[str, Any]:
    """
    Validate Phase C metadata compliance from metrics_summary.json.

    Args:
        metrics_summary_path: Path to metrics_summary.json

    Returns:
        dict with validation result and compliance summary
    """
    result = {
        'valid': False,
        'description': 'Phase C metadata compliance',
    }

    if not metrics_summary_path.exists():
        result['error'] = f'Metrics summary not found: {metrics_summary_path}'
        return result

    try:
        with metrics_summary_path.open('r') as f:
            data = json.load(f)
    except Exception as e:
        result['error'] = f'Failed to load metrics summary: {e}'
        return result

    compliance = data.get('phase_c_metadata_compliance')
    if not compliance:
        result['error'] = 'Missing phase_c_metadata_compliance field'
        return result

    if 'error' in compliance:
        result['valid'] = False
        result['error'] = f"Phase C compliance check failed: {compliance['error']}"
        return result

    # Count compliant and non-compliant splits
    total = 0
    compliant_count = 0
    non_compliant = []

    for dose_key, dose_data in compliance.items():
        if isinstance(dose_data, dict):
            for split, split_data in dose_data.items():
                if isinstance(split_data, dict):
                    total += 1
                    if split_data.get('compliant', False):
                        compliant_count += 1
                    else:
                        non_compliant.append(f"{dose_key}/{split}")

    result['valid'] = total > 0 and compliant_count == total
    result['total_splits'] = total
    result['compliant_splits'] = compliant_count
    result['non_compliant_splits'] = len(non_compliant)

    if non_compliant:
        result['non_compliant_list'] = non_compliant
        result['error'] = f'{len(non_compliant)} splits non-compliant: {", ".join(non_compliant)}'

    return result


def validate_metrics_digest(digest_path: Path) -> dict[str, Any]:
    """
    Validate metrics digest Markdown file.

    Args:
        digest_path: Path to metrics_digest.md

    Returns:
        dict with validation result and content summary
    """
    result = validate_file_exists(digest_path, 'Metrics digest')
    if not result['valid']:
        return result

    try:
        content = digest_path.read_text()
    except Exception as e:
        result['valid'] = False
        result['error'] = f'Failed to read file: {e}'
        return result

    # Check for expected sections
    expected_sections = [
        '# Phase G Dense Metrics Digest',
        '## Pipeline Summary',
        '## Highlights',
    ]

    missing_sections = []
    for section in expected_sections:
        if section not in content:
            missing_sections.append(section)

    if missing_sections:
        result['valid'] = False
        result['error'] = f'Missing sections: {", ".join(missing_sections)}'
        result['missing_sections'] = missing_sections
    else:
        result['line_count'] = len(content.splitlines())
        result['char_count'] = len(content)

    return result


def validate_metrics_delta_summary(delta_json_path: Path, hub: Path) -> dict[str, Any]:
    """
    Validate metrics_delta_summary.json structure and provenance fields.

    Args:
        delta_json_path: Path to metrics_delta_summary.json
        hub: Hub directory root for validating source_metrics path

    Returns:
        dict with validation result
    """
    result = {
        'valid': False,
        'description': 'Metrics delta summary JSON',
        'path': str(delta_json_path),
    }

    if not delta_json_path.exists():
        result['error'] = 'File not found'
        return result

    try:
        with delta_json_path.open('r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result['error'] = f'Invalid JSON: {e}'
        return result
    except Exception as e:
        result['error'] = f'Failed to read file: {e}'
        return result

    # Validate required top-level fields
    required_fields = ['generated_at', 'source_metrics', 'deltas']
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        result['error'] = f'Missing required fields: {", ".join(missing_fields)}'
        result['missing_fields'] = missing_fields
        return result

    # Validate generated_at is ISO-8601 UTC timestamp
    generated_at = data.get('generated_at')
    if not isinstance(generated_at, str) or not generated_at.endswith('Z'):
        result['error'] = f'generated_at must be ISO-8601 UTC timestamp ending with Z, got: {generated_at}'
        return result

    # Validate source_metrics points to existing file within hub
    source_metrics_rel = data.get('source_metrics')
    if not isinstance(source_metrics_rel, str):
        result['error'] = f'source_metrics must be string, got: {type(source_metrics_rel).__name__}'
        return result

    source_metrics_path = hub / source_metrics_rel
    if not source_metrics_path.exists():
        result['error'] = f'source_metrics path does not exist: {source_metrics_path}'
        result['source_metrics_path'] = str(source_metrics_path)
        return result

    # Validate deltas structure
    deltas = data.get('deltas')
    if not isinstance(deltas, dict):
        result['error'] = 'deltas field must be a dict'
        return result

    # Check for required comparison keys
    required_comparisons = ['vs_Baseline', 'vs_PtyChi']
    missing_comparisons = [c for c in required_comparisons if c not in deltas]
    if missing_comparisons:
        result['error'] = f'deltas missing required comparisons: {", ".join(missing_comparisons)}'
        result['missing_comparisons'] = missing_comparisons
        return result

    # Validate each comparison has ms_ssim and mae with amplitude/phase pairs
    for comp_name in required_comparisons:
        comp_data = deltas[comp_name]
        if not isinstance(comp_data, dict):
            result['error'] = f'deltas.{comp_name} must be a dict'
            return result

        required_metrics = ['ms_ssim', 'mae']
        missing_metrics = [m for m in required_metrics if m not in comp_data]
        if missing_metrics:
            result['error'] = f'deltas.{comp_name} missing metrics: {", ".join(missing_metrics)}'
            result['missing_metrics'] = missing_metrics
            return result

        for metric_name in required_metrics:
            metric_data = comp_data[metric_name]
            if not isinstance(metric_data, dict):
                result['error'] = f'deltas.{comp_name}.{metric_name} must be a dict'
                return result

            required_components = ['amplitude', 'phase']
            missing_components = [c for c in required_components if c not in metric_data]
            if missing_components:
                result['error'] = f'deltas.{comp_name}.{metric_name} missing components: {", ".join(missing_components)}'
                result['missing_components'] = missing_components
                return result

            # Values can be None (if source metrics unavailable) or numeric
            for component in required_components:
                val = metric_data[component]
                if val is not None and not isinstance(val, (int, float)):
                    result['error'] = f'deltas.{comp_name}.{metric_name}.{component} must be numeric or None, got: {type(val).__name__}'
                    return result

    result['valid'] = True
    result['found_fields'] = list(data.keys())
    result['source_metrics_exists'] = True
    return result


def validate_metrics_delta_highlights(highlights_txt_path: Path) -> dict[str, Any]:
    """
    Validate metrics_delta_highlights.txt has exactly 4 lines with expected format.

    Args:
        highlights_txt_path: Path to metrics_delta_highlights.txt

    Returns:
        dict with validation result
    """
    result = validate_file_exists(highlights_txt_path, 'Metrics delta highlights')
    if not result['valid']:
        return result

    try:
        content = highlights_txt_path.read_text()
    except Exception as e:
        result['valid'] = False
        result['error'] = f'Failed to read file: {e}'
        return result

    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]

    # Must have exactly 4 lines (MS-SSIM + MAE for Baseline and PtyChi)
    if len(lines) != 4:
        result['valid'] = False
        result['error'] = f'Expected exactly 4 lines, got {len(lines)}'
        result['line_count'] = len(lines)
        return result

    # Validate expected prefixes
    expected_prefixes = [
        'MS-SSIM Δ (PtychoPINN - Baseline)',
        'MS-SSIM Δ (PtychoPINN - PtyChi)',
        'MAE Δ (PtychoPINN - Baseline)',
        'MAE Δ (PtychoPINN - PtyChi)',
    ]

    for i, (line, expected_prefix) in enumerate(zip(lines, expected_prefixes)):
        if not line.startswith(expected_prefix):
            result['valid'] = False
            result['error'] = f'Line {i+1} does not start with expected prefix: {expected_prefix}'
            result['invalid_line'] = line
            return result

    result['line_count'] = len(lines)
    return result


def validate_artifact_inventory(inventory_path: Path, hub: Path) -> dict[str, Any]:
    """
    Validate artifact_inventory.txt exists and contains POSIX-relative paths.

    Args:
        inventory_path: Path to artifact_inventory.txt
        hub: Hub directory root for validating entries are relative to hub

    Returns:
        dict with validation result

    TYPE-PATH-001: All paths must be POSIX-relative (no absolute paths, no backslashes)
    """
    result = {
        'valid': False,
        'description': 'Artifact inventory',
        'path': str(inventory_path),
    }

    if not inventory_path.exists():
        result['error'] = 'artifact_inventory.txt not found'
        return result

    try:
        content = inventory_path.read_text()
    except Exception as e:
        result['error'] = f'Failed to read file: {e}'
        return result

    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]

    if not lines:
        result['error'] = 'artifact_inventory.txt is empty'
        result['line_count'] = 0
        return result

    # Validate each path entry
    invalid_entries = []
    for i, line in enumerate(lines):
        # Check for absolute paths (starts with / or contains drive letter)
        if line.startswith('/') or (len(line) > 1 and line[1] == ':'):
            invalid_entries.append({
                'line': i + 1,
                'path': line,
                'reason': 'Absolute path (must be relative to hub)'
            })
            continue

        # Check for backslashes (Windows-style paths)
        if '\\' in line:
            invalid_entries.append({
                'line': i + 1,
                'path': line,
                'reason': 'Contains backslashes (must use POSIX forward slashes)'
            })
            continue

        # Check that referenced file exists relative to hub
        artifact_path = hub / line
        if not artifact_path.exists():
            invalid_entries.append({
                'line': i + 1,
                'path': line,
                'reason': f'Referenced file not found: {artifact_path}'
            })

    if invalid_entries:
        result['valid'] = False
        result['error'] = f'{len(invalid_entries)} invalid entries in artifact_inventory.txt'
        result['invalid_entries'] = invalid_entries
        result['line_count'] = len(lines)
        return result

    result['valid'] = True
    result['line_count'] = len(lines)
    result['entries'] = lines
    return result


def validate_cli_logs(cli_dir: Path) -> dict[str, Any]:
    """
    Validate CLI logs from run_phase_g_dense.py orchestrator.

    Checks for:
    - Existence of run_phase_g_dense.log or phase_*_generation.log
    - Phase banners [1/8] through [8/8] in orchestrator log
    - SUCCESS sentinel: "SUCCESS: All phases completed"
    - Per-phase log files for all 8 phases

    Args:
        cli_dir: Path to cli/ directory containing orchestrator logs

    Returns:
        dict with validation result including:
        - valid: bool
        - description: str
        - path: str
        - error: str (if invalid)
        - found_logs: list[str] (names of found log files)
        - missing_banners: list[str] (if any phase banners missing)
        - has_success: bool (whether SUCCESS marker found)
        - found_phase_logs: list[str] (names of found per-phase log files)
        - missing_phase_logs: list[str] (names of expected but missing per-phase log files)
    """
    result = {
        'valid': True,
        'description': 'CLI orchestrator logs validation',
        'path': str(cli_dir),
    }

    if not cli_dir.exists():
        result['valid'] = False
        result['error'] = f'CLI directory not found: {cli_dir}'
        return result

    # Look for orchestrator log (run_phase_g_dense.log) - REQUIRED
    orchestrator_log = cli_dir / "run_phase_g_dense.log"
    phase_logs = list(cli_dir.glob("phase_*.log"))

    found_logs = []
    if orchestrator_log.exists():
        found_logs.append(orchestrator_log.name)
    found_logs.extend([p.name for p in phase_logs])

    result['found_logs'] = found_logs

    if not orchestrator_log.exists():
        result['valid'] = False
        result['error'] = 'Orchestrator log not found: run_phase_g_dense.log is required'
        return result

    # Validate orchestrator log content
    if orchestrator_log.exists():
        try:
            content = orchestrator_log.read_text()
        except Exception as e:
            result['valid'] = False
            result['error'] = f'Failed to read orchestrator log: {e}'
            return result

        # Check for phase banners [1/8] through [8/8]
        expected_banners = [f'[{i}/8]' for i in range(1, 9)]
        missing_banners = []
        for banner in expected_banners:
            if banner not in content:
                missing_banners.append(banner)

        # Check for SUCCESS marker
        has_success = 'SUCCESS: All phases completed' in content

        result['missing_banners'] = missing_banners
        result['has_success'] = has_success

        if missing_banners:
            result['valid'] = False
            result['error'] = f'Missing phase banners in orchestrator log: {", ".join(missing_banners)}'
            return result

        if not has_success:
            result['valid'] = False
            result['error'] = 'Orchestrator log missing SUCCESS sentinel: "SUCCESS: All phases completed"'
            return result

    # Check for required per-phase log files
    # Expected phase logs based on the 8-phase pipeline:
    # Phase C: generation, Phase D: dense view, Phase E: baseline + dense training,
    # Phase F: train + test reconstruction, Phase G: train + test comparison
    expected_phase_logs = [
        "phase_c_generation.log",
        "phase_d_dense.log",
        "phase_e_baseline.log",
        "phase_e_dense.log",
        "phase_f_train.log",
        "phase_f_test.log",
        "phase_g_train.log",
        "phase_g_test.log",
    ]

    found_phase_log_names = [p.name for p in phase_logs]
    missing_phase_logs = [log for log in expected_phase_logs if log not in found_phase_log_names]

    result['found_phase_logs'] = found_phase_log_names
    result['missing_phase_logs'] = missing_phase_logs

    if missing_phase_logs:
        result['valid'] = False
        result['error'] = f'Missing required per-phase log files: {", ".join(missing_phase_logs)}'
        return result

    # All checks passed
    result['valid'] = True
    return result


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Verify Phase G dense pipeline artifacts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--hub',
        type=Path,
        required=True,
        help='Hub directory root (contains data/, analysis/, cli/ subdirs)',
    )
    parser.add_argument(
        '--report',
        type=Path,
        required=True,
        help='Output path for verification report JSON',
    )

    args = parser.parse_args()

    # TYPE-PATH-001: Normalize to Path
    hub = Path(args.hub).resolve()
    report_path = Path(args.report).resolve()

    if not hub.exists():
        print(f"ERROR: Hub directory not found: {hub}", file=sys.stderr)
        return 2

    print(f"[verify_dense_pipeline_artifacts] Hub: {hub}")
    print(f"[verify_dense_pipeline_artifacts] Report: {report_path}")

    # Define expected artifacts
    analysis = hub / "analysis"
    cli_dir = hub / "cli"

    validations = []

    # 1. Comparison manifest
    validations.append(
        validate_json_file(
            analysis / "comparison_manifest.json",
            "Phase G comparison manifest",
            required_keys=['n_jobs', 'n_success', 'n_failed', 'jobs']
        )
    )

    # 2. Metrics summary JSON
    metrics_summary_path = analysis / "metrics_summary.json"
    validations.append(
        validate_json_file(
            metrics_summary_path,
            "Phase G metrics summary JSON",
            required_keys=['n_jobs', 'n_success', 'n_failed', 'aggregate_metrics', 'phase_c_metadata_compliance']
        )
    )

    # 3. Metrics summary Markdown
    validations.append(
        validate_file_exists(
            analysis / "metrics_summary.md",
            "Phase G metrics summary Markdown"
        )
    )

    # 4. Aggregate highlights
    validations.append(
        validate_file_exists(
            analysis / "aggregate_highlights.txt",
            "Phase G aggregate highlights"
        )
    )

    # 5. Metrics digest
    validations.append(
        validate_metrics_digest(analysis / "metrics_digest.md")
    )

    # 6. Phase C metadata compliance (extracted from metrics summary)
    if metrics_summary_path.exists():
        validations.append(
            validate_phase_c_metadata_compliance(metrics_summary_path)
        )

    # 7. Metrics delta summary JSON
    delta_json_path = analysis / "metrics_delta_summary.json"
    validations.append(
        validate_metrics_delta_summary(delta_json_path, hub)
    )

    # 8. Metrics delta highlights text
    delta_highlights_path = analysis / "metrics_delta_highlights.txt"
    validations.append(
        validate_metrics_delta_highlights(delta_highlights_path)
    )

    # 9. CLI orchestrator logs validation (phase banners + SUCCESS sentinel)
    validations.append(
        validate_cli_logs(cli_dir)
    )

    # 10. Artifact inventory (TYPE-PATH-001 compliance)
    inventory_path = analysis / "artifact_inventory.txt"
    validations.append(
        validate_artifact_inventory(inventory_path, hub)
    )

    # Aggregate results
    all_valid = all(v['valid'] for v in validations)
    n_valid = sum(1 for v in validations if v['valid'])
    n_invalid = len(validations) - n_valid

    report_data = {
        'hub': str(hub),
        'all_valid': all_valid,
        'n_checks': len(validations),
        'n_valid': n_valid,
        'n_invalid': n_invalid,
        'validations': validations,
    }

    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('w') as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[verify_dense_pipeline_artifacts] Verification Summary:")
    print(f"  Total checks: {len(validations)}")
    print(f"  Valid: {n_valid}")
    print(f"  Invalid: {n_invalid}")
    print(f"  Report: {report_path}")

    if not all_valid:
        print("\n[verify_dense_pipeline_artifacts] FAILURES:")
        for v in validations:
            if not v['valid']:
                desc = v['description']
                error = v.get('error', 'Unknown error')
                print(f"  ✗ {desc}: {error}")
        return 1

    print("\n[verify_dense_pipeline_artifacts] ✓ All checks passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
