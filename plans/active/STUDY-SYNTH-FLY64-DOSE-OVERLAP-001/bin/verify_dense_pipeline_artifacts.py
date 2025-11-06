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

    # 7. CLI logs (phase_g_*.log)
    phase_g_logs = sorted(cli_dir.glob("phase_g_*.log")) if cli_dir.exists() else []
    validations.append({
        'valid': len(phase_g_logs) > 0,
        'description': 'Phase G CLI logs',
        'path': str(cli_dir),
        'found_count': len(phase_g_logs),
        'log_files': [log.name for log in phase_g_logs] if phase_g_logs else [],
        'error': 'No phase_g_*.log files found' if not phase_g_logs else None,
    })

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
