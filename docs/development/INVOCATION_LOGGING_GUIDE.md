# Invocation Logging Guide

## Purpose

Invocation logging makes studies reproducible and auditable by preserving the
exact command line and parsed arguments used to launch each run. This is used
for debugging, run provenance, and handoff across collaborators.

## Canonical Artifacts

Developer rule: study scripts and orchestration entrypoints under
`scripts/studies/` MUST write:

- `invocation.json`: machine-readable command metadata
- `invocation.sh`: copy-paste command line used to launch the script

## Recommended Output Locations

Use deterministic locations relative to run outputs.

- Single-model script runs:
  - write to that model run directory (for example
    `OUTPUT_DIR/runs/pinn_<architecture>/invocation.*`)
- Multi-model wrappers/orchestrators:
  - write to wrapper run root (for example `OUTPUT_DIR/invocation.*`)
- Nested orchestration chains:
  - each child script writes to its own run directory
  - parent wrapper writes to parent run root
  - include parent metadata in child `invocation.json` via `extra=...`

## Implementation Pattern

At CLI entrypoint time:

1. Capture raw argv from `argv` (testable path) or `sys.argv[1:]`
2. Parse args with `argparse`
3. Write invocation artifacts before launching expensive work
4. Include optional parent context for nested workflows

Minimum metadata in `invocation.json`:

- `script`
- `argv`
- `command`
- `parsed_args`
- `cwd`
- `timestamp_utc`
- `pid`

## Drop-in Example

```python
import sys
from scripts.studies.invocation_logging import write_invocation_artifacts

def main(argv=None):
    args = parse_args(argv)
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    write_invocation_artifacts(
        output_dir=args.output_dir,
        script_path="scripts/studies/my_entrypoint.py",
        argv=raw_argv,
        parsed_args=vars(args),
        extra={"parent_run_id": args.parent_run_id} if args.parent_run_id else None,
    )
    run_workflow(args)
```

## Testing Checklist

For each study CLI/orchestrator, include tests that validate:

- `invocation.json` is created at the expected path
- `invocation.sh` is created at the expected path
- `command` includes the script path and key flags
- `argv` preserves expected flags and values
- `parsed_args` serializes paths and non-JSON-native values correctly

## Anti-Patterns

- Logging only a subset of flags (lossy provenance)
- Writing artifacts in non-deterministic locations
- Formatting commands ad hoc rather than using shared helper
- Writing invocation artifacts only after long-running execution starts
- Omitting parsed args snapshot for derived/defaulted values
