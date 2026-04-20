"""Entrypoint for PDEBench SWE longer one-step execution."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.studies.pdebench_swe.longer import main


if __name__ == "__main__":
    raise SystemExit(main())
