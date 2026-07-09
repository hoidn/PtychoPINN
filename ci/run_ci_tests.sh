#!/usr/bin/env bash
# CI test gate: CPU-only, tracked-files-only subset of tests/torch.
# Exclusion baseline: ci/known_failures.txt (node IDs -> --deselect),
# ci/collect_ignores.txt (files -> --ignore). Policy: docs/ci.md.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=""

args=(tests/torch -m "not slow" -q -rf --color=yes)

while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  args+=(--deselect "$line")
done < ci/known_failures.txt

while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  args+=(--ignore "$line")
done < ci/collect_ignores.txt

# Extra args pass through to pytest (e.g. --collect-only, -k <expr>).
python -m pytest "${args[@]}" "$@"
