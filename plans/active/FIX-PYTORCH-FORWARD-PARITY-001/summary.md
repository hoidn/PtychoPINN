### Turn Summary
Bootstrapped FIX-PYTORCH-FORWARD-PARITY-001 now that the exporter initiative is complete and Tier‑3 dwell pressure forced the fly64 dense rerun focus to stay blocked.
Created the forward_parity hub at `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/` and staged the Phase A Do Now: add optional patch-stat instrumentation to `ptycho_torch/model.py` + `ptycho_torch/inference.py`, wire CLI flags (`--log-patch-stats/--patch-stats-limit`), and emit JSON + normalized PNG dumps under `$HUB/analysis/`.
Documented the short 10‑epoch baseline commands (train → inference) that must be rerun with instrumentation enabled plus the targeted pytest selector to keep the new flags covered.
Next: implement the instrumentation, capture the CLI + pytest logs under the new hub, and update the artifact inventory once the patch stats land.
Artifacts: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/{plan/plan.md}
