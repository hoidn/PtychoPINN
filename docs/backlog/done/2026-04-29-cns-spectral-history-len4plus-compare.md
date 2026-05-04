---
priority: 28
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-spectral-history-len4plus-compare/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites:
  - 2026-04-29-pdebench-cns-history-len3plus-compare
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
signals_for_selection:
  - The completed history_len=3 CNS study improved the spectral-bottleneck row over history_len=2, but did not establish whether the gain saturates at three frames.
  - The manuscript currently reports the spectral-bottleneck CNS row as SRU-Net*, so any longer-history result must preserve the repo-row mapping explicitly.
  - The result is useful only if deltas are reported against the frozen history_len=2 and history_len=3 anchors under a matching reduced training setting.
---

# Backlog Item: Test CNS Spectral History Length Beyond 3

## Objective

- Determine whether increasing PDEBench CNS temporal context beyond
  `history_len=3` continues to improve the spectral-bottleneck CNS row
  (`spectral_resnet_bottleneck_base`, currently labeled `SRU-Net*` in the
  manuscript), and quantify the improvement or saturation.

## Scope

- Use the same official `2d_cfd_cns` dataset, reduced training setting,
  normalization policy, MSE loss, batch size, metric family, and epoch budget
  used by the completed capped CNS history-length study unless the fresh plan
  records a pre-run reason to change the cap and rerun all compared anchors.
- Run `history_len=4` for `spectral_resnet_bottleneck_base` first.
- Compare the fresh `history_len=4` row against the frozen audited
  `history_len=2` and `history_len=3` spectral-bottleneck anchors.
- Report absolute metrics and deltas for at least nRMSE, RMSE, relative L2,
  low/mid/high Fourier-band RMSE, train/eval sample counts, runtime, and valid
  windows per trajectory.
- Run `history_len=5` only if `history_len=4` improves the spectral-bottleneck
  row on aggregate error without an unacceptable high-band regression, or if
  the plan records a concrete pre-run scientific reason to test a second
  longer-history point.
- Keep this item focused on the spectral-bottleneck row. Do not rerun FNO,
  U-Net, authored FFNO, or the legacy `hybrid_resnet_cns` continuity row unless
  a later roadmap decision asks for a full same-history table.

## Notes for Reviewer

- Do not mix history lengths in a model-ranking table. This item is a temporal
  context ablation for one row, not a replacement CNS headline bundle.
- Record that `SRU-Net*` is the manuscript-facing label and
  `spectral_resnet_bottleneck_base` is the repo row. Keep the mapping in all
  machine-readable outputs and prose summaries.
- Longer histories reduce the number of valid windows per trajectory. The
  summary must state whether the comparison remains equal-footing enough for a
  direct delta claim or should be framed only as context-ablation evidence.
- If `history_len=4` or `history_len=5` improves, report by how much relative to
  both `history_len=2` and `history_len=3`; if not, report the saturation point
  and the first regressed metric.
- Do not turn this into an autoregressive rollout, full-training benchmark, or
  broad architecture sweep.
