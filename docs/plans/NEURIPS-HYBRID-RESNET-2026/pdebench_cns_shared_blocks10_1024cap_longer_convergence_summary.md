# PDEBench CNS Shared-Blocks10 1024-Cap Longer-Convergence Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-pdebench-cns-shared-blocks10-1024cap-longer-convergence`
- Date: `2026-04-29`
- Status: implementation complete; capped `80`-epoch shared-blocks10 longer-convergence follow-up complete
- Scope: freeze the authoritative `1024 / 128 / 128`, `40`-epoch shared-blocks10 reference row, prove the exact `80`-epoch inspect contract, run one fresh `80`-epoch shared-blocks10 rerun, emit a convergence audit plus a shell-validated `40ep -> 80ep` delta payload, and publish a bounded interpretation against the frozen `40`-epoch row and the existing `2048cap` scaling summary
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-pdebench-cns-shared-blocks10-1024cap-longer-convergence/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence`

This summary records capped decision-support evidence only. It does not create a benchmark-complete CNS claim and does not justify promoting `spectral_resnet_bottleneck_shared_blocks10` into a default profile bundle.

## Fixed Fairness Contract

- task: `2d_cfd_cns`
- dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- split: `1024 / 128 / 128` trajectories
- `history_len=2`
- `max_windows_per_trajectory=8`
- training loss: `mse`
- batch size: `4`
- metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`

The shell stayed fixed at:

- `base_model="spectral_resnet_bottleneck_net"`
- `hidden_channels=32`
- `fno_modes=12`
- `fno_blocks=4`
- `hybrid_downsample_steps=2`
- `hybrid_resnet_blocks=6`
- `hybrid_skip_connections=True`
- `hybrid_skip_style="add"`
- `hybrid_upsampler="pixelshuffle"`
- `spectral_bottleneck_modes=12`
- `spectral_bottleneck_gate_init=0.1`
- `spectral_bottleneck_gate_mode="shared"`
- `spectral_bottleneck_share_weights=True`
- `spectral_bottleneck_blocks=10`

The only allowed contract delta for this item was the epoch budget:

- reference row: `40` epochs
- fresh rerun: `80` epochs

No code patching was needed during this closeout pass. The required reference-manifest, shell-contract, inspect, and reporting helpers already supported the approved contract.

## Frozen References And Fresh Run

Frozen reference artifacts:

- reference manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/reference_runs_1024cap_40ep.json`
- frozen shell contract:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/reference_shell_contract_shared_blocks10_1024cap.json`
- frozen `40`-epoch run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z`

Inspect proof:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/inspect-1024cap-80ep-20260429T025556Z`

Fresh `80`-epoch run root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/cns-shared-blocks10-1024cap-80ep-20260429T025641Z`

Run-completion proof:

- tracked PID sidecar: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/cns-shared-blocks10-1024cap-80ep-20260429T025641Z.launch/pid.txt`
- tracked exit code: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/cns-shared-blocks10-1024cap-80ep-20260429T025641Z.launch/exit_code.txt`
- required fresh artifacts present:
  - `comparison_summary.json`
  - `metrics_spectral_resnet_bottleneck_shared_blocks10.json`
  - `model_profile_spectral_resnet_bottleneck_shared_blocks10.json`
  - `dataset_manifest.json`
  - `split_manifest.json`
  - `invocation.json`

The launch sidecar needed one narrow recovery during execution: the tmux wrapper did not initially persist `pid.txt`, so the exact live child PID was recovered from the tmux pane process tree and backfilled before completion proof. The recovery evidence is archived at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/launch_pid_recovery.log`.

## Fresh 80-Epoch Outcome

Observed final train-loss trajectory:

- epoch `1`: `0.1221964386`
- epoch `5`: `0.0350200680`
- epoch `40`: `0.0044652102`
- epoch `80`: `0.0018246251`
- last 10 losses:
  `[0.0019204204, 0.0019154798, 0.0018541703, 0.0018962803, 0.0018821500, 0.0017914787, 0.0018225850, 0.0018390119, 0.0017754036, 0.0018246251]`

Observed final eval metrics at `80` epochs:

- `err_nRMSE=0.0375677`
- `err_RMSE=0.895959`
- `relative_l2=0.0375677`
- `fRMSE_low=2.10180`
- `fRMSE_mid=0.170457`
- `fRMSE_high=0.215245`
- `runtime_sec=5364.49`

## Convergence Audit

Convergence audit outputs:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/convergence_audit.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/convergence_audit.csv`

Fixed convergence rule:

- `late_window_mean_prev = mean(losses[60:70])`
- `late_window_mean_final = mean(losses[70:80])`
- `late_window_ratio = late_window_mean_final / late_window_mean_prev`
- `last5_delta = losses[79] - losses[74]`
- the row remains materially improving if `late_window_ratio < 0.95` or `last5_delta <= -0.001`

Observed audit values:

- `late_window_mean_prev=0.002342`
- `late_window_mean_final=0.001852`
- `late_window_ratio=0.790954`
- `last5_delta=-0.000058`
- `still_materially_improving=true`

Interpretation under the fixed rule:

- the late-window ratio is well below `0.95`, so the `80`-epoch row still counts as materially improving at the stop point
- the final few epochs flattened relative to the stronger late-window improvement signal, but the fixed audit rule still classifies this capped stop as not fully converged

## Shell-Validated `40ep -> 80ep` Delta

Delta outputs:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/shared_blocks10_1024cap_40ep_vs_80ep.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/shared_blocks10_1024cap_40ep_vs_80ep.csv`

Reference `40`-epoch metrics from the frozen shared-blocks10 row:

- `err_nRMSE=0.0445733`
- `err_RMSE=1.06304`
- `relative_l2=0.0445733`
- `fRMSE_low=2.48274`
- `fRMSE_mid=0.216790`
- `fRMSE_high=0.293987`
- `runtime_sec=2703.72`

Fresh `80`-epoch minus frozen `40`-epoch deltas:

- `err_nRMSE=-0.00700557`
- `err_RMSE=-0.167077`
- `relative_l2=-0.00700557`
- `fRMSE_low=-0.380939`
- `fRMSE_mid=-0.0463333`
- `fRMSE_high=-0.0787420`
- `runtime_sec=+2660.77`

This same-profile mixed-budget compare improved every tracked eval metric while keeping the shell contract identical and changing only the epoch budget.

## Bounded Interpretation

- The earlier `40`-epoch `1024cap` shared-blocks10 row was materially under-converged.
- Extending the same row to `80` epochs improved every tracked eval metric against its own frozen `40`-epoch reference, with especially visible gains in aggregate error and all three Fourier bands.
- Relative to the frozen `1024cap`, `40`-epoch shared-base row from the prerequisite architecture-ablation tranche, the fresh `80`-epoch shared-blocks10 row is now lower on every reported metric:
  - shared base `40`-epoch reference:
    `err_nRMSE=0.0435010`, `err_RMSE=1.03746`, `fRMSE_low=2.41774`, `fRMSE_mid=0.222417`, `fRMSE_high=0.298296`
  - shared-blocks10 `80`-epoch rerun:
    `err_nRMSE=0.0375677`, `err_RMSE=0.895959`, `fRMSE_low=2.10180`, `fRMSE_mid=0.170457`, `fRMSE_high=0.215245`
- That changes the earlier bounded `1024cap` read: the prior aggregate deficit for shared-blocks10 at `1024cap` was not stable once the row received a longer budget.
- This item still does not settle the same-budget architecture ranking:
  - `spectral_resnet_bottleneck_base` was not rerun at `80` epochs
  - the fresh shared-blocks10 row still satisfies the fixed material-improvement rule at stop time
  - the completed `2048cap` follow-up still shows the shared base row stronger on the same `40`-epoch budget beyond `1024cap`
- The correct bounded takeaway is:
  - the frozen `40`-epoch shared-blocks10 result understated the profile materially on the `1024cap` slice
  - deeper shared spectral depth remains a live manual follow-up rather than a closed loser
  - this tranche alone is still not enough to promote shared-blocks10 into a default or benchmark-complete claim because the evidence is capped and mixed-budget

## Verification

Commands run from `/home/ollie/Documents/PtychoPINN`:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
python - <<'PY'
from pathlib import Path
import json

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence")
required = [
    artifact_root / "convergence_audit.json",
    artifact_root / "convergence_audit.csv",
    artifact_root / "shared_blocks10_1024cap_40ep_vs_80ep.json",
    artifact_root / "shared_blocks10_1024cap_40ep_vs_80ep.csv",
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_shared_blocks10_1024cap_longer_convergence_summary.md"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing required outputs: {missing}")
state = json.loads((artifact_root / "shared_blocks10_1024cap_40ep_vs_80ep.json").read_text())
assert state["allowed_contract_delta"]["delta_kind"] == "epochs_only"
print("shared-blocks10 longer-convergence outputs present")
PY
```

Observed results:

- `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py` -> `79 passed in 51.91s`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py` -> exit `0`
- payload validation passed:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/verification/payload_validation.log`
- tracked launcher exit code:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-shared-blocks10-1024cap-longer-convergence/cns-shared-blocks10-1024cap-80ep-20260429T025641Z.launch/exit_code.txt = 0`

## Claim Boundary

- this is capped CNS decision-support evidence only
- it does not satisfy the PDEBench full-training benchmark gate
- it does not justify promoting `spectral_resnet_bottleneck_shared_blocks10` into a default profile bundle
- it does not resolve the same-budget `80`-epoch architecture ranking because `spectral_resnet_bottleneck_base` was not rerun at `80` epochs
- it does establish that the frozen `40`-epoch shared-blocks10 `1024cap` row materially understated that profile on the capped CNS contract
