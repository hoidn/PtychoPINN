# PDEBench CNS History Length 3+ Compare Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-pdebench-cns-history-len3plus-compare`
- Date: `2026-04-29`
- Status: implementation complete; frozen `history_len=2` anchors, mandatory `history_len=3` inspect plus `10`/`40`-epoch pilots, cross-history sidecars, and the `history_len=4` gate decision are all recorded
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-pdebench-cns-history-len3plus-compare/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/`

This summary records capped decision-support evidence only. It does not create a
benchmark-complete CNS ranking, a rollout/autoregressive result, or a
paper-facing artifact under `/home/ollie/Documents/neurips/`.

## Fixed Compare Contract

The fresh longer-context rows and frozen references kept the capped local CNS
surface fixed everywhere except temporal context:

- task: `2d_cfd_cns`
- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- trajectories: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- batch size: `4`
- training loss: `mse`
- metric family:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`,
  `fRMSE_high`
- allowed contract deltas only:
  - frozen reference: `history_len=2`, `concat u[t-2:t] -> u[t]`,
    `input_channels=8`, raw `windows_per_trajectory=19`,
    raw `available_windows=190000`
  - mandatory fresh run: `history_len=3`, `concat u[t-3:t] -> u[t]`,
    `input_channels=12`, raw `windows_per_trajectory=18`,
    raw `available_windows=180000`
  - optional gated branch: `history_len=4`, `concat u[t-4:t] -> u[t]`,
    `input_channels=16`, raw `windows_per_trajectory=17`,
    raw `available_windows=170000`

Even as raw eligibility shrank from `19 -> 18` windows per trajectory for the
mandatory branch, the emitted capped split counts stayed fixed because
`max_windows_per_trajectory=8` remained unchanged:

- `train=4096`
- `val=512`
- `test=512`

The frozen manifest is:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history2_reference_runs.json`

The compare payloads enforce the rule:

- comparison standard: `Only history_len and its derived sample/input-channel contract may differ.`

Cross-run gallery rendering stayed non-fatal and required exact target
alignment under `np.allclose(..., atol=1e-6, rtol=1e-6)`.

## Recorded Artifacts

Frozen history-2 anchors:

- `10ep` manifest bucket:
  - spectral / hybrid:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
  - FNO / U-Net:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
- `40ep` manifest bucket:
  - spectral:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
  - backfilled `hybrid_resnet_cns`:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z`
  - FNO / U-Net:
    `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`

Inspect proofs:

- `history_len=3`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-inspect-20260429T000000Z`
- pre-gate `history_len=4` inspect only:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4-inspect-20260429T000000Z`

Fresh history-3 pilot runs:

- `10ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-10ep-20260429T071905Z`
- `40ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z`

Tracked run-completion proof:

- `10ep` exit code:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/launch-history3-pilot-10ep-20260429T071905Z/exit_code = 0`
- `40ep` exit code:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/launch-history3-pilot-40ep-20260429T073705Z/exit_code = 0`

Cross-history compare sidecars:

- `10ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/compare_10ep_history3_against_history2.json`
  and `.csv`
- `40ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/compare_40ep_history3_against_history2.json`
  and `.csv`

## Results

### `10` Epochs

Frozen history-2 ranking by `err_nRMSE`:

1. `spectral_resnet_bottleneck_base` - `0.0869938582`
2. `hybrid_resnet_cns` - `0.0944002941`
3. `fno_base` - `0.1063433066`
4. `unet_strong` - `0.6222500205`

Fresh history-3 ranking by `err_nRMSE`:

1. `fno_base` - `0.0953909084`
2. `hybrid_resnet_cns` - `0.1119609326`
3. `spectral_resnet_bottleneck_base` - `0.1407901347`
4. `unet_strong` - `0.7326587439`

Directional answer:

- `spectral_resnet_bottleneck_base` did not improve under `history_len=3` at
  `10` epochs; `err_nRMSE` worsened by `+0.0537962765`
  (`0.0869938582 -> 0.1407901347`) and `err_RMSE` worsened by
  `+1.2972486019`, even though `fRMSE_high` improved
  (`0.6955373287 -> 0.6477258205`)
- `hybrid_resnet_cns` also worsened on aggregate:
  `err_nRMSE +0.0175606385`, `err_RMSE +0.4221265316`
- `fno_base` improved and moved into first place on the capped `10`-epoch
  slice:
  `err_nRMSE -0.0109523982`, `err_RMSE -0.2666120529`
- `unet_strong` worsened materially on aggregate and on `fRMSE_high`

### `40` Epochs

Frozen history-2 ranking by `err_nRMSE`:

1. `spectral_resnet_bottleneck_base` - `0.0615620054`
2. `hybrid_resnet_cns` - `0.0644183308`
3. `fno_base` - `0.0740992129`
4. `unet_strong` - `0.6757976413`

Fresh history-3 ranking by `err_nRMSE`:

1. `spectral_resnet_bottleneck_base` - `0.0455205254`
2. `hybrid_resnet_cns` - `0.0538428985`
3. `fno_base` - `0.0567254014`
4. `unet_strong` - `0.6771671176`

Directional answer:

- `spectral_resnet_bottleneck_base` improved cleanly under `history_len=3` at
  `40` epochs:
  - `err_nRMSE 0.0615620054 -> 0.0455205254`
  - `err_RMSE 1.4877649546 -> 1.0991724730`
  - `fRMSE_high 0.4349334538 -> 0.3467437923`
- `hybrid_resnet_cns` also improved on all three tracked headline metrics:
  - `err_nRMSE 0.0644183308 -> 0.0538428985`
  - `err_RMSE 1.5567935705 -> 1.3001306057`
  - `fRMSE_high 0.3683068156 -> 0.3200356364`
- `fno_base` improved on all three tracked headline metrics:
  - `err_nRMSE 0.0740992129 -> 0.0567254014`
  - `err_RMSE 1.7907506227 -> 1.3697336912`
  - `fRMSE_high 0.6717720628 -> 0.6104770303`
- `unet_strong` stayed last and was effectively flat-to-worse on aggregate
  (`err_nRMSE +0.0013694763`, `err_RMSE +0.0194244385`) even though
  `fRMSE_high` improved (`1.3326253891 -> 1.1730086803`)

## `history_len=4` Gate

The plan required the optional `history_len=4` branch to stay closed unless the
fresh `40`-epoch `history_len=3` spectral row improved `err_nRMSE`,
`err_RMSE`, and did not worsen `fRMSE_high` against the frozen history-2
reference, with the `10`-epoch spectral direction recorded as supporting
context.

Gate record:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4_gate_decision.json`

Observed gate inputs:

- `10ep` spectral signal:
  - `improves_err_nRMSE=false`
  - `improves_err_RMSE=false`
  - `preserves_fRMSE_high=true`
- `40ep` spectral signal:
  - `improves_err_nRMSE=true`
  - `improves_err_RMSE=true`
  - `preserves_fRMSE_high=true`

Decision:

- gate status: `closed`
- reason: the `10`-epoch and `40`-epoch spectral signals disagree, and the plan
  forbids opening the gate without a written scientific reason in that case
- no `history_len=4` pilot or `compare_*_history4_against_history2.*` payloads
  were authorized

The pre-gate `history4-inspect-20260429T000000Z` proof remains in the artifact
root as an exploratory contract check only. It is not an authorized pilot
branch and must not be interpreted as the gate having opened.

## Interpretation

This backlog item answers the intended scientific question narrowly:

- on the fixed capped CNS contract, increasing temporal context from
  `history_len=2` to `history_len=3` did not help uniformly at short budget:
  the `10`-epoch slice reordered around a stronger `fno_base` row while both
  the spectral and canonical Hybrid rows got worse on aggregate error
- at `40` epochs, the same longer-context contract materially helped the three
  stronger rows (`spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`,
  `fno_base`) on both aggregate denormalized metrics and `fRMSE_high`, while
  leaving `unet_strong` essentially flat and still last
- because the spectral direction was not stable across the two approved epoch
  budgets, the plan correctly left the optional `history_len=4` branch closed

The bounded takeaway is therefore:

- `history_len=3` looks promising on this capped slice once the stronger rows
  are trained long enough
- the effect is not yet stable enough across budgets to justify an automatic
  move to even longer context
- this remains summary-local capped evidence, not a reusable repo-wide rule or
  a benchmark-complete CNS recommendation

## Verification

Commands run from `/home/ollie/Documents/PtychoPINN`:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
python - <<'PY'
from pathlib import Path
import json

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare")
required = [
    artifact_root / "compare_10ep_history3_against_history2.json",
    artifact_root / "compare_10ep_history3_against_history2.csv",
    artifact_root / "compare_40ep_history3_against_history2.json",
    artifact_root / "compare_40ep_history3_against_history2.csv",
    artifact_root / "history4_gate_decision.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing required outputs: {missing}")
gate = json.loads((artifact_root / "history4_gate_decision.json").read_text())
assert gate["gate_status"] == "closed", gate
print("history3 compare outputs and closed history4 gate validated")
PY
```

Observed results:

- `pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py`
  - see `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/final_pytest.log`
  - result: `45 passed in 51.67s`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
  - see `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/final_compileall.log`
  - result: exit `0`
- compare-sidecar generation:
  - see `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/history3_compare_generation.log`
  - result: `wrote history3 compare sidecars`
- gate decision generation:
  - see `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/history4_gate_decision.log`
  - result: `closed`
- payload validation:
  - see `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/final_output_validation.log`
  - result: `final output validation passed`
- tracked launcher exit codes:
  - `launch-history3-pilot-10ep-20260429T071905Z/exit_code = 0`
  - `launch-history3-pilot-40ep-20260429T073705Z/exit_code = 0`

## Claim Boundary

- this is capped CNS decision-support evidence only
- it does not satisfy the PDEBench full-training benchmark gate
- it does not justify opening the `history_len=4` branch under the approved
  decision rule
- it does not establish a general recommendation for the full PDEBench suite or
  for non-capped CNS runs
