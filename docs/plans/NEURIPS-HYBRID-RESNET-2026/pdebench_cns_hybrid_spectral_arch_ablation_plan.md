# PDEBench CNS Hybrid-Spectral Architectural Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a controlled PDEBench `2d_cfd_cns` architectural ablation for the hybrid-spectral family only, with the shell fixed and externally auditable, so the repo can tell whether shared-vs-non-shared spectral weights and spectral bottleneck depth materially change capped CNS results.

**Architecture:** Treat the CNS model as two separable parts: a **shell** (encoder/decoder, skip path, upsampler) and a **spectral bottleneck** (shared/non-shared weights, depth). This plan freezes the shell to the current canonical CNS choice and varies only the spectral bottleneck internals. CDI/ptycho is explicitly out of scope and must get its own separate plan because the data contract, losses, and artifact modes differ too much for a joint architecture study to be attributable.

**Tech Stack:** PyTorch, local PDEBench CNS runner, JSON/PNG artifact manifests, pytest, compileall

---

## Design For External Audit

This section is normative. The study is only correct if its artifacts prove these invariants without requiring a reader to trust repo folklore.

### Study Question

For the current PDEBench `2d_cfd_cns` contract, once the shell is locked, which spectral bottleneck choices matter:

- `spectral_bottleneck_share_weights`: `true` vs `false`
- `spectral_bottleneck_blocks`: `6`, `8`, `10`

The study is not trying to answer:

- whether local Hybrid beats spectral Hybrid
- whether `history_len=1` Markov training is better
- whether physics regularization helps
- whether larger spectral mode counts help
- whether CDI/ptycho wants the same shell or bottleneck choices

### Fixed Contract

Every row in this ablation must share the same local CNS contract:

- task: `2d_cfd_cns`
- dataset:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- resolution: `128x128`
- history contract: `history_len=2`, `concat u[t-2:t] -> u[t]`
- capped split:
  - train trajectories: `512`
  - val trajectories: `64`
  - test trajectories: `64`
- `max_windows_per_trajectory=8`
- batch size: `4`
- training loss: `mse`
- evaluation metrics:
  - `err_RMSE`
  - `err_nRMSE`
  - `relative_l2`
  - `fRMSE_low`
  - `fRMSE_mid`
  - `fRMSE_high`
- reporting surfaces:
  - `comparison_summary.json`
  - `metrics_<profile>.json`
  - `model_profile_<profile>.json`
  - sample prediction/residual PNGs
  - train-split eval + held-out eval

### Fixed Shell Invariants

The shell is not an ablation axis in this study. It must be pinned explicitly in profile config, not inherited implicitly.

Every spectral row in this study must record:

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

If any spectral row omits `hybrid_upsampler` and falls back to the model-builder default, the run is not externally auditable and should be treated as invalid for this study.

### Ablation Matrix

The primary 10-epoch ablation matrix is:

- `spectral_resnet_bottleneck_base` (`shared`, `blocks=6`) - current shared anchor
- `spectral_resnet_bottleneck_shared_blocks8` (`shared`, `blocks=8`)
- `spectral_resnet_bottleneck_shared_blocks10` (`shared`, `blocks=10`)
- `spectral_resnet_bottleneck_noshare` (`noshare`, `blocks=6`) - current non-shared anchor
- `spectral_resnet_bottleneck_noshare_blocks8` (`noshare`, `blocks=8`)
- `spectral_resnet_bottleneck_noshare_blocks10` (`noshare`, `blocks=10`)

The 40-epoch follow-up is bounded and pre-registered:

- always rerun the two anchor rows:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_noshare`
- for the `shared` family, consider only the non-anchor candidates:
  - `spectral_resnet_bottleneck_shared_blocks8`
  - `spectral_resnet_bottleneck_shared_blocks10`
- for the `noshare` family, consider only the non-anchor candidates:
  - `spectral_resnet_bottleneck_noshare_blocks8`
  - `spectral_resnet_bottleneck_noshare_blocks10`
- add at most one extra `shared` row and at most one extra `noshare` row, chosen by:
  1. lower `relative_l2`
  2. tie-breaker lower `err_nRMSE`
  3. if still tied, lower `fRMSE_high`
- if no non-anchor candidate in a family beats the anchor under that ordering, keep the anchor only and do not add a second row from that family

This keeps the follow-up attributable and bounded and prevents duplicate profile lists. Do not widen to a full 6-row 40-epoch rerun unless a later plan explicitly approves it.

### Comparison Anchors

This plan does not rerun the broader local baselines unless the contract drifts, but anchor reuse is allowed only under a strict provenance rule.

For any quantitative comparison table that includes `hybrid_resnet_cns`, `fno_base`, or `unet_strong`, the final summary must record the exact reused artifact roots and prove that each reused row matches the fixed contract for:

- dataset file
- `history_len`
- trajectory caps
- `max_windows_per_trajectory`
- training loss
- batch size
- epoch budget

Additionally, any reused `hybrid_resnet_cns` anchor must prove the same shell contract:

- `hybrid_skip_connections=True`
- `hybrid_skip_style="add"`
- `hybrid_upsampler="pixelshuffle"`

If any reused anchor fails one of those checks, that anchor may still be mentioned as contextual prose, but it must not appear in the same quantitative comparison table. In that case, rerun the anchor under the current contract or omit it from direct numerical comparison.

Known reusable roots today:

- `10`-epoch `hybrid_resnet_cns` shell-locked anchor:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/cns-upsampler-cns-shell-10ep-20260421T221400Z`
- `10`-epoch `fno_base` and `unet_strong` anchors:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
- `40`-epoch `fno_base` and `unet_strong` anchors:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`

There is no pre-existing `40`-epoch `hybrid_resnet_cns` anchor recorded here under the shell-locked contract. If a `40`-epoch numerical comparison against `hybrid_resnet_cns` is needed, rerun it under a fresh root.

### External Audit Checklist

The final study summary must let an outside reader verify:

1. every row used the same dataset file and split family
2. every row used `history_len=2`
3. every row used `mse`
4. every row pinned `skip=add`
5. every row pinned `upsampler=pixelshuffle`
6. only spectral sharing and spectral depth changed across the ablation rows
7. train/test metrics and artifact paths are present for every completed row
8. any reused non-ablation anchor lists its exact artifact root and contract-match proof

If any item cannot be proven from emitted artifacts, the study is incomplete.

## File Structure And Responsibilities

- Modify: `scripts/studies/pdebench_image128/run_config.py`
  - Add explicit shell-locked spectral ablation profiles and make shell fields explicit on existing spectral rows.
- Modify: `tests/studies/test_pdebench_image128_models.py`
  - Add profile-shape tests that prove all study rows pin skip-add + pixelshuffle and vary only approved bottleneck fields.
- Modify: `tests/studies/test_pdebench_image128_runner.py`
  - Add runner-level checks that emitted model-profile artifacts expose the expected shell fields for spectral rows.
- Optional create: `scripts/studies/pdebench_image128/audit_cns_hybrid_spectral_ablation.py`
  - Small audit helper that reads emitted artifacts and writes a compact `audit_manifest.json`.
  - Only add this if the existing artifact set is too awkward for an external audit.
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
  - Durable study summary after execution.
- Modify: `docs/findings.md`
  - Record any promoted finding once results exist.
- Modify: `docs/index.md`
  - Keep the plan and summary discoverable.

## Task 1: Codify The Auditable Shell Lock

**Files:**
- Modify: `scripts/studies/pdebench_image128/run_config.py`
- Test: `tests/studies/test_pdebench_image128_models.py`

- [ ] **Step 1: Add failing profile-contract tests for the spectral study rows**

Add tests that assert, for every spectral row in this study:

- `base_model == "spectral_resnet_bottleneck_net"`
- `hybrid_skip_connections is True`
- `hybrid_skip_style == "add"`
- `hybrid_upsampler == "pixelshuffle"`
- only `spectral_bottleneck_share_weights` and `spectral_bottleneck_blocks` vary across the matrix

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py -k "spectral and shell"
```

Expected: fail because existing spectral rows do not yet pin `hybrid_upsampler="pixelshuffle"` and the new study profiles do not exist.

- [ ] **Step 3: Add explicit shell-locked spectral profiles**

In `run_config.py`:

- make `spectral_resnet_bottleneck_base` explicitly set `hybrid_upsampler="pixelshuffle"`
- make `spectral_resnet_bottleneck_noshare` explicitly set `hybrid_upsampler="pixelshuffle"`
- add:
  - `spectral_resnet_bottleneck_shared_blocks8`
  - `spectral_resnet_bottleneck_shared_blocks10`
  - `spectral_resnet_bottleneck_noshare_blocks8`
  - `spectral_resnet_bottleneck_noshare_blocks10`

Keep all non-approved shell and bottleneck-mode fields identical to the fixed contract.

- [ ] **Step 4: Re-run the focused tests and confirm pass**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py -k "spectral and shell"
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/studies/pdebench_image128/run_config.py tests/studies/test_pdebench_image128_models.py
git commit -m "test+feat: lock spectral CNS shell for hybrid-spectral ablation"
```

## Task 2: Prove The Runner Emits Auditable Shell Metadata

**Files:**
- Modify: `tests/studies/test_pdebench_image128_runner.py`
- Optional modify: `scripts/studies/pdebench_image128/cfd_cns.py`

- [ ] **Step 1: Add a failing runner test for emitted model-profile fields**

Write a test that runs a tiny CNS readiness pass with one spectral study row and asserts the emitted `model_profile_<profile>.json` contains:

- `hybrid_skip_connections=true`
- `hybrid_skip_style="add"`
- `hybrid_upsampler="pixelshuffle"`
- the expected `spectral_bottleneck_share_weights`
- the expected `spectral_bottleneck_blocks`

- [ ] **Step 2: Run the focused runner test and confirm failure if emission is incomplete**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py -k "spectral and model_profile"
```

Expected: fail only if the emitted profile artifact does not expose the shell/bottleneck fields cleanly.

- [ ] **Step 3: Patch emission only if needed**

If the test fails, patch the smallest runner/model-profile path necessary so the study rows expose their full shell lock in `model_profile_*.json`. Do not change training behavior here.

- [ ] **Step 4: Re-run the focused runner test**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py -k "spectral and model_profile"
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tests/studies/test_pdebench_image128_runner.py scripts/studies/pdebench_image128/cfd_cns.py
git commit -m "test+feat: emit auditable spectral shell metadata"
```

Use the second path in `git add` only if `cfd_cns.py` changed.

## Task 3: Run The Primary 10-Epoch Ablation Matrix

**Files:**
- No code changes unless blocked
- Artifacts under: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-arch-ablation/`

- [ ] **Step 1: Preflight the contract**

Verify before launching:

- the CNS file exists
- no process is already writing to the chosen output root
- the profile matrix resolves

Recommended commands:

```bash
python -m pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

- [ ] **Step 2: Create a fresh timestamped output root and launch the 10-epoch matrix under tmux**

Use a fresh timestamped root, for example:

```bash
RUN_ROOT=".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-arch-ablation/cap512-10ep-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode readiness \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root "$RUN_ROOT" \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_shared_blocks8,spectral_resnet_bottleneck_shared_blocks10,spectral_resnet_bottleneck_noshare,spectral_resnet_bottleneck_noshare_blocks8,spectral_resnet_bottleneck_noshare_blocks10 \
  --history-len 2 \
  --epochs 10 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

Track the exact PID and require exit code `0`. Do not reuse a prior `cap512-10ep*` root, and do not launch if another process is already writing to the chosen root.

- [ ] **Step 3: Verify required artifacts**

Require at minimum:

- `comparison_summary.json`
- `comparison_summary.csv`
- one `metrics_*.json` per profile
- one `model_profile_*.json` per profile
- sample PNG/NPZ outputs per profile

- [ ] **Step 4: Write a compact 10-epoch ranking note**

Record:

- shared depth winner
- non-shared depth winner
- how each compares to the current `blocks=6` anchors

- [ ] **Step 5: Commit only docs/state if any repo files changed**

Do not commit `.artifacts/`.

## Task 4: Run The Bounded 40-Epoch Follow-Up

**Files:**
- No code changes unless blocked
- Artifacts under the same study root

- [ ] **Step 1: Select the 40-epoch rows using the pre-registered rule**

Always include:

- `spectral_resnet_bottleneck_base`
- `spectral_resnet_bottleneck_noshare`

Then, per family, consider only non-anchor candidates:

- `shared` non-anchor candidates:
  - `spectral_resnet_bottleneck_shared_blocks8`
  - `spectral_resnet_bottleneck_shared_blocks10`
- `noshare` non-anchor candidates:
  - `spectral_resnet_bottleneck_noshare_blocks8`
  - `spectral_resnet_bottleneck_noshare_blocks10`

Add the best non-anchor candidate from a family only if it beats the anchor under the pre-registered ordering. Otherwise keep the anchor only. Do not substitute rows by visual preference or parameter count.

- [ ] **Step 2: Create a fresh timestamped output root and launch the 40-epoch rerun under tmux**

Use a fresh timestamped root, for example:

```bash
RUN_ROOT=".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-arch-ablation/cap512-40ep-$(date -u +%Y%m%dT%H%M%SZ)"
```

Keep the same split and training contract, changing only `--epochs 40` and the selected profile list. Do not reuse a prior `cap512-40ep*` root, and do not launch if another process is already writing to the chosen root.

- [ ] **Step 3: Verify train/test emission and ranking**

For each row, capture:

- train-split `relative_l2`
- held-out `relative_l2`
- gap magnitude
- `fRMSE_high`

This is required to audit whether deeper rows merely optimize more slowly or actually generalize better.

- [ ] **Step 4: Render comparison galleries**

Generate:

- prediction gallery PNG
- residual gallery PNG

Copy user-facing PNGs into `tmp/` for quick inspection.

- [ ] **Step 5: Commit only if repo docs/tests changed**

Do not commit `.artifacts/` or `tmp/`.

## Task 5: Larger-Cap Confirmation

**Files:**
- No code changes unless blocked

- [ ] **Step 1: Select finalists**

Pick at most two finalists:

- best shared-family row from Task 4
- best non-shared-family row from Task 4

- [ ] **Step 2: Run `1024 / 128 / 128` confirmation**

Keep:

- `history_len=2`
- `epochs=40`
- `batch_size=4`
- `max_windows_per_trajectory=8`

Change only the trajectory caps to:

- train `1024`
- val `128`
- test `128`

Use a fresh timestamped root, for example:

```bash
RUN_ROOT=".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-arch-ablation/cap1024-40ep-$(date -u +%Y%m%dT%H%M%SZ)"
```

- [ ] **Step 3: Compare train/test gap compression**

This step is mandatory because the larger-cap confirmation is what makes the final interpretation externally defensible. Record whether:

- aggregate error improves
- train/test gap shrinks
- the family ranking changes

- [ ] **Step 4: Commit only if repo docs/tests changed**

Do not commit `.artifacts/`.

## Task 6: Publish The Auditable Summary

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
- Modify: `docs/findings.md`
- Modify: `docs/index.md`

- [ ] **Step 1: Write the summary with an explicit audit section**

The summary must contain:

- fixed contract
- fixed shell invariants
- ablation matrix
- exact profile IDs
- output roots
- exact artifact roots for any reused non-ablation anchors, plus contract-match proof
- train/test metrics
- gallery paths
- larger-cap confirmation
- final claim boundary

- [ ] **Step 2: Add or update a finding only if results justify it**

Possible outcomes:

- shared still best
- non-shared still best
- depth `8` or `10` becomes the best bounded CNS spectral row

Do not pre-write the conclusion.

- [ ] **Step 3: Index the summary and plan**

Update `docs/index.md` so both the plan and the final summary are discoverable.

- [ ] **Step 4: Run final verification**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md docs/findings.md docs/index.md
git commit -m "docs: publish CNS hybrid-spectral ablation results"
```

## Deferred Follow-On: CDI/Ptycho Counterpart

Do not extend this plan in place.

If the repo wants a corresponding CDI/ptycho hybrid-spectral ablation, write a separate plan with its own:

- dataset contract
- loss contract
- shell boundary
- visual artifact policy
- correctness/audit criteria

The CNS shell lock from this plan must not be assumed valid for CDI.
