# Hybrid ResNet Skip Connections + Mode Search - Stage B Execution Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute Stage B (`fno_blocks` structural axis) using the Stage-A robust champion anchor and the same promotion/robustness policy used throughout this initiative.

**Tech Stack:** PyTorch/Lightning (`ptycho_torch`), existing grid-lines + NERSC study scripts, runbook-style orchestration, JSON/CSV/Markdown artifacts.

**Companion Design:** `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md`.

## Scope

This split document owns Task 11 only (Stage-B execution commands and artifacts).

## Stage Plan Links

- Upstream Plan: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-stage-a-execution.md`
- Downstream Plan: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-stage-c-execution.md`

## Shared Contracts

- Use promotion and robustness rules from `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search-design.md` Section 6.
- Hub document: `docs/plans/2026-02-21-hybrid-resnet-skip-mode-search.md`.
- Stage ordering precondition: run Task 11 only after Stage-A robust outputs exist (`promotion/summary_seed_robust.csv`) and a single-row Stage-A champion anchor summary has been materialized.
- Stage-B transition-anchor source (single upstream artifact): producer is Stage-A `N=128` `promotion/champion_anchor_summary.csv` (single-row robust champion selected from Stage-A `promotion/summary_seed_robust.csv`), consumer field is `--promotion-source-summary`.
- Stage-B transition-anchor fail-closed rule: if the Stage-A champion anchor source is missing, has zero rows, or has more than one row, stop Stage-B execution and report the missing/ambiguous source; do not substitute `promotion/summary_seed_robust.csv`, `promotion/stage_anchor_summary.csv`, or `promotion/default_baselines.csv`.
- Stage-B downstream handoff contract: Stage-B `N=128` seed-rerank collation MUST emit one-row `promotion/champion_anchor_summary.csv`; this is the only canonical Stage-C transition anchor artifact.
- Epoch floor: all Stage-B runs MUST use at least `10` epochs (`--epochs-n128 >= 10`, `--epochs-n256 >= 10`) unless an approved exception is recorded.
- Non-canonical rule: outputs generated below the epoch floor MUST NOT be used as promotion sources.
- Between consecutive Stage-B run invocations, delete repo-root `memoized_data/` before launching the next command (`rm -rf memoized_data/`).
- Semantic guardrail validation gate (mandatory): after every Stage-B `N=128` summary generation and before any seed-rerank collation or N=256 promotion run, recompute and verify `phase_ssim_drop_vs_baseline` from baseline provenance and fail closed if any row mismatches persisted values or exceeds `--max-phase-ssim-drop`.
- Fail-closed invalidation contract: when semantic guardrail validation fails, mark current Stage-B package invalid and stop promotion. Recovery must rerun from the earliest affected scope in order: `N=128 summary -> seed-rerank runs -> robust collation -> champion anchor -> N=256 summary/promotion`.
- Canonical evidence policy for review reruns: do not pass `--reuse-existing-run-metrics` for canonical Stage-B reruns. Diagnostic reuse is allowed only when guardrail values are recomputed in-run and baseline provenance is persisted in the emitted summary.
- Per-profile baseline discoverability rule: Stage-B artifacts MUST include `promotion/default_baselines.csv` and `promotion/default_baselines.md` with exactly one true-default baseline row per active `(N, dataset_profile)` combination.
- Baseline-lane separation rule: `promotion/default_baselines.csv|.md` remains baseline/default evidence only and MUST NOT be used as Stage-B transition-anchor source.
- N=256 dual-profile rule: canonical `N=256` evaluation/promotion runs MUST include both `cameraman256_halfsplit_v1` and `custom_npz_pair_n256`.

---

### Task 11: Stage B Search (Axis 1: `fno_blocks`)

**Files:**
- Modify: none (execution + artifacts)

**Step 0: Resolve single Stage-A champion-anchor summary for Stage-B N=128**

Use the one-row champion-anchor artifact selected from Stage-A robust summary:
- `<stage_a_n128_root>/promotion/champion_anchor_summary.csv`

Selection rule for this one-row file: rank-1 robust feasible candidate from `summary_seed_robust.csv`, breaking ties deterministically by (1) higher `amp_ssim`, (2) lower `train_wall_time_sec`, (3) lower `model_params`.
Keep the true-default Stage-A control anchor only in baseline artifacts (`promotion/default_baselines.csv|.md` and/or control-anchor diagnostics), not as Stage-B canonical source.

**Step 1: Run Stage B at N=128 on Stage-A champion anchor config**

Before each Task 11 run command block (Step 1 and Step 2), run:
```bash
rm -rf memoized_data/
```

Run:
```bash
STAGE_A_CHAMPION_ANCHOR="<stage_a_n128_root>/promotion/champion_anchor_summary.csv"
STAGE_B_N128_ROOT="<stage_b_n128_root>"

python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id B \
  --promotion-source-summary "${STAGE_A_CHAMPION_ANCHOR}" \
  --fno-blocks-values 3,4 \
  --ns 128 \
  --dataset-profiles-n128 integration_grid_lines_n128_v1 \
  --epochs-n128 20 \
  --top-k-n256 0 \
  --promotion-objectives amp_ssim,train_wall_time_sec \
  --max-train-seconds-n128 2700 \
  --max-inference-seconds-n128 60 \
  --max-phase-ssim-drop 0.03 \
  --max-model-params 300000000 \
  --output-root "${STAGE_B_N128_ROOT}" \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```
Expected: `summary.md` includes `stage_id=B`, `substage_id=none`, and `fno_blocks` column.
Expected: `summary.csv` guardrail fields are semantically computed from provenance (no placeholder values): `phase_ssim_drop_vs_baseline`, `max_phase_ssim_drop`, `phase_guardrail_pass`, and baseline provenance columns.
Non-axis knobs (`modes`, `skip`, `width`) come from `--promotion-source-summary`, so canonical Stage-B runs inherit the Stage-A champion context.

**Step 1b: Validate semantic phase guardrail before promotion**

Run:
```bash
python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --validate-phase-guardrail \
  --summary-csv "${STAGE_B_N128_ROOT}/summary.csv" \
  --baseline-summary "${STAGE_A_CHAMPION_ANCHOR}" \
  --max-phase-ssim-drop 0.03 \
  --write-validation-report "${STAGE_B_N128_ROOT}/promotion/phase_guardrail_validation.json"
```

Expected:
- Exit code `0` only when recomputed drop values match persisted fields and all feasible rows satisfy `drop <= max_phase_ssim_drop`.
- Validation report is written under `promotion/` and includes per-row provenance/resolved baseline evidence.
- If this check fails: stop Task 11 promotion flow, mark Stage-B artifacts invalid, and rerun from Step 1 after applying the implementation fix.

**Step 1c: Emit N=128 baseline discoverability artifacts (mandatory)**

Run:
```bash
python - <<'PY'
import csv
from pathlib import Path

root = Path("${STAGE_B_N128_ROOT}")
summary_path = root / "summary.csv"
rows = list(csv.DictReader(summary_path.open(newline="")))
if not rows:
    raise SystemExit(f"summary.csv is empty: {summary_path}")

profiles = sorted({str(row.get("dataset_profile", "")).strip() for row in rows})
selected_rows = []
for profile in profiles:
    matches = [
        row for row in rows
        if str(row.get("dataset_profile", "")).strip() == profile
        and str(row.get("skip", "off")).strip().lower() == "off"
        and str(row.get("fno_blocks", "")).strip() == "4"
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one true-default baseline row for N=128 profile={profile}, got {len(matches)}"
        )
    row = dict(matches[0])
    row["true_default_baseline"] = "true"
    selected_rows.append(row)

promotion_dir = root / "promotion"
promotion_dir.mkdir(parents=True, exist_ok=True)
csv_path = promotion_dir / "default_baselines.csv"
md_path = promotion_dir / "default_baselines.md"

fieldnames = list(selected_rows[0].keys())
with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in selected_rows:
        writer.writerow(row)

lines = [
    "# Stage-B N=128 Default Baselines",
    "",
    "| dataset_profile | run_id | fno_blocks | skip | true_default_baseline |",
    "| --- | --- | --- | --- | --- |",
]
for row in selected_rows:
    lines.append(
        f"| {row['dataset_profile']} | {row['run_id']} | {row['fno_blocks']} | "
        f"{row['skip']} | {row['true_default_baseline']} |"
    )
md_path.write_text("\n".join(lines) + "\n")
print(csv_path)
print(md_path)
PY
```

Expected:
- `${STAGE_B_N128_ROOT}/promotion/default_baselines.csv` exists and contains exactly one `true_default_baseline=true` row for each active `N=128` profile.
- `${STAGE_B_N128_ROOT}/promotion/default_baselines.md` exists and mirrors the same run IDs/profile coverage.

**Step 2: Promote feasible Pareto-ranked top-K and run N=256**

Before this step, run the boundary seed-rerank policy on `${STAGE_B_N128_ROOT}/summary.csv`:
- rerank `top-K + next 2` at `N=128` with seeds `11` and `17`,
- produce `${STAGE_B_N128_ROOT}/promotion/summary_seed_robust.csv`,
- use that robustness summary as promotion source.
- emit `${STAGE_B_N128_ROOT}/promotion/champion_anchor_summary.csv` as the single-row Stage-B champion anchor for downstream Stage-C consumption via `--promotion-source-summary`.
- consume Step 1c baseline artifacts as mandatory pre-promotion evidence (`promotion/default_baselines.csv|.md`).
- enforce Step 1b semantic guardrail validation gate before launching rerank or downstream promotion.

Run:
```bash
STAGE_B_N128_ROOT="<stage_b_n128_root>"
STAGE_B_ROBUST_SUMMARY="${STAGE_B_N128_ROOT}/promotion/summary_seed_robust.csv"
STAGE_B_N256_ROOT="<stage_b_n256_root>"
CAMERAMAN_DP="<path/to/cameraman256_dp.hdf5>"
CAMERAMAN_PARA="<path/to/cameraman256_para.hdf5>"

python scripts/studies/runbooks/run_hybrid_resnet_mode_skip_sweep.py \
  --stage-id B \
  --promotion-source-summary "${STAGE_B_ROBUST_SUMMARY}" \
  --ns 256 \
  --dataset-profiles-n256 cameraman256_halfsplit_v1,custom_npz_pair_n256 \
  --epochs-n256 40 \
  --top-k-n256 6 \
  --promotion-objectives amp_ssim,train_wall_time_sec \
  --max-train-seconds-n256 9000 \
  --max-inference-seconds-n256 240 \
  --max-phase-ssim-drop 0.03 \
  --max-model-params 300000000 \
  --cameraman-dp "${CAMERAMAN_DP}" \
  --cameraman-para "${CAMERAMAN_PARA}" \
  --custom-n256-train-npz <path/to/lines_n256_train.npz> \
  --custom-n256-test-npz <path/to/lines_n256_test.npz> \
  --output-root "${STAGE_B_N256_ROOT}" \
  --seed 3 \
  --no-probe-mask \
  --no-torch-mae-pred-l2-match-target
```

After this run completes, emit `${STAGE_B_N256_ROOT}/promotion/default_baselines.csv` and `.md` with exactly one true-default baseline row per active `N=256` profile.
For canonical evidence reruns, do not pass `--reuse-existing-run-metrics` in Step 1/Step 2 commands.

**Step 2b: Emit N=256 baseline discoverability artifacts (mandatory)**

Run:
```bash
python - <<'PY'
import csv
from pathlib import Path

root = Path("${STAGE_B_N256_ROOT}")
summary_path = root / "summary.csv"
rows = list(csv.DictReader(summary_path.open(newline="")))
if not rows:
    raise SystemExit(f"summary.csv is empty: {summary_path}")

profiles = sorted({str(row.get("dataset_profile", "")).strip() for row in rows})
selected_rows = []
for profile in profiles:
    matches = [
        row for row in rows
        if str(row.get("dataset_profile", "")).strip() == profile
        and str(row.get("skip", "off")).strip().lower() == "off"
        and str(row.get("fno_blocks", "")).strip() == "4"
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one true-default baseline row for N=256 profile={profile}, got {len(matches)}"
        )
    row = dict(matches[0])
    row["true_default_baseline"] = "true"
    selected_rows.append(row)

promotion_dir = root / "promotion"
promotion_dir.mkdir(parents=True, exist_ok=True)
csv_path = promotion_dir / "default_baselines.csv"
md_path = promotion_dir / "default_baselines.md"

fieldnames = list(selected_rows[0].keys())
with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in selected_rows:
        writer.writerow(row)

lines = [
    "# Stage-B N=256 Default Baselines",
    "",
    "| dataset_profile | run_id | fno_blocks | skip | true_default_baseline |",
    "| --- | --- | --- | --- | --- |",
]
for row in selected_rows:
    lines.append(
        f"| {row['dataset_profile']} | {row['run_id']} | {row['fno_blocks']} | "
        f"{row['skip']} | {row['true_default_baseline']} |"
    )
md_path.write_text("\n".join(lines) + "\n")
print(csv_path)
print(md_path)
PY
```

Expected:
- `${STAGE_B_N256_ROOT}/promotion/default_baselines.csv` exists and contains exactly one `true_default_baseline=true` row for each active `N=256` profile.
- `${STAGE_B_N256_ROOT}/promotion/default_baselines.md` exists and mirrors the same run IDs/profile coverage.

**Step 2c: Fail-closed baseline evidence contract checks (mandatory)**

Run:
```bash
TASK11_CONTRACT_START_EPOCH="$(date +%s)"
python - <<'PY'
import csv
import os
from pathlib import Path

start_epoch = int(os.environ["TASK11_CONTRACT_START_EPOCH"])
roots = [
    (128, Path("${STAGE_B_N128_ROOT}")),
    (256, Path("${STAGE_B_N256_ROOT}")),
]

for n_value, root in roots:
    summary_rows = list(csv.DictReader((root / "summary.csv").open(newline="")))
    if not summary_rows:
        raise SystemExit(f"N={n_value} summary.csv is empty: {root / 'summary.csv'}")
    expected_profiles = sorted({str(row.get("dataset_profile", "")).strip() for row in summary_rows})

    for rel in ("promotion/default_baselines.csv", "promotion/default_baselines.md"):
        path = root / rel
        if not path.exists():
            raise SystemExit(f"N={n_value} missing required baseline artifact: {path}")
        if int(path.stat().st_mtime) < start_epoch:
            raise SystemExit(f"N={n_value} stale baseline artifact (mtime < check start): {path}")

    baseline_rows = list(csv.DictReader((root / "promotion" / "default_baselines.csv").open(newline="")))
    if not baseline_rows:
        raise SystemExit(f"N={n_value} default_baselines.csv is empty")
    observed_profiles = sorted({str(row.get('dataset_profile', '')).strip() for row in baseline_rows})
    if observed_profiles != expected_profiles:
        raise SystemExit(
            f"N={n_value} baseline profile coverage mismatch: expected={expected_profiles} observed={observed_profiles}"
        )
    for profile in expected_profiles:
        matches = [
            row for row in baseline_rows
            if str(row.get("dataset_profile", "")).strip() == profile
            and str(row.get("true_default_baseline", "")).strip().lower() == "true"
        ]
        if len(matches) != 1:
            raise SystemExit(
                f"N={n_value} expected exactly one true_default_baseline row for profile={profile}, got {len(matches)}"
            )

print("baseline_contract_checks_passed")
PY
```

Expected:
- Contract checks fail closed when any baseline artifact is missing, stale, malformed, or profile-incomplete.
- Recovery path on failure: regenerate baseline artifacts for the failing scope (`N=128` or `N=256`), then re-run this contract check. If regeneration cannot preserve provenance, rerun Task 11 from the earliest affected scope (`Step 1` for N=128 failures; `Step 2` for N=256 failures).

**Step 3: Commit**

No commit (execution-only stage).

---
