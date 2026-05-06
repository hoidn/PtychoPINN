# BRDT Supervised+Born 40-Epoch Paper Evidence Summary

> **Owning backlog item:** `2026-05-05-brdt-supervised-born-40ep-paper-evidence`
> **Final claim boundary:** `decision_support_convergence_followup`
> **Promotion status:** `failed`
> **Lane status:** `decision_support`
> **Caveat:** decision-support context only. The bundle's training metrics,
> history records, model state, and sample-`255` visual bundle are valid
> same-contract decision-support evidence, but the paper-evidence gate failed
> on `git_provenance` and `host_provenance` after honest reconstruction of the
> overwritten `runtime_provenance.json` from `invocation.json` (see Section 7).
> This bundle does **not** authorize manuscript paper-evidence claims, does
> **not** replace the required CDI `lines128` or PDEBench CNS pillars, and does
> **not** authorize `/home/ollie/Documents/neurips/` publication.
> **Post-hoc architecture caveat (2026-05-06):** the `ffno` row in this
> historical bundle was generated before the BRDT FFNO adapter was corrected
> to remove post-bottleneck CNN refiners. Treat its metrics and visuals as a
> legacy FFNO-local-refiner proxy result, not as a pure FFNO-paper-stack
> result. Current code uses `SpatialLifter -> SharedFactorizedFfnoBottleneck
> -> 1x1` output projection and rejects `cnn_blocks`; a pure-BRDT-FFNO
> comparison requires a fresh rerun.

## 1. Identity And Locked Contract

- Initiative: `NEURIPS-HYBRID-RESNET-2026`.
- Lane: Born-Rytov diffraction tomography (BRDT) additive candidate study.
- Fresh immutable artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/`
- Read-only lineage inputs:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/`
- Locked same-contract fields inherited from the earlier BRDT bundle:
  - dataset id `brdt128_decision_support_preflight`
  - split `2048 / 256 / 256`
  - operator `BornRytovForward2D`, Born mode, `N=D=128`, `A=64`,
    `wavelength_px=8.0`, `medium_ri=1.333`, `normalize=unitary_fft`
  - input mode `born_init_image`
  - train-only physical-`q` normalization
  - loss weights `image=1.0`, `physics=0.1`, `relative_physics=0.1`,
    `tv=1e-5`, `positivity=1e-4`
  - fixed samples `[145, 83, 255, 126]` with sample `255` required for the
    paper-facing visual bundle
- Changed fields for this additive promotion attempt only:
  - rows rerun: `hybrid_resnet`, `ffno`
  - epochs `40`
  - optimizer Adam, `lr=2e-4`
  - scheduler `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
  - batch size `16`
  - seed `42`
  - per-epoch `history.json` / `history.csv`

## 2. Outcome

The fresh 40-epoch rerun improves both neural rows against their frozen
20-epoch authorities under the same dataset, operator, split, input, and loss
contract. The training metrics, history records, model state, and sample-`255`
visual bundle are valid same-contract decision-support evidence.

The paper-evidence gate did **not** pass. Section 7 documents why: an earlier
`--rebuild-meta-only` invocation under prior code overwrote the original
training-run `runtime_provenance.json` with the rebuild host's snapshot, and
the git SHA, git-dirty state, hostname, platform, and GPU count cannot be
honestly recovered from `invocation.json`. After honest reconstruction the
gate's `git_provenance` and `host_provenance` checks fail by design, demoting
the bundle to `decision_support_convergence_followup` rather than blessing
fabricated values.

Hybrid ResNet remains the strongest image-space BRDT row in the recorded
metrics. The historical FFNO-local-refiner proxy remains competitive at much
lower parameter count, but does not displace Hybrid ResNet on this capped
contract. Do not cite that row as a pure FFNO-paper-stack comparison without a
fresh no-refiner rerun.

## 3. 20-Epoch To 40-Epoch Delta Read

| Row | image_relative_l2_phys | meas_relative_l2 | PSNR_phys | SSIM_phys | Final LR | LR drops | Improving at stop |
|---|---:|---:|---:|---:|---:|---:|---|
| `hybrid_resnet` | `0.3190 -> 0.2875` (`-0.0315`) | `0.1992 -> 0.1739` (`-0.0253`) | `29.74 -> 30.64` (`+0.90`) | `0.9471 -> 0.9556` (`+0.0085`) | `2e-4` | `0` | `true` |
| `ffno` | `0.3421 -> 0.3324` (`-0.0097`) | `0.1859 -> 0.1784` (`-0.0075`) | `29.13 -> 29.38` (`+0.25`) | `0.9420 -> 0.9455` (`+0.0035`) | `2e-4` | `0` | `true` |

Convergence read from `convergence_audit.json`:

- Both rerun rows reached the planned `40` history records and retained the
  planned scheduler configuration.
- Both rows improved on every tracked blocking metric versus the frozen
  20-epoch lineage.
- Neither row triggered a plateau LR reduction, and both remained materially
  improving over the last 5 and 10 epochs. This is a bounded additive-evidence
  result, not a claim that either row is fully converged.

## 4. Sample-255 Visual Provenance Decision

The paper-facing visual authority is the fresh sample-`255` bundle under:

- `visuals/sample_0255_compare_q.png`
- `visuals/sample_0255_error_q.png`
- `visuals/sample_0255_sinogram_residual.png`
- `figures/source_arrays/sample_0255_*`

The classical/model-based comparator for sample `255` is accepted from the
frozen `2026-04-29` baseline bundle's recorded same-contract row
`classical_born_backprop`, whose baseline `metrics.json` and source arrays mark
that row `row_status="completed"` with `paper_label="Model-based Born inverse"`.
This summary resolves the earlier stale narrative inconsistency in
`brdt_preflight_summary.md` in favor of the concrete baseline artifact payload
actually consumed by the fresh 40-epoch bundle.

## 5. Gate Result And Reasons

- Final claim boundary: `decision_support_convergence_followup`
- Promotion status: `failed`
- Lane status: `decision_support`
- Failed gate checks: `git_provenance`, `host_provenance`

Promotion did **not** pass. Most gate conditions were satisfied:

- both rerun rows completed successfully
- both rows emitted `40` history records
- scheduler fields matched the locked plan contract
- dataset identity, split manifest, run-log, and exit-code proof were present
- sample-`255` compare/error/source-array assets were present with the
  same-contract classical comparator
- Python and PyTorch identity (`python_provenance`, `torch_provenance`) were
  recovered from `invocation.json` and pass the gate
- checked-in evidence surfaces (durable summary, `paper_evidence_index.md`,
  `paper_evidence_manifest.json`, `docs/index.md`) reference this backlog
  item, artifact root, and the demoted claim boundary

But the gate honestly fails on:

- `git_provenance` — `git_sha`/`git_dirty` cannot be recovered from
  `invocation.json` and were lost when an earlier rebuild overwrote the
  original training-run `runtime_provenance.json` with the rebuild host's
  snapshot
- `host_provenance` — `hostname`/`gpu_count` were similarly lost

The paper-evidence package design's BRDT amendment has been removed; an
additive paper-evidence promotion of this bundle requires retraining on a
clean repo so the original runtime provenance is captured at training time.

This decision-support outcome remains deliberately narrow:

- BRDT remains decision-support context only, not a manuscript pillar
- the lane does not replace CDI `lines128` or PDEBench CNS
- no full-training BRDT competitiveness claim is authorized

## 6. Residual Risks

- The rows were still improving at stop and never reduced LR, so the correct
  interpretation is bounded decision-support evidence rather than fully
  converged BRDT performance.
- The classical comparator used for the sample-`255` figure is inherited by
  lineage from the baseline bundle rather than regenerated in this pass.
- The original training-run `runtime_provenance.json` was overwritten by an
  earlier rebuild path before the runner's preservation guard was added. The
  reconstructed payload restores only what `invocation.json` preserved
  (tracked PID, launch timestamp, Python/PyTorch identity); the git SHA,
  git-dirty state, hostname, platform, and GPU count are recorded as `null`
  with an explicit `provenance_reconstruction` block listing the recovered
  vs unrecoverable fields and the rationale. This loss is the proximate
  reason promotion fails. Going forward the runner preserves whatever is on
  disk exactly during rebuild, so a future training pass on a clean repo
  would record full provenance and could be promoted under the same gate.
- This item is repo-local only; `/home/ollie/Documents/neurips/` remains out of
  scope.

## 7. Reproducibility And Meta Provenance

> **Important:** the bundle's on-disk `runtime_provenance.json` was reconstructed
> from `invocation.json` after an earlier `--rebuild-meta-only` invocation under
> prior code overwrote the original training-run payload with the rebuild
> host's snapshot. invocation.json was written by the original training process
> at startup and is the only preserved authoritative record of the original
> training-run launch identity. The reconstruction restores `tracked_pid`,
> `pid`, `launch_timestamp_utc`, `python_executable`, `python_version`, and the
> `torch.*` block exactly as the original training process recorded them; it
> sets `git_sha`, `git_dirty`, `hostname`, `platform`, and `gpu_count` to
> `null` because invocation.json never preserved them and they cannot be
> honestly recovered. A `provenance_reconstruction` block at the top of
> `runtime_provenance.json` records the source path, the reconstruction
> timestamp, the recovered-vs-unrecoverable field lists, and the rationale.
> The gate's `git_provenance` and `host_provenance` checks fail on the
> reconstructed payload by design, demoting the bundle to
> `decision_support_convergence_followup` rather than blessing fabricated
> values. An additive paper-evidence promotion of this bundle requires
> retraining on a clean repo so full provenance is captured at training time.

The fresh artifact root is reproducible end-to-end from the approved runner
in `scripts/studies/born_rytov_dt/run_brdt_40ep_paper_evidence.py`:

- The full live training run is invoked with the locked CLI listed in the
  execution plan and is guarded by a per-output-root writer lock that refuses
  duplicate writers and refuses to silently overwrite a populated bundle.
- The top-level `preflight_manifest.json` claim boundary and promotion status
  are re-seeded from the recomputed `paper_evidence_gate.json` after training
  completes, so the manifest cannot present a passing additive label without
  the gate honestly passing.
- The gate's `provenance_checks` payload validates that runtime provenance
  carries `git_sha`/`git_dirty`, host identity and GPU count, the recorded
  `python_executable`/`python_version` and `torch.version`/`cuda_version` /
  `cuda_available` fields, model profiles and the run log are present,
  dataset identity and split manifest exist, the sample-`255` visual bundle
  (including the same-contract classical comparator) is materialized, and
  the durable summary at this path is checked-in and references this
  backlog item. The Python and PyTorch identity fields are validated as
  separate `python_provenance` and `torch_provenance` checks so a bundle
  whose `runtime_provenance.json` lost those fields now fails the gate.
- Meta artifacts (manifest, runtime provenance, run-exit status, convergence
  audit, gate, visuals) can be deterministically re-derived from existing
  per-row `history.{json,csv}`, `row_summary.json`, `model_profile.json`, and
  source arrays via the runner's `--rebuild-meta-only` mode without
  retraining. The meta-rebuild path acquires the same per-output-root writer
  lock as the live training path so it cannot race with an active training
  run. The rebuild path no longer re-samples Python/PyTorch/CUDA/git/host
  fields from the rebuild process: it preserves every recorded original
  `runtime_provenance.json` field exactly (`tracked_pid`,
  `launch_timestamp_utc`, `git_sha`, `git_dirty`, `hostname`, `platform`,
  `gpu_count`, `python_executable`, `python_version`, `torch.*`, etc.) and
  attaches a separate `meta_rebuild` block recording the rebuild process's
  pid, timestamp, git SHA, host, GPU count, and argv. If
  `runtime_provenance.json` is missing, unparseable, or lacks any required
  original field, the rebuild raises rather than fabricating provenance —
  bundles whose original training-run provenance has been lost must be
  retrained, not silently regenerated.
- For bundles whose original `runtime_provenance.json` was lost to an earlier
  rebuild-host overwrite, the runner provides an honest recovery path via
  `--reconstruct-runtime-provenance-from-invocation`. This mode reads the
  preserved `invocation.json` and rewrites `runtime_provenance.json` with
  only the fields invocation.json captured (`tracked_pid`/`pid`,
  `launch_timestamp_utc`, `python_executable`, `python_version`, `torch.*`),
  leaves git/host fields explicitly `null`, and records a
  `provenance_reconstruction` block. The amend path recognizes this block
  and only requires the reconstruction's declared `recovered_fields` to be
  non-null; the gate's `git_provenance` and `host_provenance` checks then
  fail on the reconstructed payload, naturally demoting the bundle rather
  than fabricating provenance. This is exactly the path applied to the
  bundle described by this summary.
- The gate's `same_contract_lineage` check actively re-validates the baseline
  and FFNO-extension bundles each time the gate is recomputed and verifies
  that this bundle's `preflight_manifest.json` `baseline_lineage` block points
  at the actual lineage roots, that the dataset id matches across all three
  manifests, and that the current bundle's split counts, fixed-sample roster,
  operator geometry, dataset normalization, per-row input mode, and
  training-contract fields (batch size, learning rate, optimizer, seed, and
  loss-weight schedule) match the frozen baseline lineage. Drift on any of
  these locked invariants now fails the gate.
- The gate's `evidence_surfaces_prepared` check requires the durable summary
  at this path, `paper_evidence_index.md`, `paper_evidence_manifest.json`,
  and the repo-wide `docs/index.md` to all reference this backlog item, and
  additionally enforces cross-surface internal consistency: each surface must
  contain the canonical artifact root path
  (`.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence`)
  and must contain the same authoritative claim-boundary string read from the
  structured `paper_evidence_manifest.json` row registry entry for this
  backlog item. When the manifest's authoritative entry advertises the
  promoted boundary `paper_evidence_brdt_additive`, the check additionally
  requires `paper_evidence_package_design.md` to reference this backlog
  item, the canonical artifact root, and the promoted boundary; the plan
  ties promotion to the presence of a checked-in evidence amendment
  consistent with the gate result, so a passed manifest with no design-doc
  amendment now fails the gate. Drift between surfaces (e.g. the manifest
  advertising one boundary while the durable summary still advertises
  another) also fails the gate.
- `metrics.json`, `combined_metrics.json`, and `metric_schema.json` are
  re-seeded with the gate's final `claim_boundary` after the gate runs, so
  the bundle's machine-consumed metric tables can never advertise the
  pre-gate label after a successful promotion. Both the live training path
  and the `--rebuild-meta-only` path apply this reseed.
- The `--rebuild-meta-only` path now refuses to fabricate completion
  evidence: if `run_exit_status.json` is missing, unparseable, or lacks
  `tracked_pid`/`exit_code`/`status`, the rebuild raises rather than
  defaulting to `exit_code=0`/`status="completed"`. A bundle whose original
  exit-status proof has been lost must be retrained, not silently
  regenerated.
- The gate's `scheduler_matches_contract` check now verifies the per-row
  recorded scheduler block against every plan-bound field
  (`reduce_on_plateau` plus `factor=0.5`, `patience=2`, `threshold=0.0`,
  `min_lr=1e-5`) and the surrounding optimizer recipe (`epochs`,
  `batch_size`, `learning_rate`). A bundle with drifted plateau settings or a
  drifted optimizer recipe now fails the gate even when the scheduler name
  matches.
- The gate's `sample_255_visual_bundle` check requires every per-sample,
  per-row source-array file (`q_target`, `sino_obs`, classical comparator,
  Hybrid ResNet, and FFNO `q_pred`/`sino_pred`) to exist for the configured
  `required_paper_sample`, instead of only checking that the classical
  panel was present and the figure list was non-empty.
- The gate's `exit_code_proof` check now requires `run_exit_status.json` to
  agree with `runtime_provenance.json` on `tracked_pid` AND to record
  `exit_code == 0` AND `status == "completed"`. The live training path now
  defers writing `run_exit_status.json` until all materialization (training,
  evaluation, metrics, visuals) has succeeded, and overwrites it with a
  failed status if the subsequent gate or manifest re-seed steps raise. The
  combined effect is that a partial run can no longer leave a stale
  `"completed"`/`0` exit-status artifact behind.
