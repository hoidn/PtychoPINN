# BRDT Sinogram-Input 40-Epoch Paper Evidence Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to execute this plan task-by-task. Do not create worktrees. Keep long-running commands under implementation ownership until they finish or a recoverable failure path is documented. For the live BRDT run, use tmux plus the `ptycho311` conda environment, track the exact launched PID, do not start a second writer against the same `--output-root`, and treat the run as complete only when that PID exits `0` and the required output artifacts are freshly written.

**Goal:** Produce the current BRDT sinogram-input candidate-evidence bundle by running the `ffno` and `sru_net` rows for `40` epochs, keeping the Born inverse as a non-learned reference only, and promoting repo-local manuscript surfaces to the new root only if the final evidence gate passes.

**Architecture:** Reuse the already-approved sinogram-input adapter contract and the existing BRDT paper-evidence packaging pattern rather than inventing a new one-off lane. The work splits into four units: harden the `run_sinogram_input_40ep` bundle so it emits the durable evidence surfaces expected by downstream tooling, execute the live run and read the resulting promotion gate, promote manuscript-facing BRDT assets from the new root only on a passing gate while otherwise preserving the current manuscript authority, then update discoverability and audit surfaces so the older Born-image-input bundles remain lineage only and cannot source sinogram-input claims.

**Tech Stack:** PATH `python`, PyTorch, task-local BRDT runners under `scripts/studies/born_rytov_dt/`, JSON/CSV/PNG/NumPy artifacts, LaTeX manuscript assets, `pytest`, `compileall`, tmux, `ptycho311`.

---

## Selected Objective

- Run BRDT `ffno` and `sru_net` learned rows on the fixed decision-support split `2048 / 256 / 256` for `40` epochs using measured complex sinograms as learned-model input.
- Keep the model-based Born inverse in the bundle only as a non-learned reference row and visualization source.
- Emit metrics, histories, model profiles, throughput, sample-`255` source arrays, and the paper-refresh artifacts under `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/`.
- Refresh the manuscript-visible BRDT table, Figure 3 context/source arrays, model-configuration table, efficiency table, manuscript PDF, and manuscript zip from that new root only if the new bundle passes the evidence gate.
- If the evidence gate does not pass after a narrow documented fix attempt, keep the current manuscript-visible BRDT authority on the existing `2026-05-06` root, but still publish the new candidate-lane summary, manifest entry, and audit result honestly.

## Scope Boundaries

- In scope: the sinogram-input `40`-epoch BRDT runner, its durable evidence outputs, repo-local BRDT paper refresh surfaces, and the discoverability/index updates needed to make this the current sinogram-input authority.
- In scope: narrow fixes to BRDT task-local scripts/tests when blocking checks reveal missing outputs, row-name drift, stale source-root constants, or gate-surface inconsistencies.
- Out of scope: new BRDT architectures, a broader BRDT hyperparameter sweep, changes to the locked dataset/operator/loss/split contract, or any claim that BRDT replaces the required CDI `lines128` or PDEBench CNS pillars.
- Out of scope: `/home/ollie/Documents/neurips/` publication work; this item only updates repo-local manuscript/package surfaces.

## Explicit Non-Goals

- Do not rerun or rewrite the completed `2026-05-05` or `2026-05-06` BRDT Born-image-input bundles as if they had used the sinogram-input contract.
- Do not silently alias the new bundle back to the historical contract. Old bundles remain discoverable lineage only.
- Do not expand this item into WaveBench, CNS, CDI, or broader manuscript-content work unrelated to BRDT source refresh.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.

## Steering And Roadmap Constraints

- BRDT remains an additive candidate lane only. Even if the run succeeds, the plan must preserve the design and steering boundary that the required NeurIPS pillars are CDI `lines128` and PDEBench CNS.
- Equal-footing comparison rules remain locked: use the existing BRDT decision-support dataset, operator authority, split counts, loss recipe, seed policy, and fixed-sample policy. Do not relax the contract to make the run easier.
- Preserve the roadmap phase routing already attached to this item: `candidate-brdt-sinogram-input`. Do not escalate this work into a later roadmap phase.
- Smoke or dry-run outputs are readiness evidence only. They are blocking pre-run checks, not performance evidence.
- Keep paper-local filenames stable where practical; update their contents and metadata to the new root rather than creating a second set of manuscript filenames unless a stable filename is impossible.

## Prerequisite Status

- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` shows no initiative-level blocked tranches. The completed mainline tranches are Phase 0, Phase 1, and selected Phase 2 readiness/execution items; nothing there blocks candidate-lane BRDT follow-up.
- The item-level prerequisite `2026-05-07-brdt-sinogram-input-adapter-contract` is already represented by `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_sinogram_input_adapter_contract.md` plus its smoke root. Treat that summary and smoke proof as the blocking prerequisite for this plan.
- If dry-run or smoke checks expose a missing artifact, import bug, or stale paper-refresh contract, diagnose/fix/rerun inside this item before declaring a blocker. Reserve `BLOCKED` only for missing hardware/resources, external dependencies outside current authority, or an unrecoverable failure after a documented narrow fix attempt.

## Implementation Architecture

- **Unit 1: Sinogram-input evidence runner.** `scripts/studies/born_rytov_dt/run_sinogram_input_40ep.py` is the owner of the new bundle contract. It must emit the run-local manifests, histories, metrics, runtime/gate payloads, and sample-`255` arrays needed by downstream tooling.
- **Unit 2: Gate-conditioned manuscript promotion.** Read `paper_evidence_gate.json` from the fresh run before changing manuscript authority. Only a passing gate permits `scripts/studies/paper_results_refresh.py`, `scripts/studies/paper_efficiency_table.py`, and `scripts/studies/paper_model_config_table.py` to repoint stable manuscript-facing BRDT assets to the new root.
- **Unit 3: Durable discoverability.** A new checked-in BRDT summary plus the evidence indexes/manifests must agree on one backlog item, one artifact root, and one claim interpretation. Historical `2026-05-05` and `2026-05-06` summaries stay discoverable but must be clearly marked as Born-image-input lineage for old-contract claims only.
- **Unit 4: Manifest-driven audit synchronization.** Any change to `paper_evidence_manifest.json` or audit assumptions must be followed by a rerun of `scripts/studies/paper_evidence_audit.py` so the checked-in audit summary and any packaged copy stay synchronized with the final BRDT status.

## File And Artifact Targets

**Mandatory contract outputs**

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/preflight_manifest.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/metric_schema.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/metrics.{json,csv}`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/combined_metrics.{json,csv}`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/combined_manifest.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/split_manifest.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/runtime_provenance.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/run_exit_status.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/paper_evidence_gate.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/convergence_audit.{json,csv}`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/rows/ffno/{history.json,history.csv,model_profile.json,row_summary.json}`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/rows/sru_net/{history.json,history.csv,model_profile.json,row_summary.json}`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/rows/classical_born_backprop/row_summary.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/figures/source_arrays/sample_0255_{q_target,sino_obs,classical_born_backprop_q_pred,ffno_q_pred,sru_net_q_pred}.npy`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/visuals/sample_0255_{compare_q,error_q,sinogram_residual}.png`

**Likely code and test surfaces**

- `scripts/studies/born_rytov_dt/run_sinogram_input_40ep.py`
- `scripts/studies/born_rytov_dt/run_sinogram_input_smoke.py`
- `scripts/studies/born_rytov_dt/run_config.py`
- `scripts/studies/born_rytov_dt/reporting.py`
- `scripts/studies/born_rytov_dt/convergence.py`
- `tests/studies/test_born_rytov_dt_preflight.py`
- `tests/studies/test_born_rytov_dt_adapters.py`

**Mandatory paper-local packaging surfaces when the new gate passes**

- `scripts/studies/paper_results_refresh.py`
- `scripts/studies/paper_efficiency_table.py`
- `scripts/studies/paper_model_config_table.py`
- `tests/studies/test_paper_results_refresh.py`
- `tests/studies/test_paper_efficiency_table.py`
- `tests/studies/test_paper_model_config_table.py`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.{tex,csv,json}`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_sample_0255_context_recon_error.png`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.{tex,csv,json}`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.{tex,csv,json}`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

**Mandatory audit/package synchronization surfaces when the manifest or audit outputs change**

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-paper-evidence-package-audit/verification/`

**Mandatory discoverability surfaces**

- Create `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_sinogram_input_40ep_paper_evidence_summary.md`
- Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
- Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json` if it is the current BRDT contract-family owner for this lane
- Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md` if and only if the new gate passes and the package design must name this additive BRDT authority explicitly
- Update `docs/index.md` if it needs to point at the new durable summary for repo-wide discoverability
- Add contract-supersession notes to the historical BRDT summaries that would otherwise be mistaken for the current sinogram-input authority

## Task 1: Harden The Sinogram-Input Bundle Contract Before The Live Run

**Files**

- Read/modify: `scripts/studies/born_rytov_dt/run_sinogram_input_40ep.py`
- Read/modify if needed: `scripts/studies/born_rytov_dt/run_sinogram_input_smoke.py`
- Test: `tests/studies/test_born_rytov_dt_preflight.py`
- Test conditionally: `tests/studies/test_born_rytov_dt_adapters.py`

- [ ] Run the required deterministic syntax/import gate before anything expensive: `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`.
- [ ] Run the required deterministic dry-run gate: `python -m scripts.studies.born_rytov_dt.run_sinogram_input_40ep --dry-run`. Verify `preflight_manifest.json` records `input_mode=sinogram`, `in_channels=2`, and the row roster contains `classical_born_backprop`, `ffno`, and `sru_net`.
- [ ] Run the required deterministic smoke gate: `python -m scripts.studies.born_rytov_dt.run_sinogram_input_smoke`. Verify the smoke summary and row-local adapter artifacts prove that both learned rows consume measured complex sinograms and that the Born inverse remains `non_learned_reference_only`.
- [ ] If the live runner still emits only the minimal dry-run/metrics payload, extend it to match the durable paper-evidence surfaces already expected by the BRDT refresh/index tooling: writer lock or equivalent single-writer protection, runtime provenance, run-exit status, split manifest, combined manifest, convergence audit, gate payload, row-local histories, and visual/source-array manifests.
- [ ] Keep the new learned SRU-Net row honest as `sru_net` in the artifact bundle. Update downstream readers to map `sru_net -> SRU-Net` instead of renaming the new bundle back to the historical `hybrid_resnet` row id.
- [ ] Reuse the historical BRDT paper-evidence/corrected-rerun packaging pattern where possible; do not invent a materially different gate or manifest schema unless the sinogram-input contract truly requires it.

**Verification**

- Blocking:
  - `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`
  - `python -m scripts.studies.born_rytov_dt.run_sinogram_input_40ep --dry-run`
  - `python -m scripts.studies.born_rytov_dt.run_sinogram_input_smoke`
  - If `run_sinogram_input_40ep.py` or the smoke path changes, run `pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "sinogram_input_40ep or sinogram_input_smoke"`.
- Supporting:
  - If adapter/model-input code changes, run `pytest -q tests/studies/test_born_rytov_dt_adapters.py -k "sinogram or input_mode"`.

## Task 2: Execute And Validate The 40-Epoch Sinogram-Input Bundle

**Files**

- Execute/update: `scripts/studies/born_rytov_dt/run_sinogram_input_40ep.py`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/`

- [ ] Before launch, confirm no active process is already writing to the selected output root. If there is one, wait for it or move to a fresh, explicitly authorized root; do not start a duplicate writer.
- [ ] Launch the long run in tmux under `ptycho311`, recording and waiting on the exact PID inside the pane. The command must remain the required runner entrypoint, not a substitute wrapper: `python -m scripts.studies.born_rytov_dt.run_sinogram_input_40ep`.
- [ ] Treat the run as complete only when the tracked PID exits `0` and the required top-level artifacts are freshly present.
- [ ] Verify both learned rows complete the full `40` epochs and that each row writes per-epoch `history.json` and `history.csv`.
- [ ] Verify the non-learned Born reference row remains present only as reference/context. The learned rows must be `ffno` and `sru_net`; no learned row may derive its input from a Born image.
- [ ] Verify `metrics` and `combined_metrics` include image-space error, measurement error, PSNR, SSIM, parameter count, evaluation throughput, and row status for the classical and neural rows.
- [ ] Verify the sample-`255` source arrays and visuals are complete for the new contract: target `q`, measured `s_obs`, Born inverse prediction, FFNO prediction, SRU-Net prediction, compare panel, error panel, and sinogram-residual panel.
- [ ] Write or update the convergence audit and gate payload so the final summary can honestly state whether the sinogram-input bundle is promoted as additive paper evidence or remains a non-promoted candidate-lane outcome.

**Verification**

- Blocking:
  - The tracked live-run PID exits `0`.
  - `preflight_manifest.json` still advertises `input_mode=sinogram` and `in_channels=2`.
  - Both learned rows show `40`-epoch histories.
  - The required source arrays for sample `255` exist under `figures/source_arrays/`.
  - `metrics.json` and `combined_metrics.json` contain image error, measurement error, PSNR, SSIM, parameter count, and throughput fields.
- Supporting:
  - Compare the new row metrics against the old Born-image-input lineage only as historical context; do not use the comparison itself as a launch gate.

## Task 3: Refresh Repo-Local BRDT Paper Surfaces From The New Root

**Files**

- Modify only if `paper_evidence_gate.json` promotes the new root: `scripts/studies/paper_results_refresh.py`
- Modify only if `paper_evidence_gate.json` promotes the new root: `scripts/studies/paper_efficiency_table.py`
- Modify only if `paper_evidence_gate.json` promotes the new root: `scripts/studies/paper_model_config_table.py`
- Test when any of those helpers change: `tests/studies/test_paper_results_refresh.py`
- Test when any of those helpers change: `tests/studies/test_paper_efficiency_table.py`
- Test when any of those helpers change: `tests/studies/test_paper_model_config_table.py`
- Modify/generated only on a passing gate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.{tex,csv,json}`
- Modify/generated only on a passing gate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_sample_0255_context_recon_error.png`
- Modify/generated only on a passing gate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.{tex,csv,json}`
- Modify/generated only on a passing gate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.{tex,csv,json}`
- Modify only on a passing gate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

- [ ] Read `paper_evidence_gate.json` immediately after Task 2 and branch explicitly before any manuscript-authority edit:
  - passing gate: promote the new root into the repo-local BRDT manuscript surfaces;
  - failed gate after a narrow documented fix attempt: keep the active manuscript authority on the current `2026-05-06-brdt-corrected-ffno-40ep-rerun` root and skip any repointing of BRDT figure/table/model-config/efficiency inputs to the failed-gate root.
- [ ] On the passing-gate branch only, repoint every BRDT paper-refresh constant away from the old `2026-05-06-brdt-corrected-ffno-40ep-rerun` root and onto the new `2026-05-07-brdt-sinogram-input-40ep-paper-evidence` root.
- [ ] On the passing-gate branch only, update the BRDT paper-refresh readers to consume the new learned-row ids and source-array filenames. `sru_net` should render as `SRU-Net`; `ffno` remains `FFNO`; `classical_born_backprop` remains the model-based Born reference.
- [ ] On the passing-gate branch only, keep the existing manuscript-facing filenames stable unless a stable name is impossible. The paper-local BRDT table may still live at `tables/brdt_decision_support_metrics.*`, but its metadata must now point to the new artifact root and the new claim interpretation.
- [ ] On the passing-gate branch only, regenerate the BRDT context figure from the new sample-`255` arrays so the bottom-left panel is the measured sinogram magnitude and the reconstruction/error panels come from the new sinogram-input predictions, not the old Born-image-input bundle.
- [ ] On the passing-gate branch only, refresh the model-configuration table so the BRDT section reads from the new root and records the new BRDT input/output contract honestly.
- [ ] On the passing-gate branch only, refresh the efficiency table so the BRDT rows compute throughput from the new root rather than from the old corrected-FFNO rerun.
- [ ] On the passing-gate branch only, update visible manuscript BRDT wording if it still says the learned model input is a Born-derived image. The manuscript should say the learned models consume the measured complex sinogram and that the Born inverse is a non-learned reference only.
- [ ] On the passing-gate branch only, rebuild the manuscript PDF with the repo-local draft after the BRDT assets are regenerated.
- [ ] On the failed-gate branch, leave the manuscript `.tex`, BRDT figure/table/model-config/efficiency assets, and active paper-refresh defaults on the existing authority. Record the preserved-old-root decision in the new durable summary and manifest instead of silently promoting the failed-gate root.

**Verification**

- Blocking:
  - If the passing-gate branch changed any paper refresh/model-config/efficiency helpers, run `pytest -q tests/studies/test_paper_results_refresh.py tests/studies/test_paper_efficiency_table.py tests/studies/test_paper_model_config_table.py`.
  - On the passing-gate branch, run `python scripts/studies/paper_results_refresh.py --write-brdt-assets --write-brdt-context-figure --write-model-config-table --write-efficiency-table`.
  - On the passing-gate branch, rebuild the manuscript PDF with the repo-local draft.
  - Confirm the branch outcome is explicit:
    - passing gate: the regenerated BRDT table/figure/model-config/efficiency assets all cite the new root;
    - failed gate: the existing manuscript-visible BRDT assets still cite the prior `2026-05-06` authority and no active manuscript surface was silently repointed.
- Supporting:
  - Scan active code/manuscript surfaces for stale root or old-contract wording:
    - `rg -n "2026-05-06-brdt-corrected-ffno-40ep-rerun|born_init_image|Born input" scripts/studies/paper_results_refresh.py scripts/studies/paper_efficiency_table.py scripts/studies/paper_model_config_table.py docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
  - On the passing-gate branch, confirm the regenerated assets and working-tree `.tex` agree on the new input contract.
  - On the failed-gate branch, confirm the summary/manifest explain why the old manuscript authority was preserved.

## Task 4: Publish The New Durable BRDT Authority And Supersession Notes

**Files**

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_sinogram_input_40ep_paper_evidence_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Modify conditionally: `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- Modify conditionally: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- Modify/generated when manifest or audit inputs change: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md`
- Modify conditionally: `docs/index.md`
- Modify conditionally if default audit inputs must change: `scripts/studies/paper_evidence_audit.py`
- Modify historical BRDT summaries that would otherwise look current for sinogram-input claims

- [ ] Write a new self-contained BRDT durable summary for this backlog item. It must record identity, contract, row roster, exact artifact root, metric read, sample-`255` visual provenance, gate result, residual risks, and an explicit reminder that BRDT remains additive candidate evidence only.
- [ ] Update `paper_evidence_manifest.json` regardless of promotion outcome. The entry must preserve the new backlog item id, artifact root, gate result, claim boundary, manuscript-authority status, and any blocked or promotion-failure reasons.
- [ ] Update `paper_evidence_index.md` and `evidence_matrix.md` so they point to the new durable summary and new artifact root with wording that matches the final gate result:
  - passing gate: mark the new root as the current additive manuscript authority;
  - failed gate: keep the new root discoverable as the current sinogram-input candidate-lane outcome while making clear that active manuscript authority stays on the prior `2026-05-06` root.
- [ ] Update `model_variant_index.json` and any BRDT contract-family registry that the evidence matrix currently uses so future paper-refresh or planning tasks can discover the new row lineage without guessing.
- [ ] Add contract-supersession notes to the older BRDT summaries that are still discoverable from the indexes, making clear that they remain valid for the historical Born-image-input contract only and must not source sinogram-input claims.
- [ ] Update `paper_evidence_package_design.md` only if the new gate passes and the package design needs an explicit amendment naming this BRDT bundle as the current additive manuscript authority.
- [ ] If `paper_evidence_manifest.json` or the default audit inputs change, rerun `python scripts/studies/paper_evidence_audit.py --repo-root .` before packaging closes. Preserve the updated `paper_evidence_package_audit_summary.md` and verification logs as the checked-in audit authority for this item.
- [ ] After the audit rerun, rebuild the repo-local manuscript zip if the packaged audit summary changed so the zip does not carry a stale evidence-audit view. On a failed-gate branch this zip rebuild must keep the old BRDT manuscript assets while carrying the current audit summary.
- [ ] If the new gate does not pass after a narrow documented fix attempt, keep the new durable summary, manifest, and discoverability surfaces honest about the failed-gate outcome and do not repoint manuscript-authority surfaces as if promotion succeeded.

**Verification**

- Blocking:
  - The new summary exists and names the new backlog item, artifact root, and contract.
  - `paper_evidence_index.md`, `paper_evidence_manifest.json`, and `evidence_matrix.md` all agree on the same backlog item, source root, and final manuscript-authority status.
  - Historical BRDT summary surfaces that remain linked from active indexes carry an explicit Born-image-input supersession note or are otherwise prevented from being mistaken for the current sinogram-input authority.
  - If `paper_evidence_manifest.json` or `scripts/studies/paper_evidence_audit.py` changed, run `python scripts/studies/paper_evidence_audit.py --repo-root .`.
  - If the audit summary changed, rebuild and validate the zip with `zip -T docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip`.
- Supporting:
  - If `docs/index.md` is updated, confirm it points to the new durable summary authority.
  - Validate updated machine-readable surfaces with `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json >/tmp/paper_evidence_manifest.json`.

## Required Deterministic Checks

These are mandatory for this item. They are not optional substitutes.

- Blocking before or during implementation:
  - `python -m scripts.studies.born_rytov_dt.run_sinogram_input_40ep --dry-run`
  - `python -m scripts.studies.born_rytov_dt.run_sinogram_input_smoke`
  - `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`

## Completion Criteria

- The live BRDT bundle at `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/` is complete, internally consistent, and uses `input_mode=sinogram` with `in_channels=2`.
- Both learned rows complete `40` epochs, the Born inverse stays non-learned reference only, and the bundle includes the required metrics, histories, model profiles, sample-`255` arrays, and throughput fields.
- If the evidence gate passes, repo-local BRDT manuscript surfaces no longer point to the old `2026-05-06` root for active BRDT claims, the manuscript-visible BRDT figure/table/model-config/efficiency surfaces are regenerated from the new root, and the manuscript package is rebuilt successfully.
- If the evidence gate fails after a narrow documented fix attempt, the repo-local BRDT manuscript authority remains on the prior `2026-05-06` root, while the new durable summary, manifest, and indexes explicitly record the failed-gate sinogram-input outcome without claiming manuscript promotion.
- A new durable BRDT summary exists, discoverability surfaces point to it with the correct gate-conditioned status, historical Born-image-input bundles remain discoverable but explicitly lineage-only for old-contract claims, and the packaged evidence-audit summary matches the final manifest state.
