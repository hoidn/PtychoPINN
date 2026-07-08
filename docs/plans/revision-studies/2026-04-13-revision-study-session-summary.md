# Revision Study Session Summary - 2026-04-13

This is a narrative handoff for the April 13, 2026 revision-study work. It is intentionally stored under `docs/plans/revision-studies/` because it summarizes session state, workflow decisions, and artifact locations. It is not a durable recurring-bug entry for `docs/findings.md`.

## Scope

Primary work centered on two related threads:

- Revision-study workflow authoring and review-loop behavior across the local workflow repo and the PtychoPINN paper-revision runs.
- Probe mischaracterization stress-test paper assets, including perturbation-curve presentation and probe context panels.

The active project checkout for most of the work was:

- PtychoPINN repo: `/home/ollie/Documents/PtychoPINN`
- Paper repo: `/home/ollie/Documents/ptychopinnpaper2`
- Orchestration repo: `/home/ollie/Documents/agent-orchestration`

## Workflow Authoring And Review-Loop Work

The session started with review of whether workflows and prompts were organized well enough to find and reuse. The main conclusion was that the repo had useful design-plan-implement pieces, but discoverability and reuse boundaries were weak.

Concrete workflow-authoring conclusions:

- Workflow prompts should not leak runtime status labels into task language. For example, the design prompt should ask for a revision-study design document or execution-ready study design, not an "approved" design before the review loop has approved it.
- Prompt instructions belong in prompt files. `depends_on.inject` should be used for runtime-resolved file lists or content, not for substantive task instructions.
- Injected docs should be treated as candidate context unless a step truly requires every listed file.
- Provider-review steps should avoid broad documentation globs. Prefer `docs/index.md` plus a narrow set of exact likely-relevant docs.
- Ambient instruction files such as `AGENTS.md` or `CLAUDE.md` should not be listed as workflow dependencies just to force reading; the agent runtime already handles them.
- If semantic enforcement matters, put the standard in the review prompt and back it with an output contract or gate instead of duplicating instruction text in both YAML and prompt prose.
- Design, design-review, and planning prompts that may affect architecture, data contracts, workflow APIs, or stable modules should explicitly instruct agents to read `docs/index.md` first when present, then use it to select relevant specs, architecture docs, workflow guides, and findings docs.

The design-review prompt was tightened after a lax approval. The issue was that the review prompt did not require enough repository-specific spec and architecture grounding before deciding whether a design introduced debt, API drift, or contract conflicts. The fix was to make review judgment depend on `docs/index.md` and relevant docs selected from that index, while keeping the injected context narrow.

The plan-review prompt was similarly revised after the plan phase approved too quickly. The key point was not to make plan review duplicate design review wholesale, but to require it to verify that the plan preserves the design's repo-grounded constraints and makes implementation-ready decisions where the design delegates details.

The workflow guide was also updated to capture session lessons as general workflow-authoring guidance rather than as revision-study folklore.

## Reusable Workflow Conclusions

The session compared the monolithic revision-study workflow to existing reusable design-plan-implement style workflows. The best existing model identified was the style used by `workflows/examples/design_plan_impl_review_stack_v2_call.yaml`, which imports reusable phase workflows such as:

- `workflows/library/tracked_design_phase.yaml`
- `workflows/library/tracked_plan_phase.yaml`
- `workflows/library/design_plan_impl_implementation_phase.yaml`

The reason the revision-study workflow initially reinvented more structure than ideal was not that reusable phases were inherently wrong. The root cause was the lack of a small, canonical phase interface for this specific design-plan-implement family. Without that interface, using the reusable pieces would have required too much adapter work or implicit contract guessing.

The agreed adjustment was to avoid overengineering. The concrete fix was to document the existing reusable subworkflows and their expected artifact surfaces clearly enough that future workflows can call them instead of redrafting design, plan, and implementation loops from scratch. This should be a convention and guide improvement, not a large new framework.

## Revision-Study Template Work

The design template in PtychoPINN was reviewed:

- Existing file: `/home/ollie/Documents/PtychoPINN/plans/templates/design_template.md`

The template was considered useful but heavy. The recommended improvements were:

- Make optional sections explicitly skippable for small changes so the template does not encourage bloated designs.
- Add guidance under consumed inputs and authority to read `docs/index.md` first, then select relevant specs, architecture docs, workflow guides, and `docs/findings.md`.
- Add a short workflow/artifact contract section for workflow-affecting designs: key artifacts, producers, consumers, and validation or gate rules.
- Keep lifecycle metadata such as `status: approved`, but avoid having prompts call a draft "approved" before a review loop has approved it.

This template is best kept in the PtychoPINN repo when it is specific to PtychoPINN design decisions and paper-revision work. A generic design template for the orchestration repo should live in the orchestration repo only if it is intentionally repo-agnostic and maintained as a workflow-authoring asset.

## Workflow Runs

### First Non-ML Single-Shot CDI Workflow

A non-ML single-shot CDI workflow was already running during the session.

Observed state at one point:

- Run id: `20260412T234347Z-5jz8pm`
- Status: running until the user later requested it be killed
- Nested current step: `ImplementationReviewLoop`
- Implementation loop iteration: 4
- Completed implementation iterations: `[0, 1, 2, 3]`
- Max implementation iterations: 40
- Latest implementation-review decision: `REVISE`
- Design phase completed after iterations: `[0, 1, 2, 3, 4]`
- Plan phase completed after iteration: `[0]`

The workflow did not resume in the design-revision step after later failures because the run had already passed the design gate. Resuming a run preserves completed gated stages unless the workflow state is explicitly rewound or a fresh run is started.

The plan phase had only one iteration because the plan-review gate approved immediately. That was not necessarily wrong if the plan faithfully implemented a repo-grounded design, but it was suspicious in context because the review prompt was too permissive before later tightening.

The user eventually requested that this workflow be killed.

### Probe Mischaracterization Workflow

A second workflow instance was requested for:

- Seed design: `/home/ollie/Documents/PtychoPINN/revision_designs/probe_mischaracterization_stress_test.md`

Before launch, the seed design was edited to prefer the PyTorch backend all else equal and to note possible VRAM utilization issues from sequential TensorFlow model instantiation.

The probe workflow investigated metrics for probe perturbation conditions, including amplitude blur, phase noise, and phase curvature. It also surfaced an HIO/ER baseline issue.

### Quantitative OOD Workflow

The user requested a workflow instance for the quantitative out-of-distribution revision design.

Because the monolith workflow input type expected a repo-local relative path, the external paper-repo design seed and reviewer checklist were copied into PtychoPINN run state:

- Seed copy: `/home/ollie/Documents/PtychoPINN/state/revision-study-fig5-ood-metrics-20260413T071529Z/revision_design_seed.md`
- Checklist copy: `/home/ollie/Documents/PtychoPINN/state/revision-study-fig5-ood-metrics-20260413T071529Z/reviewer_revision_checklist.md`

The workflow launched from the PtychoPINN repo so the provider agents would pick up the more relevant PtychoPINN `AGENTS.md` and `CLAUDE.md` guidance.

Launch details:

- Tmux session: `revision-fig5-ood`
- Tmux socket: `/tmp/claude-tmux-sockets/claude.sock`
- Orchestrator PID: `301457`
- Workflow run id: `20260413T071708Z-qnji3i`
- State root: `/home/ollie/Documents/PtychoPINN/state/revision-study-fig5-ood-metrics-20260413T071529Z`
- Output dir: `/home/ollie/Documents/PtychoPINN/artifacts/revision_studies/fig5_ood_metrics_low_frequency_phase_20260413T071529Z`

Useful tmux commands:

```bash
tmux -S /tmp/claude-tmux-sockets/claude.sock attach -t revision-fig5-ood
tmux -S /tmp/claude-tmux-sockets/claude.sock capture-pane -p -J -t revision-fig5-ood:0.0 -S -200
```

The provider command included the intended Codex bypass flag:

```bash
codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check --model gpt-5.4 --config reasoning_effort=high
```

## Probe Mischaracterization Stress-Test Work

The probe mischaracterization stress-test figure was changed so each perturbation curve includes its own zero-perturbation anchor point instead of relying only on a dashed horizontal baseline.

The important x-axis convention is family-specific perturbation magnitude:

- Amplitude blur: x is `amplitude_blur_sigma_px`
- Phase noise: x is `phase_noise_sigma_rad`
- Phase curvature: x is `1.0 - phase_curvature_scale`

This means phase curvature values below and above `1.0` are signed around the no-change point. A subtle bug was fixed where phase curvature used `abs(1.0 - value)`, which collapsed opposite directions to the same x value.

Relevant code file:

- `/home/ollie/Documents/PtychoPINN/scripts/studies/probe_mischaracterization_stress_test.py`

Relevant test file:

- `/home/ollie/Documents/PtychoPINN/tests/studies/test_probe_mischaracterization_stress_test.py`

Added or updated test coverage included:

- `test_stress_figure_series_anchor_each_family_at_baseline`
- `test_load_probe_figure_context_uses_preferred_corrupted_probe`
- `test_load_probe_figure_context_falls_back_to_available_nonbaseline_probe`
- `test_format_probe_condition_label_handles_default_visual_conditions`
- `test_write_stress_figure_with_probe_context_writes_composite_png`
- `test_stress_figure_visual_context_metadata_records_panel_policy`

The stress figure was regenerated from the existing complete run rather than by rerunning the stress study:

- Existing run root: `/home/ollie/Documents/PtychoPINN/.artifacts/revision_studies/probe_mischaracterization/full_process_isolated_20260413T020739Z`
- Metrics file: `/home/ollie/Documents/PtychoPINN/.artifacts/revision_studies/probe_mischaracterization/full_process_isolated_20260413T020739Z/metrics.json`

Representative metrics from that run:

| Condition | amp_ssim | amp_psnr |
| --- | ---: | ---: |
| baseline | 0.9240195866132853 | 69.8425440265778 |
| amplitude_blur_sigma_px_0p5 | 0.9010006112536939 | 68.69665949639601 |
| amplitude_blur_sigma_px_1p0 | 0.8644152414298151 | 67.18495973087498 |
| amplitude_blur_sigma_px_2p0 | 0.785237518773647 | 65.00888176793053 |
| phase_curvature_scale_0p25 | 0.03794652970425857 | 48.910437270866495 |
| phase_curvature_scale_0p50 | 0.06802227824908473 | 47.80580838262735 |
| phase_curvature_scale_0p75 | 0.5526736227879963 | 60.055285834625096 |
| phase_noise_sigma_rad_0p1pi_seed11 | 0.8761262155718578 | 67.5907522515338 |
| phase_noise_sigma_rad_0p2pi_seed17 | 0.6434186510607378 | 62.192542498667 |
| phase_noise_sigma_rad_0p4pi_seed23 | 0.016523540359094143 | 45.14436084154206 |

The exact values should be re-read from `metrics.json` before publication tables are finalized.

## Probe Context Panel Work

The user asked whether the paper figure should have insets or separate panels showing the ground-truth probe and an example corrupted probe. The recommendation was separate panels, not insets, because insets would reduce readability of the metric curves.

The initial plan was written at:

- `/home/ollie/Documents/PtychoPINN/docs/plans/revision-studies/probe-mischaracterization-probe-context-panels-plan.md`

The plan was updated after user feedback:

- Do not use phase curvature as the default corruption example.
- Use amplitude blur and/or phase noise as the default corruption.
- Do not show phase for an amplitude-blur corruption because that is visually misleading.
- Show ground-truth amplitude versus amplitude-blurred amplitude.
- Show ground-truth phase versus phase-noise phase.

The implemented figure layout uses a top row of probe-context panels and a bottom row of metric curves:

- `True amp`
- `Amplitude blur 1.0 amp`
- `True phase`
- `Phase noise 0.2 pi phase`
- Amplitude SSIM curve
- Amplitude PSNR curve

The implementation uses separate default conditions for amplitude and phase display:

- Amplitude context: `amplitude_blur_sigma_px_1p0`
- Phase context: `phase_noise_sigma_rad_0p2pi_seed17`

The manifest now records visual-context metadata such as:

```json
{
  "amplitude_condition_id": "amplitude_blur_sigma_px_1p0",
  "phase_condition_id": "phase_noise_sigma_rad_0p2pi_seed17",
  "amplitude_fallback_used": false,
  "phase_fallback_used": false,
  "panel_policy": "separate_probe_context_panels_not_insets",
  "display_channels": ["amplitude", "phase"]
}
```

Generated figure locations:

- Source run figure: `/home/ollie/Documents/PtychoPINN/.artifacts/revision_studies/probe_mischaracterization/full_process_isolated_20260413T020739Z/figures/probe_mischaracterization_stress.png`
- PtychoPINN scratch copy: `/home/ollie/Documents/PtychoPINN/tmp/probe_mischaracterization_stress.png`
- PtychoPINN probe scratch copy: `/home/ollie/Documents/PtychoPINN/tmp/probe/source_run/probe_mischaracterization_stress.png`
- PtychoPINN paper scratch copy: `/home/ollie/Documents/PtychoPINN/tmp/probe/paper/probe_mischaracterization_stress.png`
- Paper repo figure: `/home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png`

The regenerated paper figure was inspected as a PNG:

- Dimensions: `2400 x 1400`
- Mode: `RGBA`

## HIO/ER Investigation

The session also investigated why the HIO/ER single-shot CDI baseline looked poor.

Important interpretation:

- The HIO/ER metrics being discussed were post-reassembly/stitching metrics. The script reconstructed each patch, stacked predictions, called `stitch_predictions(...)`, and then evaluated the stitched reconstruction against `YY_ground_truth`.
- The low SSIM was not just a stitching artifact. Individual reconstructed patches were also poor.
- A PyNX replacement path was planned and attempted for a more standard CDI implementation.
- PyNX installation was attempted and then confirmed by the user from the ESRF source archive, but the final HIO/ER algorithmic interpretation still needed review against the reviewer's actual request.

The user pushed back on the idea that the right answer was to use a ptychographic reconstruction baseline, because the reviewer's comment specifically asked for HIO/ER style comparison. The better target is therefore a known-probe, object-domain HIO/ER implementation that matches the reviewer's intent, not a full ptychographic solver that changes the baseline category.

Outstanding HIO/ER clarification:

- Locate or implement a known-probe object-domain HIO/ER path that solves for the object with the probe treated as known.
- Keep the reviewer's requested comparison category intact.
- Avoid treating a poor single-frame support-constrained exit-wave solve followed by probe division as equivalent to the requested known-probe HIO/ER baseline.

## Verification

Verification commands run after the latest probe-context panel changes:

```bash
MPLBACKEND=Agg python -m pytest tests/studies/test_probe_mischaracterization_stress_test.py -k "load_probe_figure_context or stress_figure or export_paper_assets" -q
```

Result:

```text
8 passed, 30 deselected
```

Full probe stress-test module:

```bash
MPLBACKEND=Agg python -m pytest tests/studies/test_probe_mischaracterization_stress_test.py -q
```

Result:

```text
38 passed in 11.52s
```

Collection check:

```bash
python -m pytest --collect-only tests/studies/test_probe_mischaracterization_stress_test.py -q
```

Result:

```text
38 tests collected
```

Diff whitespace check:

```bash
git diff --check -- scripts/studies/probe_mischaracterization_stress_test.py tests/studies/test_probe_mischaracterization_stress_test.py docs/plans/revision-studies/probe-mischaracterization-probe-context-panels-plan.md
```

Result: exit code 0 with no output.

## Git And Artifact State

The PtychoPINN checkout was dirty before and during this work. Do not assume all modified files are from this session.

Relevant tracked files modified by the probe figure work:

- `/home/ollie/Documents/PtychoPINN/scripts/studies/probe_mischaracterization_stress_test.py`
- `/home/ollie/Documents/PtychoPINN/tests/studies/test_probe_mischaracterization_stress_test.py`

Relevant new plan or summary files:

- `/home/ollie/Documents/PtychoPINN/docs/plans/revision-studies/probe-mischaracterization-probe-context-panels-plan.md`
- `/home/ollie/Documents/PtychoPINN/docs/plans/revision-studies/2026-04-13-revision-study-session-summary.md`

Scratch and generated output locations are ignored or untracked:

- `/home/ollie/Documents/PtychoPINN/.artifacts/`
- `/home/ollie/Documents/PtychoPINN/tmp/`
- `/home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png`
- `/home/ollie/Documents/ptychopinnpaper2/data/probe_mischaracterization_metrics.json`

No final staging or commit was performed for the latest probe-context panel work before this summary was written.

## Open Follow-Ups

- Recheck the active workflow tmux sessions before assuming any long-running run has completed or crashed.
- For the quantitative OOD workflow, use the recorded tmux session and run id above to inspect progress rather than launching a duplicate run.
- For the probe stress figure, confirm the final caption text matches the new panel semantics: amplitude blur is used for amplitude context, phase noise is used for phase context.
- For HIO/ER, continue from the known-probe object-domain baseline clarification rather than reframing the reviewer request as a ptychographic reconstruction baseline.
- Before committing, run `git status --short` and stage only the intended files because the repository contains unrelated dirty changes.
