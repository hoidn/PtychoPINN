# CDI FFNO Refiner/Depth Backlog Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split CDI Lines128 FFNO correction work into an earlier cheap four-block no-refiner wave and a later depth-24 no-refiner final-results wave.

**Architecture:** Keep the current `pinn_ffno` no-refiner rerun as the four-block anchor and make the supervised no-refiner rerun plus table/figure refresh consume that anchor before any depth-24 work. Move depth-24 rows into a later explicit wave, add the missing supervised depth-24 row and final refresh item, and update roadmap/index wording so selectors do not treat depth-24 as the immediate paper-refresh dependency.

**Tech Stack:** Markdown backlog/frontmatter, NeurIPS roadmap/index docs, existing grid-lines FFNO runner contract, JSON/YAML frontmatter validation, active backlog manifest builder.

---

## Source Of Truth

- Current selected item:
  `docs/backlog/in_progress/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun.md`
- Existing cheap-wave item:
  `docs/backlog/active/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun.md`
- Existing table refresh item:
  `docs/backlog/active/2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh.md`
- Existing depth-24 item:
  `docs/backlog/active/2026-05-06-cdi-lines128-ffno-depth24-ablation.md`
- Queue index:
  `docs/backlog/index.md`
- Roadmap:
  `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`

## Tasks

### Task 1: Make the cheap four-block wave explicit

- [ ] Keep `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun` as the current four-block `fno_blocks=4`, `fno_cnn_blocks=0` PINN anchor.
- [ ] Keep `2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun` as the four-block supervised companion.
- [ ] Rewrite `2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh` as the interim four-block no-refiner table/figure refresh, not the final depth-24 paper refresh.
- [ ] Ensure this wave has earlier priorities than the depth-24 wave.

### Task 2: Defer depth-24 into a later explicit wave

- [ ] Update `2026-05-06-cdi-lines128-ffno-depth24-ablation` so it depends on the four-block no-refiner refresh and has a later priority.
- [ ] Add `2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun` for the supervised depth-24 no-refiner companion row.
- [ ] Add `2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh` to update final paper metrics/images only after the depth-24 PINN and supervised rows exist.

### Task 3: Keep roadmap and discoverability consistent

- [ ] Update `docs/backlog/index.md` to show the two-wave dependency order.
- [ ] Update the roadmap note for CDI no-refiner FFNO work to say the four-block wave is the immediate paper-asset refresh and depth-24 is the later final-results wave.
- [ ] Avoid changing active workflow state; the running workflow owns the currently selected in-progress item.

### Task 4: Validate selector-visible state

- [ ] Parse frontmatter for all touched backlog items.
- [ ] Build the active backlog manifest and confirm the new depth-24 items are present with later priorities than the four-block refresh items.
- [ ] Run `git diff --check` on touched files.
