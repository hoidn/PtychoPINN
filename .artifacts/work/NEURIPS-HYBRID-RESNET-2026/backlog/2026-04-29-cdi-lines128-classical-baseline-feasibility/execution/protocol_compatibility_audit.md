# Lines128 Classical CDI Protocol Compatibility Audit

- Date: `2026-04-30`
- Backlog item: `2026-04-29-cdi-lines128-classical-baseline-feasibility`
- Frozen prerequisite root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Audited implementation surface:
  `scripts/reconstruction/hio_cdi_benchmark.py`
- Audited contract surfaces:
  `paper_benchmark_manifest.json`, `metric_schema.json`

## Conclusion

Neither currently exposed classical solver branch is authorized for a same-contract
`lines128` extension row. The backlog item closes truthfully as
`not_protocol_compatible`.

The decisive issue is not one recoverable harness bug. The current classical path
is still the older Table-2 `N=64` study lane, and its emitted artifacts remain
per-row exploratory outputs rather than the required `lines128` paper-bundle
schema. Converting it into a valid `lines128` extension would require a broader
rewrite of dataset generation, contract reconstruction, row-schema collation, and
paper-grade provenance surfaces than the plan's single narrow fix cycle allows.

## Frozen Lines128 Contract

The accepted `lines128` CDI authority requires:

- fixed `N=128`, `gridsize=1`, `synthetic_lines`, `set_phi=true`
- custom Run1084 probe with `pad_extrapolate`, `probe_smoothing_sigma=0.5`,
  `probe_mask=off`
- fixed train/test split counts `2 / 2`, `nphotons=1e9`, fixed `seed=3`
- paper-grade bundle outputs:
  `metrics.json`, `metrics_table.csv`, `metrics_table.tex`,
  `metrics_table_best.tex`, `metric_schema.json`, `model_manifest.json`,
  `paper_benchmark_manifest.json`, and shared comparison visuals
- row-level required fields and provenance surfaces from
  `metric_schema.json`

Evidence:

- accepted bundle contract and required bundle artifacts:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/paper_benchmark_manifest.json:1-77`
- accepted row-schema and paper-grade provenance requirements:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/metric_schema.json:1-35`

## Solver Audit

### `pynx_cdi_hio_er`

- Decision: `not_protocol_compatible`
- Incompatibility categories:
  - `spatial size / split identity`
  - `metric unavailability or untrusted metric path`
  - `provenance gap`
  - `other concrete contract mismatch`
- Evidence:
  - the script hard-codes the older Table-2 lane:
    `TARGET_N = 64` and Table-2 size/offset constants
    at `scripts/reconstruction/hio_cdi_benchmark.py:29-37`
  - all generated data flows through `_table2_cfg_from_args(...)`, which fixes
    `N=64` and the Table-2 geometry rather than the accepted `lines128`
    dataset identity:
    `scripts/reconstruction/hio_cdi_benchmark.py:1969-1995`
  - the metric-contract manifest explicitly marks direct-stitch metrics as
    `fresh_same_split_direct_stitch_not_historical_table2`,
    `table2_compatible=false`, and an exploratory rerun comparator rather than
    a locked paper-bundle row:
    `scripts/reconstruction/hio_cdi_benchmark.py:961-1034`
  - the execution path emits only row-local `metrics_<row>.json`,
    `residuals_<row>.json`, and `recons/<row>/recon.npz`, with no merged
    `lines128` extension-bundle tables/manifests/visual bundle:
    `scripts/reconstruction/hio_cdi_benchmark.py:2439-2515`
- Narrow-fix assessment:
  A same-contract promotion would require replacing the Table-2-specific data
  contract, adding `lines128`-grade bundle collation, and backfilling
  paper-grade row-provenance surfaces. That is a broader rework than the plan's
  single narrow compatibility fix.

### `known_probe_object_hio_er`

- Decision: `not_protocol_compatible`
- Incompatibility categories:
  - `spatial size / split identity`
  - `probe/object representation mismatch`
  - `metric unavailability or untrusted metric path`
  - `provenance gap`
  - `other concrete contract mismatch`
- Evidence:
  - it inherits the same hard-coded `N=64` / Table-2 data path and metric
    contract as the PyNX branch:
    `scripts/reconstruction/hio_cdi_benchmark.py:29-37`,
    `:961-1034`, `:1969-1995`
  - the solver manifest describes this branch as a repo-local fixed-probe
    object-domain diagnostic row whose unknown is the object patch rather than
    the exit wave, not as the external-standard classical solver adopted for
    the main metric path:
    `scripts/reconstruction/hio_cdi_benchmark.py:736-885`
  - the emitted artifacts remain the same row-local exploratory payloads rather
    than the accepted `lines128` extension-bundle schema:
    `scripts/reconstruction/hio_cdi_benchmark.py:2439-2515`
- Narrow-fix assessment:
  Even before bundle/provenance work, this branch would need a reviewed
  same-contract decision on whether the object-domain known-probe formulation is
  acceptable as the optional classical paper row. That exceeds the plan's narrow
  fix boundary.

## Closeout Decision

- Final state: `not_protocol_compatible`
- Full classical launch: not authorized, so no classical run was started
- Preserved six-row bundle: unchanged and still the headline CDI authority
- Authoritative incompatibility root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-classical-baseline-feasibility/`
