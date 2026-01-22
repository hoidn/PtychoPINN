Title: Prepare Response — D0 parity logging + maintainer coordination

Purpose
Provide the local agent with the information needed to finalize the D0 planning step:
implementation-agnostic parity logging spec + maintainer coordination plan.

Please reply with:

1) Baseline selection
- Identify the dose_experiments run to treat as the baseline (scenario ID + brief rationale).
- If multiple legacy runs exist, state which one should be authoritative.

2) Dataset parity details
- Confirm the exact input .npz to use for legacy runs (path + filename).
- Provide checksum(s) (sha256 preferred) for each dataset file.
- If dataset parity is not possible, describe how the legacy dataset was generated and
  provide any metadata/params used (gridsize, N, nphotons, probe settings, grouping).

3) Probe provenance
- Probe source (custom/ideal), file path, and any normalization steps applied.
- Probe scale, mask settings, and any preprocessing performed.

4) Config snapshot
- Provide the config/params snapshot used for the baseline run (or the file path).
- Include key fields: gridsize, N, nphotons, n_groups, neighbor_count, batch_size, nepochs, loss_fn.

5) Commands executed
- Use the dose_experiments notebook at `~/Documents/PtychoPINN/notebooks/dose_dependence.ipynb` as the intended entry point.
- Exact simulate → train → infer commands used for the baseline run (entrypoints + working directory).
- Any deviations from defaults or manual overrides (n_groups, batch_size, etc.).

6) Artifacts available
- Paths and filenames for simulation outputs, training logs, inference outputs.
- If possible, provide a minimal artifact bundle and a brief README describing contents.

7) Preferred handoff
- Where you want artifacts to be placed under this repo (reports path) and any size constraints.

Notes
- The parity logging spec will include probe logging, stage-level stats (raw/grouped/normalized),
  intensity scale values, and inference metrics. Dataset parity is the highest priority.
