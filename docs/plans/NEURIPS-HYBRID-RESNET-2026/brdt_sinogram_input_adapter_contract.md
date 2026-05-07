# BRDT Sinogram-Input Adapter Contract Summary

> **Owning backlog item:** `2026-05-07-brdt-sinogram-input-adapter-contract`
> **Claim boundary:** feasibility-only adapter/readiness authority. NOT
> benchmark-performance or manuscript evidence. BRDT remains additive candidate
> work only; the required NeurIPS pillars are still CDI `lines128` and
> PDEBench CNS.
> **Contract authority:** this summary is the discoverable authority for the
> BRDT learned-model input contract in which `ffno` and `sru_net` consume the
> measured complex sinogram directly while the Born inverse remains a
> non-learned reference only.

## 1. Identity And Scope

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Lane: Born-Rytov diffraction tomography (BRDT) candidate study
- Artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-adapter-contract/`
- Dedicated smoke root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-adapter-contract/smoke/`
- Execution plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-adapter-contract/execution_plan.md`

This item hardens the adapter and runner surfaces only. It does not launch the
successor `40`-epoch paper-evidence item and it does not relabel any historical
Born-image-input BRDT bundle.

## 2. Locked Input Contract

The learned-model BRDT path now supports two explicit input modes:

- historical lineage: `input_mode="born_init_image"`
- new learned-model contract: `input_mode="sinogram"`

The legacy alias `direct_sinogram` remains rejected. The new learned-model
contract is:

- dataset/batch sinogram layout: `(B, 64, 128, 2)`
- model-facing layout: `(B, 2, 64, 128)`
- learned rows in scope: `ffno`, `sru_net`
- target-grid output: `(B, 1, 128, 128)`
- learned-model input source: measured complex sinogram real/imag channels
- Born-consistency target source: the same measured complex sinogram
- Born inverse role: non-learned reference only

The fixed Born inverse is intentionally absent from the learned sinogram-input
path. It remains available only for the classical reference path, optional
visualization, and the preserved historical `born_init_image` lineage.

## 3. Historical Lineage Split

The earlier BRDT summaries remain valid, but only for the old
`born_init_image` contract:

- `brdt_task_adapters.md`
- `brdt_preflight_summary.md`
- `brdt_ffno_row_extension_summary.md`
- `brdt_corrected_ffno_row_rerun_summary.md`
- `brdt_supervised_born_40ep_paper_evidence_summary.md`
- `brdt_corrected_ffno_40ep_rerun_summary.md`

Those artifacts are still discoverable and still usable for that historical
contract. This item does not rewrite them as if they had always consumed raw
sinograms directly.

## 4. Readiness Proof Surface

The feasibility-only smoke authority for the new contract is the dedicated
adapter-contract root above. Required outputs are:

- `smoke_summary.json`
- `smoke/ffno/adapter_contract.json`
- `smoke/ffno/invocation.json`
- `smoke/ffno/invocation.sh`
- `smoke/sru_net/adapter_contract.json`
- `smoke/sru_net/invocation.json`
- `smoke/sru_net/invocation.sh`

`smoke_summary.json` records:

- `input_mode="sinogram"`
- the consumed dataset manifest path
- learned rows executed exactly as `ffno` and `sru_net`
- per-row status
- proof that learned-model input and Born-consistency targets both come from
  the measured complex sinogram
- proof that the Born inverse stays `non_learned_reference_only`

## 5. Interpretation

- This item upgrades BRDT discoverability and harness correctness, not BRDT
  competitiveness.
- The new contract is intentionally separated from the historical
  `born_init_image` bundles so old and new BRDT evidence are not mixed.
- The next allowed follow-up is
  `2026-05-07-brdt-sinogram-input-40ep-paper-evidence`, which may build on
  this contract and its smoke proof.
- Until that successor item completes, the new sinogram-input lane remains
  readiness-only and does not add a benchmark row to the manuscript evidence
  package.
