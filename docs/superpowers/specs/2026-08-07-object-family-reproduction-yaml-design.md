# Compact Object-Family Reproduction YAML Design

## Scope

This design governs only the artifact-local reproduction harness under
`.artifacts/integration/object_family_n128_seed3_20ep_20260807_reproduction/`.
It does not define a public PtychoPINN configuration schema or change the
configuration accepted by `grid_lines_torch_runner.py`.

## Goal

Make the authored reproduction YAML describe the executable study rather than
its historical provenance. Preserve every explicit experiment and physics
parameter used by the 12-arm result while removing prose, duplicated output
paths, historical-reference metadata, and manually maintained integrity pins.

## Authored Configuration Contract

The compact YAML contains only:

- `source`: source worktree, runner path, and CUDA device selection;
- `output`: one base directory;
- `datasets`: train and test NPZ paths for each object family;
- `matrix`: ordered families, architectures, and profiles;
- `preflight`: the two quality-gate rows;
- `common`: all explicit options shared by every arm;
- `profiles`: all explicit profile-specific options; and
- `collation.crop_border`.

The YAML does not contain a description, study identifier, historical artifact
references, source commit, cleanliness policy, submodule pins, dataset hashes,
probe lineage, duplicated phase output paths, copied-output paths, descriptive
alignment labels, or figure filenames.

Options currently equal to runner defaults remain explicit. The compact schema
must not make the scientific recipe depend on future default values.

Dataset split values are direct paths rather than `{path, sha256}` mappings.

## Paths And Outputs

`output` is resolved relative to the compact YAML file. For
`output: output/`, the harness writes:

```text
<yaml-directory>/output/
├── preflight/
└── full/
```

Each phase owns its normal run artifacts and its derived comparison table. The
collator derives `preflight_comparison_table.png` or
`full_comparison_table.png` from the selected phase. It does not create a
second copy under repository `tmp/`.

Input paths and the source worktree remain repository-root relative unless
absolute.

## Generated Evidence

Removing provenance from authored YAML does not suppress runtime evidence. A
launch is admissible only when the source worktree has no tracked changes and
the runner plus its required FRC files exist. At launch, the harness records an
exact source observation consisting of the source `HEAD`, recursively listed
submodule checkout revisions and status, and SHA-256 hashes of the runner and
harness files. It also computes and records each input dataset hash.

The harness repeats the source observation immediately before and after every
arm and at completion, and re-hashes datasets at completion. Any difference
from the launch-time observation fails the run. Per-arm invocation records must
also report the launch-time source commit and a clean tracked tree. This detects
source or input mutation during a run without requiring users to author hashes,
Git revisions, or a cleanliness switch.

The original `study.yaml` and completed output bundle remain immutable because
their existing attestation records include the original YAML hash. The compact
contract is provided separately as `reproduce.yaml`.

## Failure Behavior

The harness rejects missing source, runner, or dataset paths; an occupied phase
output directory; source or dataset mutation during a run; failed arms;
incomplete artifacts; invocation mismatches; non-finite metrics; and incorrect
CI probe selection.

## Verification

Tests must establish that:

1. the compact YAML parses and expands to the same 12 ordered rows;
2. all generated runner arguments other than output paths are semantically
   identical to the attested recipe;
3. `output: output/` resolves relative to `reproduce.yaml` and phase names are
   derived correctly;
4. launch-time provenance is recorded and mutation is rejected;
5. preflight and full figure names are derived correctly; and
6. the existing completed result and its original YAML are not modified.

No GPU rerun is required for this schema-only adapter change; command expansion,
artifact-fixture tests, and a dry run provide the claim-matched evidence.
