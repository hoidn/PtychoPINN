# Pydantic Family Adoption Design

**Status:** Approved and implemented on `refactor-internal` as of 2026-07-28.
Both conditional family decisions are closed.

**Implementation state on `refactor-internal`:** The representation-preserving
simulation adapter and the return-new public and Torch resolver substrates are
present. The public Model/Training/Inference family adopted three cached
complete-snapshot adapters after its measured gate passed. The Torch
Data/Model/Training/Inference family terminally retained its explicit manual
resolver because a contract-faithful adapter would add more structural policy
than it could delete. Persistence codecs, execution resolution, and the
bounded legacy bridge remain outside Pydantic.

**Supersedes:** the 2026-07-28 v1 draft (formerly at
`.worktrees/ci-compatibility-ablation/tmp/`, now retired).

**Parent authority:**
`docs/superpowers/specs/2026-07-28-configuration-boundary-architecture.md`
owns the per-family adopt/hold/reject portfolio and the 8-condition Pydantic
Adoption Gate.

**Transition prerequisite:**
`docs/specs/2026-07-28-configuration-compatibility-retirement-and-schema-convergence-design.md`
owns the ordering of compatibility retirement, the `params.cfg` strangler,
and this post-strangler schema decision. This is the separately approved child
that closed the parent architecture's former two Hold rows. The parent
portfolio now records the evidence-backed outcomes.

**Family authorities (each owns its boundary contract and records its eventual
decision):**

- `docs/superpowers/specs/2026-07-28-public-config-resolution-design.md` —
  §"Pydantic Decision" (adopted; records the implemented shape and measured
  gate).
- `docs/superpowers/specs/2026-07-28-torch-config-resolution-design.md` — §14
  (`retain manual`; records the executable Pydantic 2.12.3 probes and final
  deletion ledger).
- `docs/superpowers/specs/2026-07-28-pydantic-simulation-validation-design.md`
  — implemented representation-preserving complete-document adapter.
- `docs/superpowers/specs/2026-07-28-configuration-persistence-boundaries.md`
  — unchanged; its exclusions bind this design.

**Reconciliation decisions:**

1. Both public and Torch adoption are conditional on a post-strangler
   dry-run deletion ledger; neither adapter end state is presumed.
2. Removed branch reconciliation, updater retirement, execution-lane cleanup,
   and global-state migration from this design's stages. Their accepted parent
   and sibling designs own those transitions.
3. Reclassified exact legacy projection evidence as conditional on a remaining
   bridge rather than a permanent post-strangler adoption gate.
4. Corrected the updater inventory: return-new factories avoid
   `update_existing_config()`; transition Stage 1 migrated its remaining
   callers and removed the tolerant updater, while enclosing APIs followed
   their own support contracts.
5. `TrainingConfig` and `InferenceConfig` remain mutable and retain their
   `__post_init__` compatibility behavior; only public `ModelConfig` is frozen.
6. Dead-code and compatibility deletion never count as adapter-attributable
   adoption evidence.

**Goal (achieved):** After compatibility and global-state reduction, decide independently
whether the public `ModelConfig`/`TrainingConfig`/`InferenceConfig` family and
the Torch `DataConfig`/`ModelConfig`/`TrainingConfig`/`InferenceConfig` family
should extend the approved simulation cached-`TypeAdapter` pattern demonstrated
by the public reference implementation. Each family
adopts only if the measured replacement deletes more structural-validation
complexity than it adds; otherwise it terminally records retain-manual. Every
other portfolio row keeps its current treatment. The net-deletion gate is
also this design's instrument for the transition parent's secondary goal of
net production line reduction (parent §1); a retain-manual outcome that adds
nothing serves that goal exactly as well as a net-deleting adoption.

In this design, **post-strangler** means
`modern_isolation_complete`: supported modern paths have zero `params.cfg`
reads. A named scoped bridge may remain for a declared legacy consumer; final
`global_bridge_retired` is not an adoption prerequisite.

---

## 1. What "consistent Pydantic use" means here

Consistency is **one pattern at every complete-snapshot structural boundary**,
not Pydantic everywhere. The end-state portfolio:

| Boundary | End state | Change? |
|---|---|---|
| Simulation recipe | Cached `TypeAdapter` over unchanged stdlib dataclasses | adopted and implemented |
| Public Model/Training/Inference | Three cached complete-snapshot adapters; manual semantic/resource validation retained | adopted after a positive measured gate |
| Torch Data/Model/Training/Inference | Explicit transactional resolver and manual validation | retain manual after a negative measured gate |
| `DatagenConfig` | Compatibility view of `SimulationConfig`; no independent schema (negatively asserted, §8) | no |
| `PyTorchExecutionConfig` | Manual validation (environmental/provenance contract) | no |
| CLI patches | Primitive argparse; merge, then validate the complete result | no |
| `params.cfg` projection | Pydantic rejected; lifecycle handled by the strangler design | no |
| ModelSpec/artifacts/checkpoints | Reject (versioned codecs) | no |
| MLflow dictionaries | Reject as authoritative identity | no |

The retained-manual rows are part of the consistent architecture, not
exceptions to it: partial patches, environmental resolution, and versioned
persistence have lifecycle contracts a snapshot validator cannot own (parent
doc, "Explicitly Rejected Architecture"). A "retain manual" outcome for the
public or Torch family is likewise a consistent terminal state, not a
deferral. A proposal to exceed this scope is a parent-architecture amendment,
out of scope here.

## 2. Contract-relevant implemented state

The approved resolver substrate and family decisions below are implemented on
`refactor-internal`.

### Simulation family mechanism

- `@with_config(ConfigDict(extra="forbid", revalidate_instances="always",
  validate_default=True))` stacked above `@dataclass(frozen=True)` attaches
  Pydantic configuration to stdlib dataclasses without changing the class's
  dataclass nature. Decorator order is required.
- One module-level cached adapter (`_SIMULATION_CONFIG_ADAPTER =
  TypeAdapter(SimulationConfig)`), used in two modes: conversion of a raw
  mapping at the input boundary, and `validate_python(instance, strict=True)`
  as a **check-only** re-certification whose reconstructed result is
  discarded (caller's object identity preserved).
- An exact-representation vocabulary: `Annotated[...,
  BeforeValidator(_require_exact_int | _require_exact_bool |
  _require_exact_finite_number | _require_exact_str |
  _require_exact_optional_int | _require_pair_container)]` — `type(v) is`
  checks, so `bool` is not `int`, numpy scalars and str/int subclasses are
  rejected, and canonical-hash identity (`4` vs `4.0`) is preserved.
- A domain error facade (`_raise_simulation_validation_error`) translating
  `ValidationError` into stable dotted-path domain messages; raw Pydantic
  formatting is not a public contract.
- None of the five simulation records defines `__post_init__`. F1/F7 therefore
  separately proved mutable public-record revalidation and post-init behavior
  before the public family changed.

### Public family (Stage B outcome: adopted)

- `ModelConfig` is frozen. **`TrainingConfig` and `InferenceConfig` are
  mutable** stdlib dataclasses whose `__post_init__` (a) migrates deprecated
  `n_images` to `n_groups` with a `DeprecationWarning` via
  `object.__setattr__`, and (b) on `TrainingConfig`, injects the default
  `n_groups=512` when neither field was supplied — i.e., a field value is
  written **after** any per-field validation would run. The public child
  design keeps this constructor behavior during the compatibility phase.
- The resolver in `ptycho/config/resolution.py` constructs complete
  candidates (`ModelConfig(**merged)`, `TrainingConfig(model=..., **values)`,
  `InferenceConfig(...)`) after canonicalizing the group alias itself
  (`n_images` nulled, `n_groups` set), so the deprecated `__post_init__`
  warning path does not fire on resolver-constructed records. Adapter
  conversion preserves exactly this behavior.
- The measured post-strangler hand-written structural surface deleted by the
  adoption included the
  `_require_exact_int` / `_require_optional_int` / `_require_exact_bool` /
  `_require_optional_bool` / `_require_literal` / `_require_number` /
  `_require_path` suite, the module-level domain frozensets, and the
  scalar/domain/membership bodies of `validate_model_config_structure` /
  `validate_training_config_structure` /
  `validate_inference_config_structure`.
  The semantic remainder (object-policy join, sampling coherence,
  realspace-weight cross-rule, loss-weight relationships) stays manual, as
  do `validate_runnable_training_config` and `validate_inference_resources`.
- Before adoption, the manual surface stated schema facts more than once across
  `config.py` and `resolution.py`, in four classes with unequal drift
  protection. The dry-run ledger accounted for each class separately:
  1. field-ownership name sets (`_MODEL_INPUT_NAMES`,
     `_TRAINING_INPUT_NAMES`, `_INFERENCE_INPUT_NAMES`) — hand-listed but
     tripwired by import-time asserts against `dataclasses.fields()`;
     drift is loud except under `python -O`, which strips asserts;
  2. closed value domains — public `Literal[...]` annotations had unchecked
     module-level frozenset twins (`_MODEL_ARCHITECTURES`,
     `_AMP_ACTIVATIONS`, `_PUBLIC_BACKENDS`, `_TORCH_LOSS_MODES`,
     `_GRADIENT_CLIP_ALGORITHMS`, `_OPTIMIZERS`, `_SCHEDULERS`,
     `_OBJECT_LAYOUTS`, `_TRAINING_CANVASES`); no `get_args()`
     cross-check exists, so the frozenset governs runtime while the
     annotation can silently diverge;
  3. `Path`-typed field sets (`_TRAINING_PATH_NAMES`,
     `_INFERENCE_PATH_NAMES`) — unchecked restatements of the
     annotations;
  4. per-field `_require_*` bodies — further statements of each field's
     name and constraint with no fields-versus-validator coverage
     tripwire (the gap class that produced the pre-campaign `set_phi`
     hole).
  Classes 2–4 supplied the adapter-deletable duplication measured by the
  ledger; class 1 is the ownership layer that survives adoption.
- The pre-campaign validators `validate_model_config` /
  `validate_training_config` / `validate_inference_config` in
  `ptycho/config/config.py` have zero production callers but remain
  exported, and their domains contradict the structure validators (§6).
  Their disposition belongs to ordinary compatibility cleanup and their
  removal is **not** adoption evidence.

### Torch family (Stage C outcome: retain manual)

- The four runtime records in `ptycho_torch/config_params.py` are mutable
  stdlib dataclasses totaling 157 effective fields (DataConfig 21,
  ModelConfig 83, TrainingConfig 44, InferenceConfig 9). The source contains
  158 declarations because `TrainingConfig.output_dir` appears twice; only
  its later declaration survives dataclass construction. The records are
  **mostly unvalidated today**: `DataConfig`, `TrainingConfig`, and
  `InferenceConfig` have no constructor validation; `ModelConfig.__post_init__`
  validates only the spectral-bottleneck fields; `amp_activation` is a bare
  `str`. `ModelConfig` carries tensor-typed fields (`probe_mask:
  Optional[Union[bool, torch.Tensor]]`, `probe_mask_tensor:
  Optional[torch.Tensor]`).
- The post-isolation adoption-favorable deletion ceiling was 109 production
  lines, including the flat chains in
  `_validate_training_owner_domains` and the type/membership portion of
  `_validate_data_and_model` in `ptycho_torch/config_resolution.py`, domain
  twins, numeric helpers, and unique call sites. Contract-faithful adoption
  has an addition floor of 194 lines: 157 exact-or-explicitly-unchecked field
  decisions plus adapters, configuration, tensor identity handling, the error
  facade, and conversion/revalidation call points. The minimum net increase
  is therefore 85 lines, so the simplification gate fails.
- Mutation retirement and modern-state isolation prerequisites are complete.
  Tensor fields remain in `ModelConfig` because relocating them would be a
  separate payload and persistence-identity migration; pricing their
  Pydantic bridge makes adoption still less favorable. Checkpoint
  rehydration constructors remain sanctioned persistence boundaries.

### Public-reference probe facts (torch design §14, Pydantic 2.12.3)

Naive `TypeAdapter` is unusable as-is — lax mode coerces and ignores extras;
bare `strict=True` rejects mappings wholesale; instances are not revalidated
by default; `TypeAdapter(ModelConfig)` fails schema generation on
`torch.Tensor`; `ConfigDict` cannot be passed to `TypeAdapter` for a stdlib
dataclass (must attach to the class). Each has a designed answer below; none
is a prohibition (§14: "feasibility facts, not a permanent prohibition").

## 3. Target end state per family

Both family gates require completed updater retirement, execution compatibility
closure, and the `modern_isolation_complete` strangler milestone. Neither gate
may count deletion performed by those prerequisites as Pydantic adoption
evidence.

### 3.1 Public family (Stage B — adopted)

Stage B remeasured the post-strangler manual surface. The gate passed, so the
mechanics below are implemented.

**Implemented mechanics:** the shape is fixed by the public design's
own "only acceptable later shape":

- `@with_config` (same `ConfigDict` policy as simulation) attached to
  `ModelConfig`, `TrainingConfig`, `InferenceConfig` in
  `ptycho/config/config.py`. `ModelConfig` stays frozen;
  `TrainingConfig`/`InferenceConfig` stay mutable; all stay stdlib
  dataclasses with `Literal` string domains. `__post_init__` behavior is
  unchanged and fixture-pinned (F7).
- Three cached root adapters owned by the boundary module
  `ptycho/config/resolution.py`: one each for `ModelConfig`,
  `TrainingConfig`, and `InferenceConfig`. The latter two embed
  `ModelConfig`, while the first serves the standalone complete Model
  boundary.
- Call points: (a) conversion-mode validation of the **complete merged
  mapping** after file/CLI precedence and alias resolution — replacing the
  type/membership/Path-conversion branches inside
  `validate_*_config_structure`; because the resolver canonicalizes
  `n_groups` first, adapter-driven construction must not re-trigger the
  deprecated alias path; (b) strict check-only instance revalidation
  wherever `validate_*_config_structure` is called on an
  already-constructed record today. Partial patches never enter an adapter.
- Retained manual: alias/precedence/ownership logic, unknown-key policy at
  the resolver, semantic joins, `validate_runnable_training_config`,
  `validate_inference_resources`, and any still-declared legacy projection.
- Deleted, and countable as adoption-gate 7 evidence: only the schema-aware
  branches inside the three `*_structure` validators, the `_require_*`
  primitive suite, manual Path conversion, and the domain frozensets the
  `Literal` annotations subsume. Orphaned or compatibility-only code removed
  before this cut is excluded from the ledger.

### 3.2 Torch family (Stage C — retain manual)

Stage C evaluated rather than presumed adoption. Its ledger failed the
net-deletion gate, so no Torch Pydantic production change was made.

**Entry criteria (all required before the gate is even evaluated):**

1. The transition parent's mutation-retirement stage is complete:
   `update_existing_config` has no supported production definition, import, or
   call; supported paths run through the transactional resolver; and direct
   constructions are drained through factory baselines or sanctioned
   versioned persistence boundaries.
2. A tensor-relocation decision is recorded in the torch child design:
   either the tensor-typed fields move out of `ModelConfig` into payload
   ownership (recommended — tensors are payload, not configuration, and
   `ModelSpec` already owns structural identity), or they remain and the
   gate prices the `arbitrary_types_allowed` bridge (F2).
3. Feasibility fixtures F1, F2 (if tensors remain), and F6 are green on the
   pinned Pydantic version.
4. Execution compatibility-lane closure and the `params.cfg` strangler's
   zero-modern-read milestone are complete.
5. The family's accepted-value table exists and its §6 contract items are
   resolved.

**Go/no-go instrument — dry-run deletion ledger:** before any production
change, enumerate the lines the adapter would delete (the flat chains, §2)
against the lines it would add (field annotations required for the adapter
to check anything, tensor policy, error facade, revalidation call points).
Annotation tightening that introduces new rejections is counted as contract
expansion, not deletion, and each such field requires its own compatibility
decision. If the ledger does not net positive, Stage C terminates with
"retain manual" recorded in torch child design §14 — a valid, successful
outcome.

**Mechanics if the gate passes:**

- `@with_config` on the four records with the same `ConfigDict` policy,
  plus `arbitrary_types_allowed=True` only if tensor fields remain.
- Tensor fields (if retained) get identity-preserving `Annotated`
  validators (`BeforeValidator` asserting `isinstance(v, torch.Tensor)` and
  returning the same object). Adapters are validation-only: no
  `dump_python`, `dump_json`, or schema-derived serialization anywhere
  (persistence exclusion; torch gate 5's serialization clause is satisfied
  by *not serializing*).
- One cached adapter per record, owned by
  `ptycho_torch/config_resolution.py`, applied at the structural stage of
  `resolve_training_bundle` / `resolve_inference_bundle` on **complete
  candidate records**, replacing the flat chains.
- Revalidation points for mutable records (parent doc, "Resolved runtime
  snapshot"): strict check-only revalidation at resolver exit and at the
  `config_factory` consumption boundary before projection/persistence —
  after the last supported mutation.
- Retained manual: ownership registries, precedence, provenance, alias
  conflicts, bundle-level semantic validators (torch gate 7), all
  execution-config validation, all persistence codecs.

## 4. Design decisions

1. **Attach config to the real classes; no shadow models.** The probe
   blocker ("ConfigDict cannot be passed to TypeAdapter") is resolved by
   `@with_config`, the approved mechanism demonstrated by the public-reference
   simulation implementation. The probe's
   "ephemeral configured dataclass" workaround is rejected — it is exactly
   the parallel input model the parent gate 8 forbids.
2. **Check-only strict revalidation discards the reconstruction.** Matches
   the simulation precedent. Caller object identity and tensor aliasing are
   untouched at check points; only construction-boundary conversion adopts
   the adapter's returned instance.
3. **One shared exact-representation vocabulary, primitives only.** The public
   family passed its gate, so the
   `_require_exact_*` BeforeValidators and the strict scalar `Annotated`
   aliases live in a small internal module
   (`ptycho/config/strict_types.py`) imported by adopting families
   (`ptycho_torch` already depends on `ptycho`). The module hosts only
   representation primitives — exact-type scalar guards and their aliases.
   Family policy (pair-container rules, closed string domains, path
   handling) stays declared in each family's own module and accepted-value
   table. A small leaf utility with no field lists and no ownership data is
   not the parent-rejected "central schema registry." This is code motion,
   not behavior change, and its cost remains in the deletion ledger.
4. **`bool`-as-`int` edge** (torch gate 2): covered by `type(v) is int` in
   `_require_exact_int` — `True` is rejected where an `int` is required.
   Fixture-pinned per family.
5. **Unknown keys:** the resolver's fail-closed unknown-key rejection
   remains the authoritative, family-uniform behavior (torch gate 3);
   `extra="forbid"` on the adapters is a backstop, not the contract.
6. **Error facades stay per-family** (parent invariant 7): each family owns
   a translator producing stable dotted-path fragments; facades may share
   the internal loc-formatting helper but the public message contracts are
   declared per family in their accepted-value tables.
7. **Adapters never see partial patches** (parent gate 2, torch gate 8,
   public non-use list): CLI/notebook/study overrides merge to a complete
   candidate first, always.
8. **Container conversion policy is per-family, declared in the
   accepted-value table** (e.g., whether Torch `grid_size` accepts
   list-to-tuple conversion at the mapping boundary the way simulation
   pair-containers do). No silent global policy.
9. **`__post_init__` is part of the validated contract, not an accident.**
   For records that mutate fields post-construction, the accepted-value
   table must state the post-init outcome (e.g., `TrainingConfig` resolved
   without a group count stores `n_groups=512`), and F7 proves the adapter
   observes it identically in conversion and strict-instance modes. No
   `__post_init__` is added, removed, or reordered by adoption.
10. **Deletion-ledger hygiene.** Dead code (the orphaned facade validators,
    unreferenced helpers) removed before the family cut never counts toward
    any family's net-deletion gate. Gate evidence is duplication the adapter
    itself replaces. Interim manual-layer hardening under the transition
    parent's drift-tripwire provision (deriving domain frozensets from
    their `Literal` annotations, deriving or asserting the `Path`-field
    sets) is likewise ordinary maintenance of the retained-manual
    architecture: excluded from the ledger, and it legitimately shrinks
    the adapter-deletable surface the ledger later measures. A
    post-hardening ledger that no longer nets positive is a correct
    retain-manual signal, not sandbagging.
11. **CLI Path handling stays primitive** at the CLI (parent invariant 5);
    Path conversion is owned by the adapter boundary only where the manual
    conversion it deletes lives today (F5 pins representation).
12. **`DatagenConfig` never grows an adapter.** Enforced by the scoped static
    architecture check (§8), since it is the one portfolio row with no
    structural mechanism guarding its "no independent schema" state.

## 5. Completed feasibility proofs

Per parent gate 4 ("proven with the installed Pydantic version rather than
assumed"), each applicable item was established by an executable pytest
fixture before its family's decision.

| ID | Proves | Family | Gate(s) |
|---|---|---|---|
| F1 | `@with_config` on **mutable** stdlib dataclasses: strict mapping conversion, `revalidate_instances="always"` detects an invalid mutated instance (probes showed default mode does not), `extra="forbid"` | public **and** Torch | arch 4; torch 2, 4 |
| F2 | Tensor fields: schema generation succeeds with `arbitrary_types_allowed` + `Annotated` identity validator; the same `torch.Tensor` object (by `id`) survives conversion and revalidation; no serialization attempted | Torch (only if tensors remain per §3.2) | torch 5 |
| F3 | `@with_config` is reflection-neutral on frozen **and** mutable records: `dataclasses.fields`, `inspect.signature`, positional construction, `==`, `hash` (frozen), `replace`, `asdict`, pickle round-trip unchanged | both | arch 5; public 8 |
| F4 | Nested root adapters (`TrainingConfig` embedding `ModelConfig`): conversion + strict instance modes, dotted error paths at both levels | public | arch 1; public 2, 3 |
| F5 | `Path` fields: accepted inputs and exact stored representation match the current manual conversion | public | public 4 |
| F6 | Internal versioned-wire fixtures plus `tests/torch/test_config_pydantic_artifacts.py` and `tests/fixtures/config/*` coverage pass unchanged — byte-exact persisted payloads | Torch | arch 6; torch 10 |
| F7 | `__post_init__` interplay: adapter conversion constructs through the real dataclass so post-init runs exactly once with today's semantics — `n_groups=512` injection when unsupplied, deprecated-alias warning parity (including **no** duplicate warning on resolver-canonicalized input), strict-instance revalidation of a post-init-mutated record is stable/idempotent, and `validate_default=True` does not alter post-init outcomes | public | arch 4, 5; public 5, 8 |
| F8 | Conditional legacy-projection safety: while a declared legacy bridge remains, `dataclass_to_legacy_dict()` and `update_legacy_dict()` output for representative Training/Inference records is byte-identical to the pre-cut contract. If the bridge has retired, the applicable versioned historical codec owns this evidence instead. | public | arch 6; public 9 |

F6 remains persistence acceptance evidence. F8 applies only while its bridge
or a versioned historical codec remains contractually live; an already retired
global bridge is not resurrected as an adoption gate. Focused evidence also
confirmed that both families' resolver/validator paths neither read nor write
`params.cfg`.

## 6. Contract-alignment outcomes

The parent doc required conflicts to be resolved in the owning contract before
validation froze one interpretation. The implementation selected:

1. **`batch_size`:** the public structural boundary requires an exact built-in
   positive integer; the orphaned power-of-two facade is not authoritative.
2. **`N` domain:** public `Literal[64, 128, 256]` and the structure
   validator agree on the three-value set, while `ptycho/model.py` accepts
   64–1024 powers of two. The former is the public authoring domain; the latter
   is protected legacy-runtime tolerance.
3. **Scheduler domains:** the public four-value set and the Torch six-value set
   (including `MultiStage` and `Adaptive`) remain distinct. Each family retains
   its documented domain; bridges do not reinterpret Torch-only values.
4. **Accelerator choices:** unresolved execution accepts `auto`, `cpu`, `gpu`,
   `cuda`, or `mps`; `tpu` is rejected.
5. **Unknown keys and groups:** mapping boundaries fail closed, `n_groups` is
   canonical, and deprecated `n_images` is normalized at the public boundary.

Both family evaluations ran only after these outcomes, the transition
prerequisites, and the installed-version fixtures were green.

## 7. Completed staged approach

This child began after the transition parent reached its schema-convergence
stage. Earlier branch reconciliation, updater retirement, execution-lane
closure, and global-state strangling remained outside this design.

- **Stage A — proofs and contracts: complete.** Remeasured both
  post-strangler manual surfaces and ran the feasibility fixtures.
- **Stage B — public gate: adopted.** The final production diff is
  +304/-530 (226 fewer lines). The fixed 321-line eligible structural
  deletion surface exceeds all 304 production insertions by 17 lines even
  under the conservative ledger.
- **Stage C — Torch gate: retain manual.** The 109-line deletion ceiling is
  smaller than the 194-line addition floor, so adoption would add at least 85
  production lines before delivering equivalent policy.
- **Stage D — closure: complete.** The parent portfolio, family decisions,
  user/developer configuration guide, and documentation routing record those
  outcomes.

## 8. Acceptance evidence requirements

A family counts as adopted only with all of:

- Focused suites green: the family's resolution/validation tests, the §5
  fixtures, byte-exact persistence evidence (F6), and F8 only while its
  declared compatibility boundary remains.
- **Measured adapter-attributable net deletion** (parent gate 7; torch gate
  9; public item 10): `git diff --numstat` over the family's cut showing
  hand-written type/domain/membership branches deleted exceeds adapter +
  annotation + facade lines added, excluding all pre-cut compatibility,
  strangler, and dead-code deletions; and the replaced validators have fewer
  policy branches without moving them behind opaque helpers. An
  adoption that adds net structural-validation code fails its own gate and
  is reverted.
- A scoped static architecture check: no `BaseModel`, no
  `pydantic.dataclasses`, no `validate_assignment`, no `model_dump` /
  `dump_python` / `dump_json` in production config modules; `import
  pydantic` confined to the declared owner modules; no `TypeAdapter`
  constructed over `DatagenConfig` (decision 12). This may be a focused test
  or a claim-matched source check; the invariant, not a repository-wide grep
  implementation, is authoritative.
- Unchanged-contract checks: dataclass reflection/signature/equality pins
  (F3 patterns) running against the migrated records; `__post_init__`
  behavior pins (F7); no new `params.cfg` dependency; and applicable current
  persistence/legacy projection evidence unchanged.

## 9. Explicitly rejected

- **Pydantic for the Reject/manual rows** (`PyTorchExecutionConfig`, CLI
  patches, `params.cfg`, ModelSpec/artifacts/checkpoints, MLflow):
  re-affirmed; rationale owned by the parent doc. "Consistent" does not
  mean "total."
- **`BaseModel` / `pydantic.dataclasses`:** rejected in all three family
  designs — constructor-time validation, reflection, `replace`, mutation
  timing, and wire behavior all change without a governing requirement.
- **Central schema/validation registry or generic framework:** parent
  rejection stands; adapters and facades stay family-owned. The
  `strict_types` primitives module is bounded by decision 3 and is not a
  registry.
- **Shadow validated models** (the probe's ephemeral configured dataclass):
  violates parent gate 8; superseded by `@with_config` on the real classes.
- **Adapter-level defaulting of partial patches:** parent boundary-kind
  rule — "Pydantic defaults must not turn an omitted patch value into an
  explicit override."
- **Adoption by presumption:** committing either family to an adapter end
  state before its deletion ledger nets positive. Stages B and C exist to
  decide, not to ratify.
- **Counting dead-code or migration removal as adoption evidence:** prior
  compatibility and strangler deletions are free wins with or without
  Pydantic and prove nothing about an adapter.
- **Tensor-carrying config records as a permanent adapter surface:**
  `arbitrary_types_allowed` plus identity validators is an acceptable
  priced bridge inside Stage C, not an end state; the recommended end state
  moves tensors to payload ownership (§3.2, entry criterion 2).

## 10. Resolution of v1 open questions

1. *Shared `strict_types` module vs per-family duplication* — the public family
   adopted, so the shared primitives-only leaf exists (decision 3). Its cost is
   included in the public family's ledger.
2. *Fold CLI Path handling into the adapter boundary?* — no; CLI stays
   primitive (decision 11).
3. *Negative test for `DatagenConfig`?* — use the scoped static architecture
   check described by decision 12 and §8.
