# Callchain Tracing (Evidence‑Only, Question‑Driven)

These instructions drive an active, question‑oriented exploration of the codebase to map and explain the hot path (Entry → Config → Core Pipeline → Normalization/Scaling → Sink), produce a concise callgraph with file:line anchors, and propose numeric tap points for parity/debugging — without modifying production code.

## Inputs (provided at invocation)

- <analysis_question> (required): Free‑text description of the behavior/bug/perf issue to investigate.
- <initiative_id> (required): Short slug for the output folder under `plans/active/<initiative_id>/reports/` (e.g., `cli-scaling`).
- <scope_hints> (optional): Hints like “CLI flags”, “normalization/scaling”, “IO”, “perf”, “latency hotspot”.
- `roi_hint` (optional): Minimal input/ROI to exercise the path (e.g., a single pixel, a tiny record, or a 1‑request sample).
- `namespace_filter` (optional): Primary module/package prefix for dynamic tracing/profiling (reduces noise).
- `time_budget_minutes` (optional, default 30): Cap exploration and dynamic tracing accordingly.

All outputs MUST be written under `plans/active/<initiative_id>/reports/…` and follow the deliverable formats below.

---

## Discovery Sources (doc‑first indirection)

Resolve anchors through documentation first — do not hardcode source paths in this prompt. Use:

- `docs/index.md` — canonical map to the rest of the docs.
- `docs/architecture.md` and `docs/DEVELOPER_GUIDE.md` — component boundaries, high‑level flow, and two-system architecture notes.
- `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` — parity/trace SOP and test discovery.
- Component contracts referenced from the architecture/dev guides (e.g., `docs/CONFIGURATION.md`, `docs/debugging/*`).
- CLI/runner overview in `docs/COMMANDS_REFERENCE.md`, `README.md`, or workflow guides under `docs/workflows/` and `docs/WORKFLOW_GUIDE.md`.

If the relevant page is missing, fall back to the heuristics in this prompt and RECORD assumptions explicitly in your outputs.

---

## Entry Point Selection (question‑driven)

1) Parse `analysis_question` into tokens (examples):
   - Domain: scaling, normalization, weighting, absorption/attenuation, ROI/masking, output/sink, cache.
   - Surfaces: CLI flags/args, API routes, background jobs, writers.
   - Perf: latency, hotspot, allocation, device/dtype.

2) Build a Candidate Entrypoint Table from docs (not code yet):
   - Candidate (doc anchor/title)
   - Relevance signals (token overlaps, doc keywords)
   - Confidence (High/Med/Low)
   - Expected code regions (module/class/function names if discoverable from docs)

3) Rank by token overlap + doc centrality; pick the top 1–2 as primary entrypoints and 1 fallback. Save the table verbatim.

Write this table to the deliverable `callchain/static.md` (see format).

---

## Active Exploration (code stepping, minimal reads)

From the selected entrypoint(s), read relevant source files in small, targeted chunks and trace control/data flow that matches the analysis question.

Identify and annotate, each with a one‑line purpose and file:line anchor:

- Entry & orchestration: main/CLI/server/runner; the first function(s) that bind external inputs.
- Config flow: where inputs/flags/env become internal state or configuration objects/structs.
- Core pipeline stages: compute/aggregate/sampling steps relevant to the question.
- Normalization/Scaling chain (when applicable): factor order and owners, e.g.,
  - step/iteration averaging or division
  - geometry/obliquity/solid‑angle
  - attenuation/absorption/parallax
  - constants/time/exposure/flux/fluence
- Sinks/outputs: writers, serializers, final state transitions, ROI/masking applied near the end.

Keep reads focused and under the time budget; prefer anchoring function signatures and key statements over copying code.

---

## Dynamic Call Tracing (optional, scoped)

If time remains and it will help, run a minimal, module‑filtered dynamic trace for a small ROI that exercises the path (e.g., a single item/pixel). Use a profiler/tracer appropriate to the language/runtime. Restrict to `namespace_filter` (or the project’s primary package) to reduce noise. Do not modify production code.

Store the filtered call tree as `callgraph/dynamic.txt`.

---

## Numeric Tap Plan (place taps from the callgraph)

Propose 5–7 numeric taps aligned to the mapped callgraph and the analysis question. Examples:

- Config snapshot: key inputs as they enter the pipeline.
- Pre‑aggregation metric: the quantity before any normalization/weights.
- Per‑factor contributions: e.g., geometry/obliquity/omega; attenuation/absorption; weights.
- Normalization step: where/when division or averaging by steps occurs.
- Final scaling constants: exposure/flux/fluence/time/physical constants.
- Pre‑sink: value just before writing/masking.

Taps MUST read already‑computed values via public APIs or cached intermediates; DO NOT re‑derive algorithms in trace code. Emit the plan to `trace/tap_points.md` with file:line anchors for each tap location.

---

## Constraints & Guardrails

- Evidence‑only: no production hot‑path edits. Use a harness or script if needed.
- Respect Protected Assets: do not delete/rename any documented, protected files; write under `plans/active/<initiative_id>/reports/…` only.
- Keep device/dtype/runtime neutrality (no hidden transfers or precision assumptions); use minimal ROI.
- Reference all findings with `path:line` anchors; avoid copying code blocks.
- Stable key names across traces/taps to enable diff tooling later.

---

## Deliverables (standard format)

Write the following files under `plans/active/<initiative_id>/reports/`:

1) `callchain/static.md`
   - Analysis Question
   - Candidate Entry Points (table: Candidate | Relevance | Confidence | Expected Code Region)
   - Selected Entrypoint(s) and why
   - Config Flow (where, how, keys) | `path:line`
   - Core Pipeline Stages (purpose) | `path:line`
   - Normalization/Scaling Chain (factors, order, owners) | `path:line`
   - Sinks/Outputs (or final state transitions) | `path:line`
   - Callgraph Edge List (`A → B` | why | `path:line` of A,B)
   - Data/Units & Constants (definitions/uses) | `path:line`
   - Device/dtype handling (assumptions) | `path:line`
   - Gaps/Unknowns + Confirmation Plan

2) `callgraph/dynamic.txt` (optional)
   - Module‑filtered call tree for a minimal run; include the command or profiler settings at the top.

3) `trace/tap_points.md`
   - Key | Purpose | Owning Function `path:line` | Expected Units

4) `summary.md`
   - One‑page, question‑oriented narrative of the path, the most likely factor/order to tap first, and the next step to confirm.

5) `env/trace_env.json`
   - Tool/runtime versions, commit SHA, OS, optional device, and any relevant environment variables.

---

## Heuristics (when docs are thin)

If discovery sources are incomplete, apply these patterns and RECORD your assumptions in `static.md`:

- Entrypoints: files/functions named main/cli/server/runner; language‑specific startup idioms.
- Config: modules named config/settings/options/env; constructors that ingest flags/env.
- Pipeline verbs: simulate/process/render/compute/execute/run.
- Normalization/scaling: normalize/scale/weight/steps/average/exposure/flux/fluence/time/omega/absorption/attenuation.
- Sinks: write/save/export/serialize.

---

## Exit Criteria

- Question‑driven static callgraph with `path:line` anchors from entry to sink.
- Clear description of the factor/order most relevant to the analysis question.
- Proposed numeric taps mapped to owning functions with anchors.
- Optional dynamic call tree captured for a minimal run.
- A succinct `summary.md` stating “first taps to collect” and open questions.

---

## Dos / Don’ts

- Do: keep exploration within the time budget; prefer anchoring over copying code.
- Do: keep outputs small, structured, and diff‑friendly.
- Do: tie every recommendation to a concrete file:line anchor or a doc anchor.
- Don’t: modify production code or re‑implement algorithms in trace/harness code.
- Don’t: run full test suites; at most “collect‑only” selectors to validate nodes.

---

## Example Invocation (for supervisor or scripts)

Run this prompt with variables:

```
analysis_question: "Why are normalized values ~1e5 higher than reference in no‑noise runs? Focus: normalization/scaling and final fluence/exposure application."
initiative_id: "cli-scaling"
scope_hints: ["CLI flags", "normalization", "scaling", "fluence"]
roi_hint: "single minimal case"
namespace_filter: "<project primary package>"
time_budget_minutes: 30
```

Then follow the Procedure sections above, producing the Deliverables exactly under `plans/active/cli-scaling/reports/…`.
