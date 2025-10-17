Summary: Capture a complete PyTorch↔spec configuration field map to unblock Phase B.B1.
Mode: Docs
Focus: INTEGRATE-PYTORCH-001 — Prepare for PyTorch Backend Integration with Ptychodus
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/{config_schema_map.md,scope_notes.md}
Do Now: INTEGRATE-PYTORCH-001 Attempt #4 — Phase B.B1 config schema mapping; build the table and scope notes first (no tests this loop).
If Blocked: Record the blocker and partial findings in scope_notes.md, log it under Attempt #4 in docs/fix_plan.md, and flag it for the next supervisor sync.
Priorities & Rationale:
- Phase B.B1 demands a field-by-field audit before any bridge code lands (plans/active/INTEGRATE-PYTORCH-001/implementation.md:46).
- The spec defines the dataclass contract + KEY_MAPPINGS PyTorch must honor (specs/ptychodus_api_spec.md:61-149).
- CONFIG-001 reminds us params.cfg must be populated via update_legacy_dict; mapping has to cite the dotted keys (docs/findings.md:9).
- PyTorch singleton schemas diverge sharply from TF dataclasses and need clarity before we script failing tests (ptycho_torch/config_params.py:8-128).
How-To Map:
- Use `rg -n "^\s{4}[A-Za-z_]" ptycho_torch/config_params.py` to list singleton fields and capture them verbatim in the mapping table.
- Cross-reference each field against `ptycho/config/config.py` dataclasses and `specs/ptychodus_api_spec.md` tables; note the legacy key from KEY_MAPPINGS when it exists.
- Author `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/config_schema_map.md` with four columns: PyTorch Field | TF Dataclass Field | params.cfg Key | Notes/Gap.
- Capture MVP vs full-parity scope decisions, unresolved questions (Q1/Q2), and any consumer lookups in `scope_notes.md` within the same directory.
- When field usage is unclear, inspect `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py` for how the dataclasses are consumed and reference the relevant lines.
Pitfalls To Avoid:
- Do not touch production code or plan structure during this evidence loop.
- Avoid guessing dotted legacy keys; confirm via KEY_MAPPINGS or spec tables.
- If a PyTorch-only field has no TF counterpart, mark it explicitly rather than deleting it.
- Keep the artifact directory timestamp exactly as provided; no reuse or renaming.
- Skip test commands entirely—Phase B.B2 will cover the failing test.
- Stay scoped to configuration; don’t drift into data pipeline analysis yet.
- Keep notes concise; summarize insights instead of copying large code blocks.
- Remember params.cfg expects dotted keys; reflect that in the mapping.
- Use only documented tooling (python, rg, sed); no external scripts.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/implementation.md:46
- specs/ptychodus_api_spec.md:61
- ptycho/config/config.py:72
- ptycho_torch/config_params.py:8
- docs/findings.md:9
- plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T031500Z/open_questions.md:8
Next Up:
- Phase B.B2: author the failing test once mapping results are reviewed.
