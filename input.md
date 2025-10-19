Summary: Implement the PyTorch fixture generator and turn the new regression tests GREEN.
Mode: TDD
Focus: [TEST-PYTORCH-001] Author PyTorch integration workflow regression — Phase B2 fixture generator TDD
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_fixture_pytorch_integration.py -vv
Artifacts: plans/active/TEST-PYTORCH-001/reports/2025-10-19T225900Z/phase_b_fixture/{fixture_generation.log,pytest_fixture_green.log,summary.md}

Do Now:
1. TEST-PYTORCH-001 B2.C @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — Implement `generate_fixture()` in `scripts/tools/make_pytorch_integration_fixture.py`, emit `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz` + `.json` metadata via design §4, and capture CLI output with `tee plans/active/TEST-PYTORCH-001/reports/2025-10-19T225900Z/phase_b_fixture/fixture_generation.log`; tests: none.
2. TEST-PYTORCH-001 B2.C @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — Run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_fixture_pytorch_integration.py -vv | tee plans/active/TEST-PYTORCH-001/reports/2025-10-19T225900Z/phase_b_fixture/pytest_fixture_green.log` to confirm GREEN, then update plan rows B2.C to `[x]`; tests: pytest tests/torch/test_fixture_pytorch_integration.py -vv.
3. TEST-PYTORCH-001 B2.D @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — Author `fixture_notes.md` + `summary.md` under the new artifact hub capturing checksum, runtime delta, regeneration command, and append docs/fix_plan Attempt summarizing B2.C/D completion; tests: none.

If Blocked: Store intermediate findings (partial generator, failing selector output) under the 2025-10-19T225900Z hub, leave B2.C/B2.D `[P]`, and record blocker in docs/fix_plan Attempts with error text plus follow-up plan.

Priorities & Rationale:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md B2.C/B2.D — Only remaining checklist items before Phase B can close.
- specs/data_contracts.md §1 — Canonical NPZ contract drives dtype/shape enforcement (float32 diffraction, complex64 object/probe).
- docs/findings.md (DATA-001, FORMAT-001) — Fixtures must remove legacy `(H,W,N)` ordering and ensure normalization.
- generator_design.md §4–§6 — Provides authoritative pseudocode, CLI contract, and validation matrix for implementation.
- docs/workflows/pytorch.md §§4–8 — Confirms RawData/PyTorch loader expectations and CONFIG-001 sequencing.

How-To Map:
- Implement `generate_fixture()` exactly per design pseudocode: transpose `(H,W,N)→(N,H,W)`, slice first 64 positions, downcast dtypes, preserve optional coord keys, and save compressed NPZ.
- Use helper functions for checksum + metadata (`hashlib.sha256`, `datetime.now(timezone.utc)`); ensure metadata records commit SHA via `git rev-parse HEAD`.
- Generate fixture via `python scripts/tools/make_pytorch_integration_fixture.py --source datasets/Run1084_recon3_postPC_shrunk_3.npz --output tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --subset-size 64 --metadata-out tests/fixtures/pytorch_integration/minimal_dataset_v1.json`.
- After CLI run, compute `sha256sum tests/fixtures/pytorch_integration/minimal_dataset_v1.npz` and cross-check against metadata before logging results in `fixture_notes.md`.
- Targeted pytest command (GREEN step): `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_fixture_pytorch_integration.py -vv`.
- Update plan tables (`plan.md` B2.C/B2.D, implementation.md B2 row) and append docs/fix_plan Attempt with artifact links (`fixture_generation.log`, `pytest_fixture_green.log`, `fixture_notes.md`, generated fixture checksum).
- Capture `summary.md` describing runtime impact (<45s goal), fixture location, and next steps for Phase B3.

Pitfalls To Avoid:
- Do not leave generator producing 1087 samples; enforce deterministic first-64 subset or update acceptance criteria.
- Keep diffraction normalized (no photon scaling) and validate max < 10.0 per test expectation.
- Ensure metadata JSON includes checksum and `subset_size`; missing fields will fail tests.
- Avoid loading torch in generator; stick to numpy/json/hashlib so CLI stays lightweight.
- Do not commit binary fixtures outside `tests/fixtures/pytorch_integration/`; keep repo root clean.
- When running pytest, ensure the newly generated fixture is committed so future loops can reuse it.
- Do not change tests to skip—make them pass via fixture generation.
- Maintain ASCII encoding in notes; no fancy formatting outside markdown basics.
- Preserve red log under original timestamp; store new GREEN log only under 2025-10-19T225900Z.
- Remember to update docs/fix_plan and plan tables before ending loop to keep ledger authoritative.

Pointers:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T215300Z/phase_b_fixture/fixture_scope.md
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T220500Z/phase_b_fixture/generator_design.md
- specs/data_contracts.md#L1
- docs/workflows/pytorch.md#L1
- docs/findings.md:12

Next Up: 1. TEST-PYTORCH-001 B3.A — point integration regression at the new fixture once generator + documentation are complete.
