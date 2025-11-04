# Phase E6 Training Bundle Real Runs — Summary

**Timestamp:** 2025-11-04T20:57:00Z
**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Loop:** 2025-11-06T210500Z
**Focus:** Phase E6 dense/baseline bundle evidence with archive size parity

---

## Acceptance Criteria

**AT-E6:** Phase E6 dense/baseline deterministic evidence & archive parity
- ✅ Archive helper validates `bundle_size_bytes` parity against filesystem
- ✅ `bundle_checksums.txt` emits `sha256  filename  size_bytes` format
- ✅ Deterministic dense (gs2) + baseline (gs1) runs at dose=1000 complete
- ✅ Bundle SHA+size evidence recorded and cross-validated

---

## Implementation Summary

### Code Changes

**File:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/archive_phase_e_outputs.py:72-133`

**Changes:**
1. Added `size_bytes` computation via `bundle_dest.stat().st_size` (line 101)
2. Extended `entries` tuple to include size: `(name, sha256, size_bytes)` (line 102)
3. Added manifest `bundle_size_bytes` parity validation (lines 122-129):
   - Fail-fast on missing field with descriptive error
   - Fail-fast on size mismatch with both values in error message
4. Updated checksum file format to emit size (line 133): `{sha256}  {name}  {size_bytes}\n`

**Rationale:** Implements Phase E6 Do Now requirement for size tracking with fail-fast integrity checks, maintaining existing SHA validation patterns.

---

## Bundle Evidence

| View | Dose | Gridsize | Bundle SHA256 | Size (bytes) | Status |
|------|------|----------|---------------|--------------|--------|
| baseline | 1e+03 | 1 | c52f2fc0d327b5776a22ba0ffaa30c4b10929ba564080062f179f782d4799365 | 8,595,443 | ✅ |
| dense | 1e+03 | 2 | 77b7c219cab6bbf60e05d61e27730432aab6a9f98b164cb1a4f30bf59df12ed8 | 8,615,672 | ✅ |

**Validation:** Archive helper confirmed SHA256 + size parity with manifest.

---

## Artifacts

All under: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T210500Z/phase_e_training_bundle_real_runs_exec/`

- `cli/dose1000_{baseline,dense}.log` — Training stdout
- `green/pytest_training_cli_bundle_size_green.log` — Targeted test
- `data/wts_{baseline_gs1,dense_gs2}.h5.zip` — Archived bundles
- `analysis/bundle_checksums.txt` — SHA256 + size evidence
- `analysis/archive_phase_e.log` — Archive helper stdout

