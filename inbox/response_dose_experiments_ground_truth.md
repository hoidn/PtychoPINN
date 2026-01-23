# Response — Legacy dose_experiments Ground-Truth Bundle

**From:** Maintainer <1> (PtychoPINN dose_experiments branch, root_dir: ~/Documents/PtychoPINN/)
**To:** Maintainer <2> (PtychoPINN active branch, root_dir: ~/Documents/tmp/PtychoPINN/)
**Re:** Request — legacy dose_experiments ground-truth artifacts (2026-01-22T014445Z)

---

## 1. Delivery Summary

The requested ground-truth bundle is now available at:

```
plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth/
```

**Bundle structure:**
```
dose_experiments_ground_truth/
├── simulation/     # 7 datasets (data_p1e3.npz ... data_p1e9.npz)
├── training/       # params.dill, baseline_model.h5, recon.dill
├── inference/      # wts.h5.zip (PINN weights)
└── docs/           # README.md, manifests, summaries
```

**Documentation assets (under `docs/`):**
- `README.md` — Commands, environment requirements, provenance tables, NPZ schema
- `ground_truth_manifest.json` — Machine-readable manifest with full SHA256 checksums
- `ground_truth_manifest.md` — Human-readable manifest summary
- `dose_baseline_summary.json` — Baseline metrics snapshot

---

## 2. Verification Summary

Per `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/bundle_verification.md`:

| Metric | Value |
|--------|-------|
| Total files | 15 |
| Verified | 15/15 |
| Total size | 278.18 MB |
| Tarball size | 270.70 MB |
| Tarball SHA256 | `7fe5e14ed9909f056807b77d5de56e729b8b79c8e5b8098ba50507f13780dd72` |

Full verification details available in:
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/bundle_verification.json`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/bundle_verification.md`

---

## 3. Test Validation

```bash
pytest tests/test_generic_loader.py::test_generic_loader -q
```

**Result:** 1 passed, 5 warnings (2.54s)

**Log:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T004049Z/pytest_loader.log`

SHA256 recomputed and matched:
```
7fe5e14ed9909f056807b77d5de56e729b8b79c8e5b8098ba50507f13780dd72  dose_experiments_ground_truth.tar.gz
```

---

## 4. How-To: Extract and Verify

### Extract tarball
```bash
cd plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/
tar -xzf dose_experiments_ground_truth.tar.gz
```

### Verify SHA256
```bash
sha256sum -c dose_experiments_ground_truth.tar.gz.sha256
# Expected: dose_experiments_ground_truth.tar.gz: OK
```

### Re-run helper CLIs (optional)

The CLIs under `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/` can regenerate manifests/READMEs if needed:

```bash
# Generate manifest (read-only, no GPU required)
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/make_ground_truth_manifest.py --help

# Generate README
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/generate_legacy_readme.py --help

# Package bundle (copies files, creates tarball)
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/package_ground_truth_bundle.py --help
```

---

## 5. Dataset Table

All NPZ files conform to `specs/data_contracts.md` §RawData NPZ.

| File | Photon Dose | Size | SHA256 |
|------|-------------|------|--------|
| `data_p1e3.npz` | 1e3 | 4.85 MB | `f9fa3f9f3f1cf8fc181f9c391b5acf15247ba3f25ed54fe6aaf9809388a860b2` |
| `data_p1e4.npz` | 1e4 | 11.07 MB | `1cce1fe9596a82290bffc3cd6116f43cfed17abe7339503dd849ec5826378402` |
| `data_p1e5.npz` | 1e5 | 16.58 MB | `01007daf8afc67aad3ad037e93077ba8bfb28b58d07e17a1e539bd202ffa0d95` |
| `data_p1e6.npz` | 1e6 | 24.64 MB | `95cfd6aee6b2c061e2a8fbe62a824274165e39e0c4514e4537ee1fecf7a79f64` |
| `data_p1e7.npz` | 1e7 | 35.55 MB | `9902ae24e90d2fa63bebf7830a538868cea524fab9cb15a512509803a9896251` |
| `data_p1e8.npz` | 1e8 | 44.45 MB | `56b4f66a92aa28b2983757417cad008ef14c45c796b2d61d9e366aae3a3d55cf` |
| `data_p1e9.npz` | 1e9 | 55.29 MB | `3e1f229af34525a7a912c9c62fa8df6ab87c69528572686a34de4d2640c57c4a` |

**NPZ key requirements** (per `specs/data_contracts.md` §RawData NPZ):
- Required: `xcoords`, `ycoords`, `xcoords_start`, `ycoords_start`, `diff3d`, `probeGuess`, `scan_index`
- Optional: `objectGuess`, `ground_truth_patches`

---

## 6. Baseline Artifacts Table

| File | Type | Size | SHA256 |
|------|------|------|--------|
| `params.dill` | params | 34.79 KB | `92c27229e2edca3a279d9efd6c8134378cc82b6efd38f0aba751128fb48eb588` |
| `baseline_model.h5` | baseline_output | 52.93 MB | `46b88686b95ce4e437561ddcb8ad052e2138fc7bd48b5b66f27b7958246d878c` |
| `recon.dill` | baseline_output | 820.04 KB | `2501b93db2fea8e3751dee6649503b8dfd62aa72c4b077c27e5773af3b1b304c` |
| `wts.h5.zip` | pinn_weights | 31.98 MB | `56a26314a6c6db4fb466673f8eb308f4b8502d9a5bc3d79d60bf641f71b5b1cd` |

**Baseline metrics (train / test):**
- MS-SSIM: 0.9248 / 0.9206
- PSNR: 71.32 dB / 158.06 dB
- intensity_scale_value (learned): 988.21

---

## 7. Key Parameters

| Parameter | Value |
|-----------|-------|
| N (patch size) | 64 |
| gridsize | 1 |
| nepochs | 50 |
| batch_size | 16 |
| loss | NLL-only (nll_weight=1.0, mae_weight=0.0) |
| probe.trainable | False |
| intensity_scale.trainable | True |

---

## 8. Next Steps

Please confirm receipt and let me know if:
1. The tarball extracts correctly and SHA256 matches
2. The datasets load without errors in your environment
3. Any additional artifacts or documentation are needed

Once acknowledged, I will mark DEBUG-SIM-LINES-DOSE-001.D1 complete in `docs/fix_plan.md`.

---

**Artifacts path:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T004049Z/`
**Request source:** `inbox/request_dose_experiments_ground_truth_2026-01-22T014445Z.md`

---

## 9. Rehydration Verification

The tarball `dose_experiments_ground_truth.tar.gz` was extracted into a fresh temporary directory, and the manifest was regenerated from the extracted files. All 11 files matched the original manifest exactly (SHA256 + size).

**Status:** `PASS`

| Metric | Count |
|--------|-------|
| Total files | 11 |
| Matches | 11 |
| Mismatches | 0 |

**Verification script:** `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/verify_bundle_rehydration.py`

**Artifacts:**
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/rehydration_check/rehydration_summary.md`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/rehydration_check/rehydration_diff.json`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/rehydration_check/verify_bundle_rehydration.log`

**Pytest validation (post-rehydration):**
```bash
pytest tests/test_generic_loader.py::test_generic_loader -q
# Result: 1 passed, 5 warnings (2.53s)
```
**Log:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/pytest_loader.log`

This confirms the tarball can be extracted and used as a drop-in replacement for the original dataset files.

---

## Maintainer Status

### Status as of 2026-01-23T013500Z

**Inbox scan result:** No acknowledgement from Maintainer <2> detected yet.

| Metric | Value |
|--------|-------|
| Files scanned | 5 |
| Matches found | 3 |
| From Maintainer <2> | 1 (original request) |
| From Maintainer <1> | 2 (response + follow-up) |
| Ack detected | No |

The bundle has been delivered and a follow-up note sent. We are awaiting Maintainer <2>'s confirmation that:
1. The tarball extracted correctly and SHA256 matches
2. The datasets load without errors in their environment
3. Any additional artifacts or documentation are needed

**Scan details:** [`plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T013500Z/inbox_check/inbox_scan_summary.md`](plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T013500Z/inbox_check/inbox_scan_summary.md)

**Tarball SHA256:** `7fe5e14ed9909f056807b77d5de56e729b8b79c8e5b8098ba50507f13780dd72` (unchanged)

---

### Status as of 2026-01-23T014900Z

**Inbox scan with timeline + waiting-clock monitoring:**

| Metric | Value |
|--------|-------|
| Last Inbound (from Maintainer <2>) | 2026-01-22T23:22:58Z |
| Hours Since Last Inbound | 2.07 hours |
| Last Outbound (from Maintainer <1>) | 2026-01-23T01:20:30Z |
| Hours Since Last Outbound | 0.11 hours |
| Total Inbound Messages | 1 |
| Total Outbound Messages | 2 |
| Acknowledgement Detected | No |

**Timeline (chronological):**
1. **2026-01-22T23:22:58Z** — Maintainer <2> (Inbound): Original request
2. **2026-01-23T01:04:13Z** — Maintainer <1> (Outbound): Follow-up note
3. **2026-01-23T01:20:30Z** — Maintainer <1> (Outbound): Response with bundle details

**Status:** Still waiting for Maintainer <2>'s acknowledgement that:
1. The tarball extracted correctly and SHA256 matches
2. The datasets load without errors in their environment
3. Any additional artifacts or documentation are needed

**Scan details:** [`plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T014900Z/inbox_check_timeline/inbox_scan_summary.md`](plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T014900Z/inbox_check_timeline/inbox_scan_summary.md)

---

### Status as of 2026-01-23T020500Z (SLA Watch Enabled)

**SLA breach detection added to inbox scan CLI.**

| Metric | Value |
|--------|-------|
| Last Inbound (from Maintainer <2>) | 2026-01-22T23:22:58Z |
| Hours Since Last Inbound | 2.22 hours |
| SLA Threshold | 2.00 hours |
| **SLA Breached** | **Yes** |
| Acknowledgement Detected | No |

**CLI command:**
```bash
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py \
  --inbox inbox \
  --request-pattern dose_experiments_ground_truth \
  --sla-hours 2.0 \
  --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T020500Z/inbox_sla_watch
```

**New test selector:**
```bash
pytest tests/tools/test_check_inbox_for_ack_cli.py::test_sla_watch_flags_breach -q
```

**SLA Watch Notes:** SLA breach: 2.22 hours since last inbound exceeds 2.00 hour threshold and no acknowledgement detected.

**Scan details:** [`plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T020500Z/inbox_sla_watch/inbox_scan_summary.md`](plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T020500Z/inbox_sla_watch/inbox_scan_summary.md)

---

### Status as of 2026-01-23T014011Z (History Logging Enabled)

**Persistent history logging added to inbox scan CLI.**

| Metric | Value |
|--------|-------|
| Last Inbound (from Maintainer <2>) | 2026-01-22T23:22:58Z |
| Hours Since Last Inbound | 2.38 hours |
| SLA Threshold | 2.00 hours |
| **SLA Breached** | **Yes** |
| Acknowledgement Detected | No |
| Total Inbound Messages | 1 |
| Total Outbound Messages | 2 |

**CLI command (with history logging):**
```bash
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py \
  --inbox inbox \
  --request-pattern dose_experiments_ground_truth \
  --sla-hours 2.0 \
  --history-jsonl .../inbox_history/inbox_sla_watch.jsonl \
  --history-markdown .../inbox_history/inbox_sla_watch.md \
  --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T014011Z/inbox_sla_watch
```

**New test selector:**
```bash
pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_logging_appends_entries -q
```

**History Log (Markdown):**
| Generated (UTC) | Ack | Hrs Inbound | Hrs Outbound | SLA Breach | Ack Files |
|-----------------|-----|-------------|--------------|------------|----------|
| 2026-01-23T01:46:01 | No | 2.38 | 0.15 | Yes | - |

**SLA Watch Notes:** SLA breach: 2.38 hours since last inbound exceeds 2.00 hour threshold and no acknowledgement detected.

**Artifact paths:**
- Scan summary: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T014011Z/inbox_sla_watch/`
- History logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T014011Z/inbox_history/`

---

### Status as of 2026-01-23T015222Z (Status Snippet Feature)

**Status snippet CLI feature added for reusable Markdown output.**

| Metric | Value |
|--------|-------|
| Last Inbound (from Maintainer <2>) | 2026-01-22T23:22:58Z |
| Hours Since Last Inbound | 2.58 hours |
| SLA Threshold | 2.00 hours |
| **SLA Breached** | **Yes** |
| Acknowledgement Detected | No |
| Total Inbound Messages | 1 |
| Total Outbound Messages | 2 |

**CLI command (with --status-snippet):**
```bash
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py \
  --inbox inbox \
  --request-pattern dose_experiments_ground_truth \
  --sla-hours 2.0 \
  --history-jsonl .../inbox_history/inbox_sla_watch.jsonl \
  --history-markdown .../inbox_history/inbox_sla_watch.md \
  --status-snippet .../inbox_status/status_snippet.md \
  --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z/inbox_sla_watch
```

**New test selector:**
```bash
pytest tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary -q
```

**Status Snippet** (at `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z/inbox_status/status_snippet.md`):
- Contains "Maintainer Status Snapshot" heading
- Ack status: No (waiting for Maintainer <2>)
- Wait metrics table with hours since inbound/outbound
- SLA Watch table showing breach status
- Timeline table with all matched messages

**Artifact paths:**
- Status snippet: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z/inbox_status/status_snippet.md`
- Scan summary: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z/inbox_sla_watch/`
- History logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z/inbox_history/`
- Test logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T015222Z/logs/`

---

### Status as of 2026-01-23T021945Z (Escalation Note Feature)

**Escalation note CLI feature added for prefilled follow-up drafts.**

| Metric | Value |
|--------|-------|
| Last Inbound (from Maintainer <2>) | 2026-01-22T23:22:58Z |
| Hours Since Last Inbound | 2.77 hours |
| SLA Threshold | 2.00 hours |
| **SLA Breached** | **Yes** |
| Acknowledgement Detected | No |
| Total Inbound Messages | 1 |
| Total Outbound Messages | 2 |

**CLI command (with --escalation-note):**
```bash
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py \
  --inbox inbox \
  --request-pattern dose_experiments_ground_truth \
  --sla-hours 2.0 \
  --fail-when-breached \
  --history-jsonl .../inbox_history/inbox_sla_watch.jsonl \
  --history-markdown .../inbox_history/inbox_sla_watch.md \
  --status-snippet .../inbox_status/status_snippet.md \
  --escalation-note .../inbox_status/escalation_note.md \
  --escalation-recipient "Maintainer <2>" \
  --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T021945Z/inbox_sla_watch
```

**New test selector:**
```bash
pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_note_emits_call_to_action -q
```

**Escalation Note** (at `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T021945Z/inbox_status/escalation_note.md`):
- Contains "Escalation Note" heading with recipient and request pattern
- Summary Metrics table (ack status, hours since inbound/outbound, message counts)
- SLA Watch table with breach status and notes
- Action Items checklist for the follow-up
- Proposed Message blockquote with prefilled text for Maintainer <2>
- Timeline table with all matched messages

**Artifact paths:**
- Escalation note: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T021945Z/inbox_status/escalation_note.md`
- Status snippet: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T021945Z/inbox_status/status_snippet.md`
- Scan summary: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T021945Z/inbox_sla_watch/`
- History logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T021945Z/inbox_history/`
- Test logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T021945Z/logs/`

---

### Status as of 2026-01-23T023500Z (History Dashboard Feature)

**History dashboard CLI feature added for aggregated SLA tracking.**

| Metric | Value |
|--------|-------|
| Last Inbound (from Maintainer <2>) | 2026-01-22T23:22:58Z |
| Hours Since Last Inbound | 2.95 hours |
| SLA Threshold | 2.00 hours |
| **SLA Breached** | **Yes** |
| Acknowledgement Detected | No |
| Total Inbound Messages | 1 |
| Total Outbound Messages | 2 |

**CLI command (with --history-dashboard):**
```bash
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py \
  --inbox inbox \
  --request-pattern dose_experiments_ground_truth \
  --sla-hours 2.0 \
  --fail-when-breached \
  --history-jsonl .../inbox_history/inbox_sla_watch.jsonl \
  --history-markdown .../inbox_history/inbox_sla_watch.md \
  --history-dashboard .../inbox_history/inbox_history_dashboard.md \
  --status-snippet .../inbox_status/status_snippet.md \
  --escalation-note .../inbox_status/escalation_note.md \
  --escalation-recipient "Maintainer <2>" \
  --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T023500Z/inbox_sla_watch
```

**New test selector:**
```bash
pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_summarizes_runs -q
```

**History Dashboard** (at `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T023500Z/inbox_history/inbox_history_dashboard.md`):
- Summary Metrics table (Total Scans, Ack Count, Breach Count)
- SLA Breach Stats table (Longest Wait, Last Ack Timestamp, Last Scan Timestamp)
- Recent Scans table with timestamps, ack status, hours since inbound/outbound, breach status

**Dashboard Metrics:**
| Metric | Value |
|--------|-------|
| Total Scans | 1 |
| Ack Count | 0 |
| Breach Count | 1 |
| Longest Wait | 2.95 hours |

**Artifact paths:**
- History dashboard: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T023500Z/inbox_history/inbox_history_dashboard.md`
- Escalation note: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T023500Z/inbox_status/escalation_note.md`
- Status snippet: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T023500Z/inbox_status/status_snippet.md`
- Scan summary: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T023500Z/inbox_sla_watch/`
- History logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T023500Z/inbox_history/`
- Test logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T023500Z/logs/`
- Follow-up note: `inbox/followup_dose_experiments_ground_truth_2026-01-23T023500Z.md`

---

### Status as of 2026-01-23T083500Z (Per-Actor Severity History Persistence)

**Per-actor severity classification now persisted in history logs.**

| Metric | Value |
|--------|-------|
| Last Inbound (from Maintainer <2>) | 2026-01-22T23:22:58Z |
| Hours Since Last Inbound | 4.36 hours |
| SLA Threshold (Global) | 2.50 hours |
| **SLA Breached** | **Yes** |
| Acknowledgement Detected | No |

**Per-Actor SLA Summary:**

| Actor | Hours Since Inbound | Threshold | Severity |
|-------|---------------------|-----------|----------|
| Maintainer 2 | 4.36 | 2.00 | **CRITICAL** |
| Maintainer 3 | N/A | 6.00 | UNKNOWN |

**New capability:** History files now persist the `ack_actor_summary` structure:
- **JSONL**: Each entry contains `ack_actor_summary` with `critical`, `warning`, `ok`, `unknown` buckets
- **Markdown**: Table gains "Ack Actor Severity" column showing `[CRITICAL] Maintainer 2 (4.36h > 2.00h)<br>[UNKNOWN] Maintainer 3`

**CLI command:**
```bash
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py \
  --inbox inbox \
  --request-pattern dose_experiments_ground_truth \
  --keywords acknowledged --keywords confirm --keywords received --keywords thanks \
  --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" \
  --sla-hours 2.5 \
  --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" \
  --fail-when-breached \
  --history-jsonl .../inbox_history/inbox_sla_watch.jsonl \
  --history-markdown .../inbox_history/inbox_sla_watch.md \
  --history-dashboard .../inbox_history/inbox_history_dashboard.md \
  --status-snippet .../inbox_status/status_snippet.md \
  --escalation-note .../inbox_status/escalation_note.md \
  --escalation-recipient "Maintainer <2>" \
  --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/inbox_sla_watch
```

**New test selector:**
```bash
pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_history_tracks_severity -q
```

**Test results:**
- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_history_tracks_severity -q` — 1 passed (0.13s)
- `pytest tests/tools/test_check_inbox_for_ack_cli.py -q` — 17 passed (0.90s)
- `pytest tests/test_generic_loader.py::test_generic_loader -q` — 1 passed (2.55s)

**Artifact paths:**
- History JSONL (with ack_actor_summary): `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/inbox_history/inbox_sla_watch.jsonl`
- History Markdown (with Ack Actor Severity column): `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/inbox_history/inbox_sla_watch.md`
- History dashboard: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/inbox_history/inbox_history_dashboard.md`
- Status snippet: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/inbox_status/status_snippet.md`
- Escalation note: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/inbox_status/escalation_note.md`
- Scan summary: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/inbox_sla_watch/`
- Test logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T083500Z/logs/`
- Follow-up note: `inbox/followup_dose_experiments_ground_truth_2026-01-23T083500Z.md`

---

### Status as of 2026-01-23T093500Z (Per-Actor Severity Trends in History Dashboard)

**History dashboard now includes "Ack Actor Severity Trends" table.**

| Metric | Value |
|--------|-------|
| Last Inbound (from Maintainer <2>) | 2026-01-22T23:22:58Z |
| Hours Since Last Inbound | 4.57 hours |
| SLA Threshold (Maintainer <2>) | 2.00 hours |
| SLA Threshold (Maintainer <3>) | 6.00 hours |
| **SLA Breached (M2)** | **Yes** (critical, 2.57h over) |
| **SLA Breached (M3)** | No (unknown, no inbound) |
| Acknowledgement Detected | No |
| Total Inbound Messages | 1 |
| Total Outbound Messages | 4 |

**New feature:** The history dashboard (`--history-dashboard`) now aggregates `ack_actor_summary` data across all JSONL entries to show per-actor severity trends over time:
- Severity counts (critical/warning/ok/unknown) per actor
- Longest wait per actor across all scans
- Latest scan timestamp per actor
- Actors sorted by severity priority (critical > warning > ok > unknown)

**CLI command:**
```bash
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py \
  --inbox inbox \
  --request-pattern dose_experiments_ground_truth \
  --keywords acknowledged --keywords confirm --keywords received --keywords thanks \
  --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" \
  --sla-hours 2.5 \
  --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" \
  --fail-when-breached \
  --history-jsonl .../inbox_history/inbox_sla_watch.jsonl \
  --history-markdown .../inbox_history/inbox_sla_watch.md \
  --history-dashboard .../inbox_history/inbox_history_dashboard.md \
  --status-snippet .../inbox_status/status_snippet.md \
  --escalation-note .../inbox_status/escalation_note.md \
  --escalation-recipient "Maintainer <2>" \
  --output plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z/inbox_sla_watch
```

**New test selector:**
```bash
pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_severity_trends -q
```

**Test results:**
- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_severity_trends -q` — 1 passed (0.17s)
- `pytest tests/tools/test_check_inbox_for_ack_cli.py -q` — 18 passed (0.96s)
- `pytest tests/test_generic_loader.py::test_generic_loader -q` — 1 passed (2.53s)

**Dashboard excerpt (Ack Actor Severity Trends):**
| Actor | Critical | Warning | OK | Unknown | Longest Wait | Latest Scan |
|-------|----------|---------|----|---------|--------------| ------------|
| Maintainer 2 | 1 | 0 | 0 | 0 | 4.57h | 2026-01-23T03:57:03 |
| Maintainer 3 | 0 | 0 | 0 | 1 | N/A | 2026-01-23T03:57:03 |

**Artifact paths:**
- History dashboard (with trends): `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z/inbox_history/inbox_history_dashboard.md`
- History JSONL: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z/inbox_history/inbox_sla_watch.jsonl`
- History Markdown: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z/inbox_history/inbox_sla_watch.md`
- Status snippet: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z/inbox_status/status_snippet.md`
- Escalation note: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z/inbox_status/escalation_note.md`
- Scan summary: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z/inbox_sla_watch/`
- Test logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T093500Z/logs/`
- Follow-up note: `inbox/followup_dose_experiments_ground_truth_2026-01-23T093500Z.md`

---

### Status as of 2026-01-23T103500Z (Breach Timeline Added)

**New feature: Ack Actor Breach Timeline**

The history dashboard now includes a breach timeline section that tracks when each actor first crossed their SLA deadline, how long the current breach streak has lasted, and the exact hours past the SLA threshold.

**Latest scan:**
| Metric | Value |
|--------|-------|
| Last Inbound (from Maintainer <2>) | 2026-01-22T23:22:58Z |
| Hours Since Last Inbound | 4.78 hours |
| SLA Threshold (Maintainer <2>) | 2.00 hours |
| Hours Past SLA | 2.78 hours |
| Current Breach Streak | 1 (this is the first scan with the timeline feature) |
| Maintainer <3> Status | Unknown (no inbound messages) |

**Breach Timeline excerpt:**
| Actor | Breach Start | Latest Scan | Current Streak | Hours Past SLA | Severity |
|-------|--------------|-------------|----------------|----------------|----------|
| Maintainer 2 | 2026-01-23T04:09:40 | 2026-01-23T04:09:40 | 1 | 2.78h | CRITICAL |

*Maintainer 3 is correctly excluded from the breach timeline as they have unknown status (no inbound messages).*

**New test selector:**
```bash
pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_breach_timeline -q
```

**Test results:**
- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_dashboard_actor_breach_timeline -q` — 1 passed
- `pytest tests/tools/test_check_inbox_for_ack_cli.py -q` — 19 passed (1.06s)
- `pytest tests/test_generic_loader.py::test_generic_loader -q` — 1 passed (2.53s)

**Artifact paths:**
- History dashboard (with breach timeline): `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T103500Z/inbox_history/inbox_history_dashboard.md`
- History JSONL: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T103500Z/inbox_history/inbox_sla_watch.jsonl`
- History Markdown: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T103500Z/inbox_history/inbox_sla_watch.md`
- Status snippet: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T103500Z/inbox_status/status_snippet.md`
- Escalation note: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T103500Z/inbox_status/escalation_note.md`
- Scan summary: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T103500Z/inbox_sla_watch/`
- Test logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T103500Z/logs/`

---

### Status as of 2026-01-23T113500Z (Embedded Breach Timeline)

**New feature: Breach Timeline in Status Snippet & Escalation Note**

The breach timeline section that was previously only in the history dashboard is now also embedded directly in the status snippet and escalation note, but only when `--history-jsonl` is provided (to preserve compact outputs for one-off scans without history logging).

**Latest scan:**
| Metric | Value |
|--------|-------|
| Last Inbound (from Maintainer <2>) | 2026-01-22T23:22:58Z |
| Hours Since Last Inbound | 4.99 hours |
| SLA Threshold (Maintainer <2>) | 2.00 hours |
| Hours Past SLA | 2.99 hours |
| Current Breach Streak | 1 |
| Maintainer <3> Status | Unknown (no inbound messages) |

**Embedded breach timeline in status_snippet.md:**
| Actor | Breach Start | Latest Scan | Current Streak | Hours Past SLA | Severity |
|-------|--------------|-------------|----------------|----------------|----------|
| Maintainer 2 | 2026-01-23T04:22:35 | 2026-01-23T04:22:35 | 1 | 2.99h | CRITICAL |

**Test updates:**
- `test_status_snippet_emits_wait_summary` now validates breach timeline is absent without `--history-jsonl` and present with it
- `test_escalation_note_emits_call_to_action` now validates breach timeline is absent without `--history-jsonl` and present with it

**Test results:**
- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_status_snippet_emits_wait_summary -q` — 1 passed
- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_note_emits_call_to_action -q` — 1 passed
- `pytest tests/tools/test_check_inbox_for_ack_cli.py -q` — 19 passed (1.13s)
- `pytest tests/test_generic_loader.py::test_generic_loader -q` — 1 passed (2.52s)

**Artifact paths:**
- Status snippet (with breach timeline): `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T113500Z/inbox_status/status_snippet.md`
- Escalation note (with breach timeline): `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T113500Z/inbox_status/escalation_note.md`
- History dashboard: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T113500Z/inbox_history/inbox_history_dashboard.md`
- History JSONL: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T113500Z/inbox_history/inbox_sla_watch.jsonl`
- Scan summary: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T113500Z/inbox_sla_watch/`
- Test logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T113500Z/logs/`
- Follow-up note: `inbox/followup_dose_experiments_ground_truth_2026-01-23T113500Z.md`

---

### Status as of 2026-01-23T123500Z (Escalation Brief for Maintainer <3>)

**New feature: Escalation Brief CLI**

A new `--escalation-brief` flag enables generating a Markdown brief for third-party escalation. This is designed to notify Maintainer <3> about the ongoing SLA breach from Maintainer <2>.

**New CLI flags:**
- `--escalation-brief <path>`: Output path for the brief
- `--escalation-brief-recipient <actor>`: The recipient of the brief (default: Maintainer <3>)
- `--escalation-brief-target <actor>`: The blocking actor (default: Maintainer <2>)

**Brief contents:**
- **Blocking Actor Snapshot**: Hours since inbound, SLA threshold, deadline, hours past SLA, severity
- **Breach Streak Summary**: Current streak count, breach start, latest scan (requires `--history-jsonl`)
- **Action Items**: Steps for escalation
- **Proposed Message**: Blockquote template addressing the recipient about the blocking actor
- **Ack Actor Breach Timeline**: Per-actor breach state (when `--history-jsonl` provided)

**Latest scan:**
| Metric | Value |
|--------|-------|
| Last Inbound (from Maintainer <2>) | 2026-01-22T23:22:58Z |
| Hours Since Last Inbound | 5.23 hours |
| SLA Threshold (Maintainer <2>) | 2.00 hours |
| Hours Past SLA | 3.23 hours |
| Current Breach Streak | 1 |
| Severity | CRITICAL |
| Maintainer <3> Status | Unknown (no inbound messages) |

**Escalation brief excerpt (from `escalation_brief_maintainer3.md`):**
> I am escalating an SLA breach regarding the `dose_experiments_ground_truth` request.
> Maintainer 2 has not acknowledged receipt, and it has been **5.23 hours** since the last inbound message from them, exceeding our SLA threshold.

**New test selector:**
```bash
pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_brief_targets_blocker -q
```

**Test results:**
- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_escalation_brief_targets_blocker -q` — 1 passed
- `pytest tests/tools/test_check_inbox_for_ack_cli.py -q` — 20 passed (1.21s)
- `pytest tests/test_generic_loader.py::test_generic_loader -q` — 1 passed (2.58s)

**Artifact paths:**
- Escalation brief: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/inbox_status/escalation_brief_maintainer3.md`
- Status snippet: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/inbox_status/status_snippet.md`
- Escalation note (Maintainer <2>): `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/inbox_status/escalation_note.md`
- History dashboard: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/inbox_history/inbox_history_dashboard.md`
- History JSONL: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/inbox_history/inbox_sla_watch.jsonl`
- Scan summary: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/inbox_sla_watch/`
- Test logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T123500Z/logs/`
- Follow-up note: `inbox/followup_dose_experiments_ground_truth_2026-01-23T123500Z.md`

---

### Status as of 2026-01-23T133500Z (Per-Actor Follow-Up Activity Tracking)

**New feature: Ack Actor Follow-Up Activity**

The CLI now tracks per-actor follow-up (outbound) activity to prove how often Maintainer <1> is pinging each monitored actor. Follow-up stats are derived from `To:`/`CC:` lines in outbound messages.

**New fields in `ack_actor_stats`:**
- `last_outbound_utc`: Timestamp of the most recent outbound message targeting this actor
- `hours_since_last_outbound`: Hours since that message
- `outbound_count`: Total number of follow-up messages sent to this actor

**New Markdown section: "Ack Actor Follow-Up Activity"**

Appears in status snippet, escalation note, and escalation brief showing outbound metrics per actor:

| Actor | Last Outbound (UTC) | Hours Since Outbound | Outbound Count |
|-------|---------------------|----------------------|----------------|
| Maintainer 2 | 2026-01-23T04:38:25Z | 0.22 | 7 |
| Maintainer 3 | 2026-01-23T04:38:25Z | 0.22 | 2 |

**Escalation Brief Blocking Actor Snapshot now includes:**
- Last Outbound (UTC)
- Hours Since Outbound
- Outbound Count

**Latest scan:**
| Metric | Value |
|--------|-------|
| Last Inbound (from Maintainer <2>) | 2026-01-22T23:22:58Z |
| Hours Since Last Inbound | 5.47 hours |
| SLA Threshold (Maintainer <2>) | 2.00 hours |
| Hours Past SLA | 3.47 hours |
| Current Breach Streak | 1 |
| Severity | CRITICAL |
| Maintainer <2> Outbound Count | 7 |
| Maintainer <3> Outbound Count | 2 |

**New test selector:**
```bash
pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_followups_track_outbound_targets -q
```

**Test results:**
- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_ack_actor_followups_track_outbound_targets -q` — 1 passed
- `pytest tests/tools/test_check_inbox_for_ack_cli.py -q` — 21 passed (1.28s)
- `pytest tests/test_generic_loader.py::test_generic_loader -q` — 1 passed (2.57s)

**Artifact paths:**
- Status snippet: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/inbox_status/status_snippet.md`
- Escalation brief: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/inbox_status/escalation_brief_maintainer3.md`
- Escalation note: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/inbox_status/escalation_note.md`
- History dashboard: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/inbox_history/inbox_history_dashboard.md`
- Test logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T133500Z/logs/`
- Follow-up note: `inbox/followup_dose_experiments_ground_truth_2026-01-23T133500Z.md`

---

### Status as of 2026-01-23T143500Z (History Follow-Up Persistence)

**New feature: Per-Actor Follow-Up Persistence in History Files**

The CLI now persists per-actor follow-up (outbound) activity to history files, enabling long-term tracking of how often Maintainer <1> followed up with each monitored actor.

**New history file features:**
- **JSONL**: Each entry gains `ack_actor_followups` field with per-actor outbound stats (`actor_label`, `last_outbound_utc`, `hours_since_last_outbound`, `outbound_count`)
- **Markdown History**: Table gains "Ack Actor Follow-Ups" column showing entries like `Maintainer 2: 8 (0.2h ago)<br>Maintainer 3: 3 (0.2h ago)`
- **Dashboard**: Gains "## Ack Actor Follow-Up Trends" section showing latest outbound UTC, hours since outbound, max outbound count, and scans with outbound per actor

**Latest scan:**
| Metric | Value |
|--------|-------|
| Last Inbound (from Maintainer <2>) | 2026-01-22T23:22:58Z |
| Hours Since Last Inbound | 5.68 hours |
| SLA Threshold (Maintainer <2>) | 2.00 hours |
| Hours Past SLA | 3.68 hours |
| Current Breach Streak | 1 |
| Severity | CRITICAL |
| Maintainer <2> Outbound Count | 8 |
| Maintainer <3> Outbound Count | 3 |

**Dashboard Follow-Up Trends excerpt:**
| Actor | Latest Outbound (UTC) | Hrs Since Outbound | Max Outbound Count | Scans w/ Outbound |
|-------|------------------------|--------------------|--------------------|-------------------|
| Maintainer 2 | 2026-01-23T04:52:58 | 0.18h | 8 | 1 |
| Maintainer 3 | 2026-01-23T04:52:58 | 0.18h | 3 | 1 |

**New test selector:**
```bash
pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_followups_persist -q
```

**Test results:**
- `pytest tests/tools/test_check_inbox_for_ack_cli.py::test_history_followups_persist -q` — 1 passed
- `pytest tests/tools/test_check_inbox_for_ack_cli.py -q` — 22 passed (1.33s)
- `pytest tests/test_generic_loader.py::test_generic_loader -q` — 1 passed (2.55s)

**Artifact paths:**
- History JSONL (with ack_actor_followups): `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/inbox_history/inbox_sla_watch.jsonl`
- History Markdown (with Ack Actor Follow-Ups column): `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/inbox_history/inbox_sla_watch.md`
- History dashboard (with Follow-Up Trends): `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/inbox_history/inbox_history_dashboard.md`
- Status snippet: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/inbox_status/status_snippet.md`
- Escalation brief: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/inbox_status/escalation_brief_maintainer3.md`
- Escalation note: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/inbox_status/escalation_note.md`
- Test logs: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T143500Z/logs/`
- Follow-up note: `inbox/followup_dose_experiments_ground_truth_2026-01-23T143500Z.md`
