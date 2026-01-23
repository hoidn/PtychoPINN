# Follow-Up — Legacy dose_experiments Ground-Truth Bundle

**From:** Maintainer <1> (PtychoPINN dose_experiments branch, root_dir: ~/Documents/PtychoPINN/)
**To:** Maintainer <2> (PtychoPINN active branch, root_dir: ~/Documents/tmp/PtychoPINN/)
**Re:** Follow-up on delivered ground-truth bundle (DEBUG-SIM-LINES-DOSE-001)
**Date:** 2026-01-23T01:19Z

---

## Summary

This is a follow-up regarding the legacy `dose_experiments` ground-truth bundle delivered per your request at `inbox/request_dose_experiments_ground_truth_2026-01-22T014445Z.md`.

The bundle is complete and verified. Engineering work is finished; I am awaiting your acknowledgement to close out this initiative.

---

## Delivery Location

**Bundle root:**
```
plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth/
```

**Tarball:**
```
plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth.tar.gz
```

**Tarball SHA256:**
```
7fe5e14ed9909f056807b77d5de56e729b8b79c8e5b8098ba50507f13780dd72
```

---

## Verification Evidence

1. **Bundle verification:** All 15 files verified via SHA256 checksums
   - Log: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/bundle_verification.md`

2. **Tarball rehydration:** Extracted and re-verified — all 11 data files match original manifest
   - Summary: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/rehydration_check/rehydration_summary.md`
   - Diff log: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/rehydration_check/rehydration_diff.json`

3. **Pytest loader gate:** Latest run confirms environment is healthy
   - Log: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T011900Z/pytest_loader.log`

---

## Contents Recap

| Folder | Contents |
|--------|----------|
| `simulation/` | 7 datasets (`data_p1e3.npz` through `data_p1e9.npz`) |
| `training/` | `params.dill`, `baseline_model.h5`, `recon.dill` |
| `inference/` | `wts.h5.zip` (PINN weights) |
| `docs/` | README.md, manifests, baseline summary |

**Key parameters:** N=64, gridsize=1, nepochs=50, NLL-only loss

---

## Request for Acknowledgement

Please confirm:
1. The tarball extracts correctly and SHA256 matches
2. The datasets load in your environment
3. Any additional artifacts or documentation needed

Once you acknowledge, I will mark `DEBUG-SIM-LINES-DOSE-001` complete in `docs/fix_plan.md`.

---

## References

- **Original response:** `inbox/response_dose_experiments_ground_truth.md`
- **Request source:** `inbox/request_dose_experiments_ground_truth_2026-01-22T014445Z.md`
- **Implementation plan:** `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md`
- **Fix plan entry:** `docs/fix_plan.md` §DEBUG-SIM-LINES-DOSE-001

---

*Awaiting your response.*
