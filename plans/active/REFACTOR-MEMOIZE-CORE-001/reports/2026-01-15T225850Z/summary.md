### Turn Summary
Phase A inventory confirmed memoize_raw_data is only used by synthetic_helpers and now needs a core home (`ptycho/cache.py`) plus a compatibility shim to keep imports stable.
docs/fix_plan.md, the implementation plan, and input.md now capture the move/shim scope along with the mapped pytest selectors and artifacts hub for Phase B1-B3.
Next: Ralph creates `ptycho/cache.py`, converts the scripts shim, updates synthetic_helpers, and runs the two synthetic helper pytest selectors.
Artifacts: plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T225850Z/
