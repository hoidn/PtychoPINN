# Paper Text Fixes Design (Results/Discussion)

**Goal:** Restructure and tighten the Results and Discussion text for submission‑quality clarity, fixing repetition, TODOs, and section boundaries while preserving the validated scientific claims.

## Scope
- Files: `/home/ollie/Documents/tmp/paper/ptychopinn_2025.tex` (Results + Discussion) and figure captions where needed (e.g., Fig.~\ref{fig:fivepanel}).
- Add a new **Methods** subsection: “Datasets and Evaluation Protocol” near the end of Methods.
- Keep throughput claim **only once** (Results → Computational Performance; predict‑only warm throughput).
- Defer Siemens‑star `8192.png` provenance; keep placeholder per prior decision.
- Metrics regeneration for SIM‑LINES‑4X remains in the submission‑quality plan; text should reference the table without quoting stale numbers.

## Results Structure (target order)
1) Reconstruction Quality  
2) Overlap‑Free Reconstruction  
3) Photon‑Limited Performance  
4) Data Efficiency  
5) Out‑of‑distribution Generalization  
6) Computational Performance

### Results rules
- Results are **observational**; only brief interpretation.  
- Move Siemens‑star train/test split details to Methods.  
- Keep explicit numbers for data‑efficiency trends (SSIM thresholds) where already supported by Fig.~\ref{fig:ssim}.  
- Overlap‑Free: keep Fig.~\ref{fig:recon_2x2} for qualitative context and add Table~\ref{tab:sim_lines_metrics} for gs1 vs gs2 numeric comparison (no numbers quoted).  
- OOD: explicitly contrast PINN vs baseline and note APS→LCLS transfer in Results; update Fig.~\ref{fig:fivepanel} caption to state APS→LCLS condition.  
- Computational Performance: single predict‑only throughput statement (no repetition elsewhere).  

## Discussion Structure (consolidated)
1) Physics‑constrained flexibility & single‑shot  
2) Dose efficiency  
3) Generalization & data efficiency (combine in‑dist overfitting + OOD)  
4) Limitations & extensions  
Plus a brief “Implications for modern light sources” paragraph (qualitative, no throughput).

### Discussion rules
- Move interpretive statements (overfitting, inductive bias, implications) into Discussion.  
- Remove the standalone “Conclusion” subsection inside Discussion (keep main Conclusion only).  
- Resolve TODOs where possible; avoid introducing new TODOs.

## Methods addition
Add **Methods → Datasets and Evaluation Protocol** near the end of Methods:
- Define APS (Velociprobe Siemens‑star), LCLS‑XPP (Run1084), and **fly001‑synthetic** (simulated from APS Siemens‑star reconstructions).
- Specify Siemens‑star train/test split (upper half train, lower half test).
- Note OOD protocol: train on APS, test on LCLS with beamline‑specific forward parameters.

## Non‑goals
- No changes to core physics/model code.
- No figure regeneration beyond previously agreed placeholder/lookup tasks.
- No new metrics until SIM‑LINES‑4X regeneration is complete.
