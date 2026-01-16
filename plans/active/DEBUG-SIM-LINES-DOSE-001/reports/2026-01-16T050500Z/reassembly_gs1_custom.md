# Reassembly Limits — gs1_custom

- Scenario: `gs1_custom`
- Snapshot: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json`
- Timestamp: 2026-01-16T01:35:29.934502+00:00
- Gridsize: 1
- Group count: 1000
- Neighbor count: 4
- Padded size: 74
- Legacy offset: 4
- Legacy max_position_jitter: 10

## Subset Stats

### Train split
- Raw points: 1000
- Requested groups: 1000
- Actual groups: 1000
- Group limit used: 64
- Combined offset abs max per axis: axis0 abs_max=381.782, axis1 abs_max=193.236
- Required canvas: 828 (max |offset|=381.78190598668795)
- Fits padded size (74): False (Δ=-754, ratio=0.0893719806763285)
- Reassembly sums (size=828): padded=10484630.00, required=192568336.00, loss=94.5553717616379%

### Test split
- Raw points: 1000
- Requested groups: 1000
- Actual groups: 1000
- Group limit used: 64
- Combined offset abs max per axis: axis0 abs_max=381.895, axis1 abs_max=381.835
- Required canvas: 828 (max |offset|=381.8949744624406)
- Fits padded size (74): False (Δ=-754, ratio=0.0893719806763285)
- Reassembly sums (size=828): padded=0.00, required=192554064.00, loss=100.0%
