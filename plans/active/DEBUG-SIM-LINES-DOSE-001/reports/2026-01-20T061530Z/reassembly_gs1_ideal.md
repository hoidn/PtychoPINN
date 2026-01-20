# Reassembly Limits — gs1_ideal

- Scenario: `gs1_ideal`
- Snapshot: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json`
- Timestamp: 2026-01-20T05:18:54.833588+00:00
- Gridsize: 1
- Group count: 1000
- Neighbor count: 4
- Padded size: 828
- Legacy offset: 4
- Legacy max_position_jitter: 764

## Subset Stats

### Train split
- Raw points: 1000
- Requested groups: 1000
- Actual groups: 1000
- Group limit used: 64
- Combined offset abs max per axis: axis0 abs_max=381.782, axis1 abs_max=193.236
- Required canvas: 828 (max |offset|=381.78190598668795)
- Fits padded size (828): True (Δ=0, ratio=1.0)
- Reassembly sums (size=828): padded=192568336.00, required=192568336.00, loss=0.0%

### Test split
- Raw points: 1000
- Requested groups: 1000
- Actual groups: 1000
- Group limit used: 64
- Combined offset abs max per axis: axis0 abs_max=381.895, axis1 abs_max=381.835
- Required canvas: 828 (max |offset|=381.8949744624406)
- Fits padded size (828): True (Δ=0, ratio=1.0)
- Reassembly sums (size=828): padded=192554064.00, required=192554064.00, loss=0.0%
