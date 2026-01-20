# Reassembly Limits — gs2_ideal

- Scenario: `gs2_ideal`
- Snapshot: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json`
- Timestamp: 2026-01-20T06:25:20.466724+00:00
- Gridsize: 2
- Group count: 1000
- Neighbor count: 4
- Padded size: 826
- Legacy offset: 4
- Legacy max_position_jitter: 758

## Subset Stats

### Train split
- Raw points: 4000
- Requested groups: 1000
- Actual groups: 1000
- Group limit used: 64
- Combined offset abs max per axis: axis0 abs_max=383.230, axis1 abs_max=197.385
- Required canvas: 824 (max |offset|=383.230018915455)
- Fits padded size (826): True (Δ=0, ratio=1.0)
- Reassembly sums (size=824): padded=694371392.00, required=694371392.00, loss=0.0%

### Test split
- Raw points: 4000
- Requested groups: 1000
- Actual groups: 1000
- Group limit used: 64
- Combined offset abs max per axis: axis0 abs_max=382.697, axis1 abs_max=383.147
- Required canvas: 826 (max |offset|=383.1471172660616)
- Fits padded size (826): True (Δ=0, ratio=1.0)
- Reassembly sums (size=826): padded=689582208.00, required=689582208.00, loss=0.0%
