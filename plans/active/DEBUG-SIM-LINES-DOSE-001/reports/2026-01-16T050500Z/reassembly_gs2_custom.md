# Reassembly Limits — gs2_custom

- Scenario: `gs2_custom`
- Snapshot: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/sim_lines_4x_params_snapshot.json`
- Timestamp: 2026-01-16T01:36:16.522465+00:00
- Gridsize: 2
- Group count: 1000
- Neighbor count: 4
- Padded size: 78
- Legacy offset: 4
- Legacy max_position_jitter: 10

## Subset Stats

### Train split
- Raw points: 4000
- Requested groups: 1000
- Actual groups: 1000
- Group limit used: 64
- Combined offset abs max per axis: axis0 abs_max=383.230, axis1 abs_max=197.385
- Required canvas: 831 (max |offset|=383.230018915455)
- Fits padded size (78): False (Δ=-753, ratio=0.09386281588447654)
- Reassembly sums (size=832): padded=5225777.00, required=694371328.00, loss=99.2474088734263%

### Test split
- Raw points: 4000
- Requested groups: 1000
- Actual groups: 1000
- Group limit used: 64
- Combined offset abs max per axis: axis0 abs_max=382.697, axis1 abs_max=383.147
- Required canvas: 831 (max |offset|=383.1471172660616)
- Fits padded size (78): False (Δ=-753, ratio=0.09386281588447654)
- Reassembly sums (size=832): padded=0.00, required=689657216.00, loss=100.0%
