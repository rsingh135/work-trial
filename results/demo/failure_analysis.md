# Failure analysis — demo (GRU seed 0)

## 1. Do errors compound in multi-step prediction?

Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):

| step | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| error | 0.14 | 0.23 | 0.36 | 0.33 | 0.50 |
| n | 150 | 104 | 66 | 6 | 2 |

- Of 17 suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is **1.00** (vs overall per-step error 0.31); 100% of them are mostly wrong afterwards (derailment).
- DL-similarity by prefix length: k=1: 0.860 (n=44), k=2: 0.940 (n=42), k=3: 1.000 (n=42), k=4-5: 0.972 (n=16), k=6-10: 1.000 (n=6)

## 2. Are predictions internally coherent?

- Post-terminal continuation rate (rollout emits another activity after a terminal one): **0.013**
- EOS precision / recall (next-event head): 0.996 / 1.000
- Mean predicted vs true suffix length: 2.15 vs 2.31; valid-termination rate 1.000
- Rollouts with the same activity ≥3× consecutively (loops): 0.000
- Δt: MAE 74.1 h, medAE 15.3 h, 80% interval coverage 0.78 (mean width 106 h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).
- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **0.60**

## 3. What is lost from long histories?

- Next-activity accuracy by prefix length: 1: 0.701 (n=224), 2: 0.951 (n=224), 3: 0.987 (n=224), 4-5: 0.969 (n=292), 6-10: 1.000 (n=51)
- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: 0.887 (n=716)
- Accuracy by distance (events) since the most recent informative event: 1-2: 0.816 (n=49), 3-5: 1.000 (n=26)
- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: 1-2: 0.114 (n=36), 3-5: 0.000 (n=37), 6-10: 0.000 (n=2). A KL that decays toward 0 with distance means the state has forgotten the event.

## 4. Concrete failure cases (3 worst greedy suffixes among true suffixes of length ≥ 3)

### case `declaration 86632` — prefix length 3, DL-sim 0.33
- prefix: Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by PRE_APPROVER → Declaration FINAL_APPROVED by SUPERVISOR
- true suffix: Declaration REJECTED by MISSING → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by PRE_APPROVER → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled
- greedy suffix: Request Payment → Payment Handled → <EOS>
- at the prefix end, top-3 next: `Request Payment` 0.95, `Declaration REJECTED by MISSING` 0.03, `Declaration FINAL_APPROVED by SUPERVISOR` 0.01; true next `Declaration REJECTED by MISSING`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [5.9, 226.3] h, true 118.9 h

### case `declaration 95925` — prefix length 2, DL-sim 0.40
- prefix: Declaration SUBMITTED by EMPLOYEE → Declaration FINAL_APPROVED by SUPERVISOR
- true suffix: Declaration REJECTED by MISSING → Declaration SUBMITTED by EMPLOYEE → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled
- greedy suffix: Request Payment → Payment Handled → <EOS>
- at the prefix end, top-3 next: `Request Payment` 0.94, `Declaration REJECTED by MISSING` 0.02, `Declaration FINAL_APPROVED by SUPERVISOR` 0.02; true next `Declaration REJECTED by MISSING`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [7.7, 220.6] h, true 18.6 h

### case `declaration 89167` — prefix length 2, DL-sim 0.40
- prefix: Declaration SUBMITTED by EMPLOYEE → Declaration FINAL_APPROVED by SUPERVISOR
- true suffix: Declaration REJECTED by MISSING → Declaration SUBMITTED by EMPLOYEE → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled
- greedy suffix: Request Payment → Payment Handled → <EOS>
- at the prefix end, top-3 next: `Request Payment` 0.95, `Declaration REJECTED by MISSING` 0.02, `Declaration FINAL_APPROVED by SUPERVISOR` 0.01; true next `Declaration REJECTED by MISSING`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [7.4, 197.2] h, true 621.9 h
