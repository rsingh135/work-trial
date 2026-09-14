# Failure analysis — domestic_random (gru_multihead seed 0)

## 1. Do errors compound in multi-step prediction?

Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):

| step | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| error | 0.14 | 0.25 | 0.36 | 0.38 | 0.36 | 0.54 |
| n | 999 | 768 | 548 | 265 | 42 | 24 |

- Of 126 suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is **0.94** (vs overall per-step error 0.34); 94% of them are mostly wrong afterwards (derailment).
- DL-similarity by prefix length: k=1: 0.860 (n=244), k=2: 0.905 (n=216), k=3: 0.987 (n=213), k=4-5: 0.983 (n=264), k=6-10: 0.979 (n=58), k=11-20: 0.914 (n=5)

## 2. Are predictions internally coherent?

- Post-terminal continuation rate (rollout emits another activity after a terminal one): **0.024**
- EOS precision / recall (next-event head): 0.997 / 0.999
- Mean predicted vs true suffix length: 2.74 vs 3.11; valid-termination rate 1.000
- Rollouts with the same activity ≥3× consecutively (loops): 0.000
- Δt: MAE 39.6 h, medAE 2.7 h, 80% interval coverage 0.80 (mean width 77 h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).
- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **0.87**

## 3. What is lost from long histories?

- Next-activity accuracy by prefix length: 1: 0.738 (n=1558), 2: 0.734 (n=1557), 3: 0.994 (n=1557), 4-5: 0.978 (n=2861), 6-10: 0.971 (n=906), 11-20: 0.900 (n=40), 21-10000: 1.000 (n=4)
- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: 0.864 (n=6087)
- Accuracy by distance (events) since the most recent informative event: 1-2: 0.811 (n=503), 3-5: 0.997 (n=336)
- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: 1-2: 0.143 (n=300), 3-5: 0.001 (n=424), 6-10: 0.000 (n=101), 11+: 0.000 (n=13). A KL that decays toward 0 with distance means the state has forgotten the event.

## 4. Concrete failure cases (3 worst greedy suffixes among true suffixes of length ≥ 3)

### case `declaration 91001` — prefix length 3, DL-sim 0.00
- prefix: Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE
- true suffix: Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE
- greedy suffix:  → <EOS>
- at the prefix end, top-3 next: `<EOS>` 0.98, `Declaration SUBMITTED by EMPLOYEE` 0.02, `Declaration REJECTED by EMPLOYEE` 0.00; true next `Declaration SUBMITTED by EMPLOYEE`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [-0.0, 114.4] h, true 25.9 h

### case `declaration 113462` — prefix length 1, DL-sim 0.17
- prefix: Declaration SUBMITTED by EMPLOYEE
- true suffix: Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled
- greedy suffix: Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled → <EOS>
- at the prefix end, top-3 next: `Declaration APPROVED by ADMINISTRATION` 0.67, `Declaration REJECTED by ADMINISTRATION` 0.15, `Declaration FINAL_APPROVED by SUPERVISOR` 0.12; true next `Declaration REJECTED by ADMINISTRATION`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 55.9] h, true 41.8 h

### case `declaration 91001` — prefix length 1, DL-sim 0.20
- prefix: Declaration SUBMITTED by EMPLOYEE
- true suffix: Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE
- greedy suffix: Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → <EOS>
- at the prefix end, top-3 next: `Declaration REJECTED by ADMINISTRATION` 0.79, `Declaration APPROVED by ADMINISTRATION` 0.09, `Declaration REJECTED by PRE_APPROVER` 0.04; true next `Declaration REJECTED by SUPERVISOR`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 2.3] h, true 0.0 h
