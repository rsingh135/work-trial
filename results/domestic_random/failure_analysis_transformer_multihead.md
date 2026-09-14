# Failure analysis — domestic_random (transformer_multihead seed 0)

## 1. Do errors compound in multi-step prediction?

Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):

| step | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| error | 0.14 | 0.25 | 0.36 | 0.39 | 0.37 | 0.54 |
| n | 999 | 768 | 548 | 266 | 41 | 24 |

- Of 127 suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is **0.94** (vs overall per-step error 0.34); 94% of them are mostly wrong afterwards (derailment).
- DL-similarity by prefix length: k=1: 0.860 (n=244), k=2: 0.907 (n=216), k=3: 0.986 (n=213), k=4-5: 0.981 (n=264), k=6-10: 0.979 (n=58), k=11-20: 0.914 (n=5)

## 2. Are predictions internally coherent?

- Post-terminal continuation rate (rollout emits another activity after a terminal one): **0.024**
- EOS precision / recall (next-event head): 0.997 / 0.999
- Mean predicted vs true suffix length: 2.74 vs 3.11; valid-termination rate 1.000
- Rollouts with the same activity ≥3× consecutively (loops): 0.000
- Δt: MAE 41.7 h, medAE 8.5 h, 80% interval coverage 0.80 (mean width 89 h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).
- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **0.79**

## 3. What is lost from long histories?

- Next-activity accuracy by prefix length: 1: 0.737 (n=1558), 2: 0.736 (n=1557), 3: 0.993 (n=1557), 4-5: 0.977 (n=2861), 6-10: 0.971 (n=906), 11-20: 0.900 (n=40), 21-10000: 1.000 (n=4)
- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: 0.864 (n=6087)
- Accuracy by distance (events) since the most recent informative event: 1-2: 0.805 (n=503), 3-5: 0.997 (n=336)
- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: 1-2: 0.219 (n=300), 3-5: 0.002 (n=424), 6-10: 0.001 (n=101), 11+: 0.004 (n=13). A KL that decays toward 0 with distance means the state has forgotten the event.

## 4. Concrete failure cases (3 worst greedy suffixes among true suffixes of length ≥ 3)

### case `declaration 91001` — prefix length 3, DL-sim 0.00
- prefix: Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE
- true suffix: Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE
- greedy suffix:  → <EOS>
- at the prefix end, top-3 next: `<EOS>` 0.99, `Declaration SUBMITTED by EMPLOYEE` 0.01, `Declaration REJECTED by ADMINISTRATION` 0.00; true next `Declaration SUBMITTED by EMPLOYEE`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [-0.0, 7912.2] h, true 25.9 h

### case `declaration 113462` — prefix length 1, DL-sim 0.17
- prefix: Declaration SUBMITTED by EMPLOYEE
- true suffix: Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled
- greedy suffix: Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled → <EOS>
- at the prefix end, top-3 next: `Declaration APPROVED by ADMINISTRATION` 0.80, `Declaration FINAL_APPROVED by SUPERVISOR` 0.09, `Declaration REJECTED by ADMINISTRATION` 0.05; true next `Declaration REJECTED by ADMINISTRATION`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 80.5] h, true 41.8 h

### case `declaration 91001` — prefix length 1, DL-sim 0.20
- prefix: Declaration SUBMITTED by EMPLOYEE
- true suffix: Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE
- greedy suffix: Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → <EOS>
- at the prefix end, top-3 next: `Declaration REJECTED by ADMINISTRATION` 0.47, `Declaration APPROVED by ADMINISTRATION` 0.38, `Declaration FINAL_APPROVED by SUPERVISOR` 0.08; true next `Declaration REJECTED by SUPERVISOR`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 0.3] h, true 0.0 h
