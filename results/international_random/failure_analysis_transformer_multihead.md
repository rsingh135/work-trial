# Failure analysis — international_random (transformer_multihead seed 0)

## 1. Do errors compound in multi-step prediction?

Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error | 0.12 | 0.21 | 0.27 | 0.33 | 0.38 | 0.45 | 0.48 | 0.50 | 0.48 | 0.52 | 0.54 | 0.50 | 0.00 |
| n | 1000 | 913 | 827 | 732 | 617 | 489 | 353 | 236 | 129 | 23 | 13 | 4 | 1 |

- Of 153 suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is **0.91** (vs overall per-step error 0.37); 90% of them are mostly wrong afterwards (derailment).
- DL-similarity by prefix length: k=1: 0.824 (n=97), k=2: 0.830 (n=98), k=3: 0.873 (n=103), k=4-5: 0.892 (n=219), k=6-10: 0.913 (n=371), k=11-20: 0.935 (n=112)

## 2. Are predictions internally coherent?

- Post-terminal continuation rate (rollout emits another activity after a terminal one): **0.043**
- EOS precision / recall (next-event head): 0.995 / 0.999
- Mean predicted vs true suffix length: 5.10 vs 5.72; valid-termination rate 1.000
- Rollouts with the same activity ≥3× consecutively (loops): 0.000
- Δt: MAE 142.8 h, medAE 21.2 h, 80% interval coverage 0.80 (mean width 1197 h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).
- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **0.86**

## 3. What is lost from long histories?

- Next-activity accuracy by prefix length: 1: 0.714 (n=968), 2: 0.689 (n=968), 3: 0.888 (n=964), 4-5: 0.912 (n=1927), 6-10: 0.920 (n=4368), 11-20: 0.969 (n=1453), 21-10000: 1.000 (n=15)
- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: 0.863 (n=7961)
- Accuracy by distance (events) since the most recent informative event: 1-2: 0.875 (n=968), 3-5: 0.965 (n=713), 6-10: 0.918 (n=147), 11+: 1.000 (n=5)
- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: 1-2: 0.073 (n=548), 3-5: 0.034 (n=763), 6-10: 0.043 (n=423), 11+: 0.011 (n=96). A KL that decays toward 0 with distance means the state has forgotten the event.

## 4. Concrete failure cases (3 worst greedy suffixes among true suffixes of length ≥ 3)

### case `declaration 78711` — prefix length 7, DL-sim 0.00
- prefix: Permit SUBMITTED by EMPLOYEE → Permit FINAL_APPROVED by SUPERVISOR → Start trip → End trip → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration APPROVED by BUDGET OWNER
- true suffix: Declaration APPROVED by SUPERVISOR → Declaration REJECTED by DIRECTOR → Declaration REJECTED by EMPLOYEE
- greedy suffix: Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled → <EOS>
- at the prefix end, top-3 next: `Declaration FINAL_APPROVED by SUPERVISOR` 0.97, `Declaration REJECTED by SUPERVISOR` 0.02, `End trip` 0.00; true next `Declaration APPROVED by SUPERVISOR`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [22.4, 236.9] h, true 158.9 h

### case `declaration 65696` — prefix length 7, DL-sim 0.00
- prefix: Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit FINAL_APPROVED by SUPERVISOR → Start trip → End trip → Send Reminder → Declaration SUBMITTED by EMPLOYEE
- true suffix: Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled
- greedy suffix: Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → <EOS>
- at the prefix end, top-3 next: `Declaration REJECTED by ADMINISTRATION` 0.60, `Declaration APPROVED by ADMINISTRATION` 0.40, `Permit APPROVED by ADMINISTRATION` 0.00; true next `Declaration APPROVED by ADMINISTRATION`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 21.8] h, true 0.0 h

### case `declaration 72086` — prefix length 6, DL-sim 0.00
- prefix: Permit SUBMITTED by EMPLOYEE → Permit APPROVED by PRE_APPROVER → Permit FINAL_APPROVED by SUPERVISOR → Start trip → End trip → Declaration SUBMITTED by EMPLOYEE
- true suffix: Declaration APPROVED by PRE_APPROVER → Declaration FINAL_APPROVED by SUPERVISOR → Declaration REJECTED by MISSING
- greedy suffix: Declaration REJECTED by PRE_APPROVER → Declaration REJECTED by EMPLOYEE → <EOS>
- at the prefix end, top-3 next: `Declaration REJECTED by PRE_APPROVER` 0.46, `Declaration APPROVED by PRE_APPROVER` 0.42, `Declaration REJECTED by ADMINISTRATION` 0.04; true next `Declaration APPROVED by PRE_APPROVER`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 80.5] h, true 0.0 h
