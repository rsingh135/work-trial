# Failure analysis — international_chrono (transformer_multihead seed 0)

## 1. Do errors compound in multi-step prediction?

Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| error | 0.09 | 0.18 | 0.22 | 0.30 | 0.35 | 0.41 | 0.46 | 0.52 | 0.55 | 0.61 | 1.00 |
| n | 1000 | 903 | 815 | 708 | 604 | 488 | 353 | 248 | 140 | 18 | 2 |

- Of 144 suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is **0.93** (vs overall per-step error 0.43); 92% of them are mostly wrong afterwards (derailment).
- DL-similarity by prefix length: k=1: 0.805 (n=88), k=2: 0.803 (n=82), k=3: 0.866 (n=101), k=4-5: 0.879 (n=189), k=6-10: 0.950 (n=435), k=11-20: 0.979 (n=104), k=21-10000: 1.000 (n=1)

## 2. Are predictions internally coherent?

- Post-terminal continuation rate (rollout emits another activity after a terminal one): **0.034**
- EOS precision / recall (next-event head): 0.998 / 1.000
- Mean predicted vs true suffix length: 5.47 vs 6.23; valid-termination rate 1.000
- Rollouts with the same activity ≥3× consecutively (loops): 0.000
- Δt: MAE 99.7 h, medAE 21.1 h, 80% interval coverage 0.77 (mean width 439 h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).
- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **0.80**

## 3. What is lost from long histories?

- Next-activity accuracy by prefix length: 1: 0.866 (n=968), 2: 0.605 (n=968), 3: 0.857 (n=968), 4-5: 0.915 (n=1936), 6-10: 0.929 (n=4750), 11-20: 0.972 (n=1655), 21-10000: 1.000 (n=21)
- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: 0.875 (n=8521)
- Accuracy by distance (events) since the most recent informative event: 1-2: 0.900 (n=980), 3-5: 0.969 (n=706), 6-10: 0.925 (n=120), 11+: 1.000 (n=6)
- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: 1-2: 0.025 (n=552), 3-5: 0.023 (n=773), 6-10: 0.014 (n=400), 11+: 0.012 (n=87). A KL that decays toward 0 with distance means the state has forgotten the event.

## 4. Concrete failure cases (3 worst greedy suffixes among true suffixes of length ≥ 3)

### case `declaration 58964` — prefix length 6, DL-sim 0.17
- prefix: Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit FINAL_APPROVED by SUPERVISOR → Start trip → End trip → Declaration SUBMITTED by EMPLOYEE
- true suffix: Declaration APPROVED by ADMINISTRATION → Declaration REJECTED by SUPERVISOR → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE
- greedy suffix: Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → <EOS>
- at the prefix end, top-3 next: `Declaration REJECTED by ADMINISTRATION` 0.63, `Declaration APPROVED by ADMINISTRATION` 0.37, `End trip` 0.00; true next `Declaration APPROVED by ADMINISTRATION`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 76.7] h, true 0.0 h

### case `declaration 58965` — prefix length 10, DL-sim 0.20
- prefix: Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit FINAL_APPROVED by SUPERVISOR → Start trip → End trip → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION
- true suffix: Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE
- greedy suffix: Declaration REJECTED by EMPLOYEE → <EOS>
- at the prefix end, top-3 next: `Declaration REJECTED by EMPLOYEE` 1.00, `Declaration SUBMITTED by EMPLOYEE` 0.00, `End trip` 0.00; true next `Declaration SUBMITTED by EMPLOYEE`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [8.9, 225.4] h, true 40.8 h

### case `declaration 55836` — prefix length 7, DL-sim 0.27
- prefix: Start trip → End trip → Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit APPROVED by BUDGET OWNER → Permit FINAL_APPROVED by SUPERVISOR → Declaration SUBMITTED by EMPLOYEE
- true suffix: Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration APPROVED by BUDGET OWNER → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled
- greedy suffix: Declaration APPROVED by ADMINISTRATION → Declaration APPROVED by BUDGET OWNER → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled → End trip → Payment Handled → <EOS>
- at the prefix end, top-3 next: `Declaration APPROVED by ADMINISTRATION` 0.76, `Declaration REJECTED by ADMINISTRATION` 0.23, `Start trip` 0.00; true next `Declaration REJECTED by ADMINISTRATION`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 102.9] h, true 0.0 h
