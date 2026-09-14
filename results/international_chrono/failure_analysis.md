# Failure analysis — international_chrono (gru_multihead seed 0)

## 1. Do errors compound in multi-step prediction?

Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error | 0.11 | 0.19 | 0.24 | 0.31 | 0.37 | 0.44 | 0.51 | 0.57 | 0.60 | 0.62 | 0.50 | 0.50 | 1.00 |
| n | 1000 | 903 | 815 | 708 | 602 | 490 | 354 | 248 | 151 | 48 | 18 | 6 | 1 |

- Of 156 suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is **0.95** (vs overall per-step error 0.46); 95% of them are mostly wrong afterwards (derailment).
- DL-similarity by prefix length: k=1: 0.782 (n=88), k=2: 0.782 (n=82), k=3: 0.874 (n=101), k=4-5: 0.892 (n=189), k=6-10: 0.951 (n=435), k=11-20: 0.973 (n=104), k=21-10000: 1.000 (n=1)

## 2. Are predictions internally coherent?

- Post-terminal continuation rate (rollout emits another activity after a terminal one): **0.036**
- EOS precision / recall (next-event head): 0.998 / 1.000
- Mean predicted vs true suffix length: 5.60 vs 6.23; valid-termination rate 1.000
- Rollouts with the same activity ≥3× consecutively (loops): 0.000
- Δt: MAE 88.0 h, medAE 17.9 h, 80% interval coverage 0.80 (mean width 758 h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).
- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **0.82**

## 3. What is lost from long histories?

- Next-activity accuracy by prefix length: 1: 0.819 (n=968), 2: 0.564 (n=968), 3: 0.856 (n=968), 4-5: 0.915 (n=1936), 6-10: 0.928 (n=4750), 11-20: 0.968 (n=1655), 21-10000: 1.000 (n=21)
- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: 0.864 (n=8521)
- Accuracy by distance (events) since the most recent informative event: 1-2: 0.888 (n=980), 3-5: 0.973 (n=706), 6-10: 0.933 (n=120), 11+: 1.000 (n=6)
- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: 1-2: 0.188 (n=552), 3-5: 0.003 (n=773), 6-10: 0.000 (n=400), 11+: 0.000 (n=87). A KL that decays toward 0 with distance means the state has forgotten the event.

## 4. Concrete failure cases (3 worst greedy suffixes among true suffixes of length ≥ 3)

### case `declaration 58964` — prefix length 6, DL-sim 0.17
- prefix: Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit FINAL_APPROVED by SUPERVISOR → Start trip → End trip → Declaration SUBMITTED by EMPLOYEE
- true suffix: Declaration APPROVED by ADMINISTRATION → Declaration REJECTED by SUPERVISOR → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE
- greedy suffix: Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → <EOS>
- at the prefix end, top-3 next: `Declaration REJECTED by ADMINISTRATION` 0.55, `Declaration APPROVED by ADMINISTRATION` 0.43, `Declaration REJECTED by PRE_APPROVER` 0.01; true next `Declaration APPROVED by ADMINISTRATION`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 1.3] h, true 0.0 h

### case `declaration 58965` — prefix length 10, DL-sim 0.20
- prefix: Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit FINAL_APPROVED by SUPERVISOR → Start trip → End trip → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION
- true suffix: Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE
- greedy suffix: Declaration REJECTED by EMPLOYEE → <EOS>
- at the prefix end, top-3 next: `Declaration REJECTED by EMPLOYEE` 0.99, `Declaration REJECTED by ADMINISTRATION` 0.00, `Permit REJECTED by MISSING` 0.00; true next `Declaration SUBMITTED by EMPLOYEE`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [3.1, 120.6] h, true 40.8 h

### case `declaration 58964` — prefix length 5, DL-sim 0.29
- prefix: Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit FINAL_APPROVED by SUPERVISOR → Start trip → End trip
- true suffix: Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration REJECTED by SUPERVISOR → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE
- greedy suffix: Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → <EOS>
- at the prefix end, top-3 next: `Declaration SUBMITTED by EMPLOYEE` 0.65, `Send Reminder` 0.18, `Declaration SAVED by EMPLOYEE` 0.17; true next `Declaration SUBMITTED by EMPLOYEE`
- first divergence at suffix step 1; Δt 80% interval at prefix end: [54.8, 968.0] h, true 131.0 h
