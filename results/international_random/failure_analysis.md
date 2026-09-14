# Failure analysis — international_random (gru_multihead seed 0)

## 1. Do errors compound in multi-step prediction?

Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error | 0.12 | 0.22 | 0.29 | 0.35 | 0.39 | 0.45 | 0.48 | 0.51 | 0.50 | 0.65 | 0.70 | 0.67 | 0.00 |
| n | 1000 | 913 | 827 | 736 | 610 | 486 | 351 | 236 | 131 | 23 | 10 | 3 | 1 |

- Of 149 suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is **0.91** (vs overall per-step error 0.41); 91% of them are mostly wrong afterwards (derailment).
- DL-similarity by prefix length: k=1: 0.822 (n=97), k=2: 0.826 (n=98), k=3: 0.870 (n=103), k=4-5: 0.889 (n=219), k=6-10: 0.915 (n=371), k=11-20: 0.924 (n=112)

## 2. Are predictions internally coherent?

- Post-terminal continuation rate (rollout emits another activity after a terminal one): **0.041**
- EOS precision / recall (next-event head): 0.999 / 0.997
- Mean predicted vs true suffix length: 5.10 vs 5.72; valid-termination rate 1.000
- Rollouts with the same activity ≥3× consecutively (loops): 0.000
- Δt: MAE 141.1 h, medAE 17.3 h, 80% interval coverage 0.80 (mean width 386 h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).
- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **0.84**

## 3. What is lost from long histories?

- Next-activity accuracy by prefix length: 1: 0.741 (n=968), 2: 0.687 (n=968), 3: 0.893 (n=964), 4-5: 0.912 (n=1927), 6-10: 0.919 (n=4368), 11-20: 0.968 (n=1453), 21-10000: 1.000 (n=15)
- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: 0.865 (n=7961)
- Accuracy by distance (events) since the most recent informative event: 1-2: 0.880 (n=968), 3-5: 0.962 (n=713), 6-10: 0.932 (n=147), 11+: 1.000 (n=5)
- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: 1-2: 0.255 (n=548), 3-5: 0.010 (n=763), 6-10: 0.001 (n=423), 11+: 0.000 (n=96). A KL that decays toward 0 with distance means the state has forgotten the event.

## 4. Concrete failure cases (3 worst greedy suffixes among true suffixes of length ≥ 3)

### case `declaration 74252` — prefix length 4, DL-sim 0.00
- prefix: Permit SUBMITTED by EMPLOYEE → Permit APPROVED by SUPERVISOR → Start trip → End trip
- true suffix: Permit FINAL_APPROVED by DIRECTOR → Declaration SAVED by EMPLOYEE → Permit REJECTED by MISSING
- greedy suffix: Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled → <EOS>
- at the prefix end, top-3 next: `Declaration FINAL_APPROVED by SUPERVISOR` 0.44, `Declaration SAVED by EMPLOYEE` 0.15, `Permit FINAL_APPROVED by DIRECTOR` 0.13; true next `Permit FINAL_APPROVED by DIRECTOR`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [56.5, 450.8] h, true 616.1 h

### case `declaration 78711` — prefix length 7, DL-sim 0.00
- prefix: Permit SUBMITTED by EMPLOYEE → Permit FINAL_APPROVED by SUPERVISOR → Start trip → End trip → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration APPROVED by BUDGET OWNER
- true suffix: Declaration APPROVED by SUPERVISOR → Declaration REJECTED by DIRECTOR → Declaration REJECTED by EMPLOYEE
- greedy suffix: Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled → <EOS>
- at the prefix end, top-3 next: `Declaration FINAL_APPROVED by SUPERVISOR` 0.99, `Declaration REJECTED by SUPERVISOR` 0.01, `Declaration APPROVED by BUDGET OWNER` 0.00; true next `Declaration APPROVED by SUPERVISOR`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [3.7, 168.8] h, true 158.9 h

### case `declaration 66019` — prefix length 8, DL-sim 0.00
- prefix: Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit APPROVED by BUDGET OWNER → Permit FINAL_APPROVED by SUPERVISOR → Start trip → End trip → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION
- true suffix: Declaration APPROVED by BUDGET OWNER → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE
- greedy suffix: Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled → <EOS>
- at the prefix end, top-3 next: `Declaration FINAL_APPROVED by SUPERVISOR` 0.67, `Declaration APPROVED by BUDGET OWNER` 0.29, `Declaration REJECTED by SUPERVISOR` 0.03; true next `Declaration APPROVED by BUDGET OWNER`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.4, 185.2] h, true 28.4 h
