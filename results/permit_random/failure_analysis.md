# Failure analysis — permit_random (GRU seed 0)

## 1. Do errors compound in multi-step prediction?

Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error | 0.14 | 0.26 | 0.34 | 0.40 | 0.49 | 0.52 | 0.54 | 0.62 | 0.62 | 0.64 | 0.75 | 0.87 | 0.93 | 0.82 | 0.71 |
| n | 500 | 452 | 405 | 358 | 303 | 255 | 213 | 156 | 102 | 56 | 40 | 23 | 14 | 11 | 7 |

- Of 238 suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is **0.93** (vs overall per-step error 0.58); 95% of them are mostly wrong afterwards (derailment).
- DL-similarity by prefix length: k=1: 0.726 (n=34), k=2: 0.664 (n=34), k=3: 0.807 (n=39), k=4-5: 0.790 (n=76), k=6-10: 0.882 (n=187), k=11-20: 0.877 (n=113), k=21-10000: 0.858 (n=17)

## 2. Are predictions internally coherent?

- Post-terminal continuation rate (rollout emits another activity after a terminal one): **0.162**
- EOS precision / recall (next-event head): 0.983 / 0.995
- Mean predicted vs true suffix length: 6.61 vs 7.67; valid-termination rate 0.982
- Rollouts with the same activity ≥3× consecutively (loops): 0.028
- Δt: MAE 136.4 h, medAE 17.6 h, 80% interval coverage 0.82 (mean width 426 h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).
- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **0.80**

## 3. What is lost from long histories?

- Next-activity accuracy by prefix length: 1: 0.751 (n=1061), 2: 0.652 (n=1061), 3: 0.803 (n=1058), 4-5: 0.842 (n=2100), 6-10: 0.884 (n=4293), 11-20: 0.899 (n=2523), 21-10000: 0.796 (n=505)
- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: 0.821 (n=9342)
- Accuracy by distance (events) since the most recent informative event: 1-2: 0.841 (n=1116), 3-5: 0.893 (n=825), 6-10: 0.805 (n=298), 11+: 0.849 (n=199)
- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: 1-2: 0.269 (n=596), 3-5: 0.049 (n=837), 6-10: 0.012 (n=550), 11+: 0.004 (n=446). A KL that decays toward 0 with distance means the state has forgotten the event.

## 4. Concrete failure cases (3 worst greedy suffixes among true suffixes of length ≥ 3)

### case `travel permit 55543` — prefix length 6, DL-sim 0.00
- prefix: Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Start trip → End trip → Permit APPROVED by BUDGET OWNER → Permit FINAL_APPROVED by SUPERVISOR
- true suffix: Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration APPROVED by BUDGET OWNER → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled
- greedy suffix: Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder (truncated)
- at the prefix end, top-3 next: `Send Reminder` 0.61, `Declaration SUBMITTED by EMPLOYEE` 0.34, `Declaration SAVED by EMPLOYEE` 0.02; true next `Declaration SUBMITTED by EMPLOYEE`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [17.8, 2022.6] h, true 41.1 h

### case `travel permit 36508` — prefix length 5, DL-sim 0.00
- prefix: Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit FINAL_APPROVED by SUPERVISOR → Start trip → End trip
- true suffix: Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled
- greedy suffix: Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder (truncated)
- at the prefix end, top-3 next: `Send Reminder` 0.56, `Declaration SUBMITTED by EMPLOYEE` 0.42, `Declaration SAVED by EMPLOYEE` 0.01; true next `Declaration SUBMITTED by EMPLOYEE`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [82.0, 1404.6] h, true 41.2 h

### case `travel permit 22205` — prefix length 5, DL-sim 0.00
- prefix: Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit FINAL_APPROVED by SUPERVISOR → Start trip → End trip
- true suffix: Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled
- greedy suffix: Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder → Send Reminder (truncated)
- at the prefix end, top-3 next: `Send Reminder` 0.88, `Declaration SUBMITTED by EMPLOYEE` 0.11, `Declaration SAVED by EMPLOYEE` 0.01; true next `Declaration SUBMITTED by EMPLOYEE`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [358.0, 1747.1] h, true 33.4 h
