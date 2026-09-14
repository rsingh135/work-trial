# Failure analysis — domestic_chrono_strict (gru_multihead seed 0)

## 1. Do errors compound in multi-step prediction?

Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):

| step | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| error | 0.10 | 0.25 | 0.34 | 0.41 | 0.42 | 0.41 |
| n | 1000 | 793 | 593 | 282 | 40 | 17 |

- Of 116 suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is **1.00** (vs overall per-step error 0.32); 100% of them are mostly wrong afterwards (derailment).
- DL-similarity by prefix length: k=1: 0.902 (n=220), k=2: 0.894 (n=224), k=3: 0.989 (n=219), k=4-5: 0.982 (n=287), k=6-10: 0.980 (n=47), k=11-20: 0.850 (n=3)

## 2. Are predictions internally coherent?

- Post-terminal continuation rate (rollout emits another activity after a terminal one): **0.019**
- EOS precision / recall (next-event head): 0.999 / 1.000
- Mean predicted vs true suffix length: 2.76 vs 3.13; valid-termination rate 1.000
- Rollouts with the same activity ≥3× consecutively (loops): 0.000
- Δt: MAE 38.2 h, medAE 3.2 h, 80% interval coverage 0.79 (mean width 79 h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).
- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **0.86**

## 3. What is lost from long histories?

- Next-activity accuracy by prefix length: 1: 0.931 (n=1558), 2: 0.639 (n=1555), 3: 0.995 (n=1553), 4-5: 0.985 (n=3013), 6-10: 0.977 (n=1046), 11-20: 0.926 (n=27)
- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: 0.894 (n=6399)
- Accuracy by distance (events) since the most recent informative event: 1-2: 0.851 (n=464), 3-5: 0.997 (n=338)
- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: 1-2: 0.523 (n=284), 3-5: 0.001 (n=422), 6-10: 0.000 (n=88), 11+: 0.000 (n=8). A KL that decays toward 0 with distance means the state has forgotten the event.

## 4. Concrete failure cases (3 worst greedy suffixes among true suffixes of length ≥ 3)

### case `declaration 142380` — prefix length 2, DL-sim 0.25
- prefix: Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION
- true suffix: Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE
- greedy suffix: Declaration REJECTED by EMPLOYEE → <EOS>
- at the prefix end, top-3 next: `Declaration REJECTED by EMPLOYEE` 0.97, `Declaration REJECTED by MISSING` 0.02, `Declaration SUBMITTED by EMPLOYEE` 0.00; true next `Declaration REJECTED by EMPLOYEE`
- first divergence at suffix step 1; Δt 80% interval at prefix end: [11.3, 44.1] h, true 22.0 h

### case `declaration 138767` — prefix length 2, DL-sim 0.38
- prefix: Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION
- true suffix: Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration APPROVED by BUDGET OWNER → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled
- greedy suffix: Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled → <EOS>
- at the prefix end, top-3 next: `Declaration FINAL_APPROVED by SUPERVISOR` 0.56, `Declaration APPROVED by BUDGET OWNER` 0.42, `Declaration REJECTED by SUPERVISOR` 0.01; true next `Declaration REJECTED by SUPERVISOR`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.6, 82.5] h, true 21.3 h

### case `declaration 137798` — prefix length 2, DL-sim 0.38
- prefix: Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION
- true suffix: Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration APPROVED by BUDGET OWNER → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled
- greedy suffix: Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled → <EOS>
- at the prefix end, top-3 next: `Declaration FINAL_APPROVED by SUPERVISOR` 0.63, `Declaration APPROVED by BUDGET OWNER` 0.34, `Declaration REJECTED by SUPERVISOR` 0.01; true next `Declaration REJECTED by SUPERVISOR`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [11.8, 64.4] h, true 20.0 h
