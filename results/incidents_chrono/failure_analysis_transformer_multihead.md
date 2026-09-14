# Failure analysis — incidents_chrono (transformer_multihead seed 0)

## 1. Do errors compound in multi-step prediction?

Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error | 0.26 | 0.52 | 0.63 | 0.72 | 0.73 | 0.73 | 0.70 | 0.78 | 0.77 | 0.79 | 0.78 | 0.64 | 0.68 | 0.56 | 0.79 |
| n | 1000 | 820 | 647 | 465 | 334 | 230 | 176 | 129 | 94 | 67 | 40 | 25 | 19 | 16 | 14 |

- Of 363 suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is **0.89** (vs overall per-step error 0.67); 92% of them are mostly wrong afterwards (derailment).
- DL-similarity by prefix length: k=1: 0.384 (n=165), k=2: 0.415 (n=137), k=3: 0.606 (n=160), k=4-5: 0.599 (n=198), k=6-10: 0.657 (n=237), k=11-20: 0.572 (n=94), k=21-10000: 0.597 (n=9)

## 2. Are predictions internally coherent?

- Post-terminal continuation rate (rollout emits another activity after a terminal one): **0.000**
- EOS precision / recall (next-event head): 0.995 / 1.000
- Mean predicted vs true suffix length: 12.42 vs 4.95; valid-termination rate 0.880
- Rollouts with the same activity ≥3× consecutively (loops): 0.000
- Δt: MAE 14.2 h, medAE 0.1 h, 80% interval coverage 0.81 (mean width 66 h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).
- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **0.69**

## 3. What is lost from long histories?

- Next-activity accuracy by prefix length: 1: 0.977 (n=1133), 2: 0.434 (n=1129), 3: 0.832 (n=708), 4-5: 0.696 (n=1191), 6-10: 0.768 (n=1369), 11-20: 0.743 (n=509), 21-10000: 0.704 (n=54)
- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: 0.733 (n=2777)
- Accuracy by distance (events) since the most recent informative event: 1-2: 0.692 (n=2235), 3-5: 0.573 (n=398), 6-10: 0.677 (n=31)
- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: 1-2: 0.330 (n=568), 3-5: 0.030 (n=475), 6-10: 0.024 (n=295), 11+: 0.023 (n=142). A KL that decays toward 0 with distance means the state has forgotten the event.

## 4. Concrete failure cases (3 worst greedy suffixes among true suffixes of length ≥ 3)

### case `1-740595511` — prefix length 2, DL-sim 0.00
- prefix: Accepted|In Progress → Accepted|In Progress
- true suffix: Accepted|Wait → Completed|Resolved → Completed|Closed
- greedy suffix: Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Assigned → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Assigned → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Assigned → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress (truncated)
- at the prefix end, top-3 next: `Queued|Awaiting Assignment` 0.57, `Completed|In Call` 0.15, `Accepted|Wait - User` 0.14; true next `Accepted|Wait`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 0.7] h, true 0.1 h

### case `1-740651428` — prefix length 5, DL-sim 0.00
- prefix: Accepted|In Progress → Accepted|In Progress → Queued|Awaiting Assignment → Queued|Awaiting Assignment → Accepted|In Progress
- true suffix: Accepted|Wait - User → Completed|Resolved → Completed|Closed
- greedy suffix: Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Assigned → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Assigned → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Assigned → Accepted|In Progress (truncated)
- at the prefix end, top-3 next: `Queued|Awaiting Assignment` 0.60, `Accepted|Assigned` 0.15, `Accepted|Wait - User` 0.13; true next `Accepted|Wait - User`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 0.8] h, true 0.2 h

### case `1-740489995` — prefix length 3, DL-sim 0.02
- prefix: Accepted|In Progress → Accepted|In Progress → Queued|Awaiting Assignment
- true suffix: Accepted|In Progress → Completed|Resolved → Completed|Closed
- greedy suffix: Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Assigned → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Assigned → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Assigned → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Assigned → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment (truncated)
- at the prefix end, top-3 next: `Accepted|In Progress` 0.93, `Queued|Awaiting Assignment` 0.07, `Accepted|Wait - User` 0.00; true next `Accepted|In Progress`
- first divergence at suffix step 1; Δt 80% interval at prefix end: [0.0, 152.1] h, true 0.5 h
