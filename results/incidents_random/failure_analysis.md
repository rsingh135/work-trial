# Failure analysis — incidents_random (GRU seed 0)

## 1. Do errors compound in multi-step prediction?

Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error | 0.30 | 0.50 | 0.56 | 0.67 | 0.70 | 0.74 | 0.67 | 0.74 | 0.73 | 0.69 | 0.72 | 0.74 | 0.63 | 0.66 | 0.65 |
| n | 998 | 900 | 720 | 542 | 392 | 279 | 214 | 180 | 159 | 133 | 118 | 107 | 91 | 73 | 65 |

- Of 451 suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is **0.86** (vs overall per-step error 0.65); 92% of them are mostly wrong afterwards (derailment).
- DL-similarity by prefix length: k=1: 0.420 (n=100), k=2: 0.362 (n=123), k=3: 0.427 (n=102), k=4-5: 0.492 (n=174), k=6-10: 0.488 (n=263), k=11-20: 0.459 (n=171), k=21-10000: 0.491 (n=67)

## 2. Are predictions internally coherent?

- Post-terminal continuation rate (rollout emits another activity after a terminal one): **0.000**
- EOS precision / recall (next-event head): 0.978 / 1.000
- Mean predicted vs true suffix length: 19.84 vs 8.04; valid-termination rate 0.655
- Rollouts with the same activity ≥3× consecutively (loops): 0.000
- Δt: MAE 26.9 h, medAE 0.2 h, 80% interval coverage 0.82 (mean width 62 h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).
- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **0.67**

## 3. What is lost from long histories?

- Next-activity accuracy by prefix length: 1: 0.984 (n=1134), 2: 0.574 (n=1133), 3: 0.863 (n=874), 4-5: 0.725 (n=1603), 6-10: 0.740 (n=2415), 11-20: 0.723 (n=1596), 21-10000: 0.701 (n=670)
- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: 0.804 (n=2934)
- Accuracy by distance (events) since the most recent informative event: 1-2: 0.706 (n=4640), 3-5: 0.602 (n=918), 6-10: 0.676 (n=71), 11+: 0.692 (n=13)
- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: 1-2: 0.196 (n=572), 3-5: 0.005 (n=597), 6-10: 0.002 (n=537), 11+: 0.001 (n=687). A KL that decays toward 0 with distance means the state has forgotten the event.

## 4. Concrete failure cases (3 worst greedy suffixes among true suffixes of length ≥ 3)

### case `1-737380876` — prefix length 8, DL-sim 0.00
- prefix: Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Assigned → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress
- true suffix: Accepted|Wait - Implementation → Completed|Resolved → Completed|Closed
- greedy suffix: Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress (truncated)
- at the prefix end, top-3 next: `Queued|Awaiting Assignment` 0.46, `Completed|Resolved` 0.19, `Accepted|Assigned` 0.13; true next `Accepted|Wait - Implementation`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 2.3] h, true 0.1 h

### case `1-700822736` — prefix length 4, DL-sim 0.00
- prefix: Accepted|In Progress → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress
- true suffix: Accepted|Wait → Completed|Resolved → Completed|Closed
- greedy suffix: Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress (truncated)
- at the prefix end, top-3 next: `Queued|Awaiting Assignment` 0.35, `Accepted|Wait - User` 0.16, `Completed|Resolved` 0.15; true next `Accepted|Wait`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 41.8] h, true 0.0 h

### case `1-722415150` — prefix length 19, DL-sim 0.00
- prefix: Accepted|In Progress → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Wait → Accepted|Wait - User → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Wait - User → Accepted|Wait - User → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Wait - User → Queued|Awaiting Assignment → Accepted|In Progress
- true suffix: Accepted|Wait - User → Completed|Resolved → Completed|Closed
- greedy suffix: Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Assigned → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress (truncated)
- at the prefix end, top-3 next: `Queued|Awaiting Assignment` 0.42, `Accepted|Assigned` 0.27, `Completed|Resolved` 0.09; true next `Accepted|Wait - User`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 4.0] h, true 0.8 h
