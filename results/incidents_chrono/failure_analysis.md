# Failure analysis — incidents_chrono (gru_multihead seed 0)

## 1. Do errors compound in multi-step prediction?

Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error | 0.26 | 0.51 | 0.62 | 0.74 | 0.73 | 0.75 | 0.75 | 0.82 | 0.78 | 0.79 | 0.74 | 0.68 | 0.64 | 0.77 | 0.83 |
| n | 1000 | 820 | 637 | 457 | 342 | 236 | 181 | 134 | 98 | 72 | 53 | 40 | 36 | 30 | 24 |

- Of 362 suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is **0.90** (vs overall per-step error 0.69); 93% of them are mostly wrong afterwards (derailment).
- DL-similarity by prefix length: k=1: 0.261 (n=165), k=2: 0.337 (n=137), k=3: 0.576 (n=160), k=4-5: 0.596 (n=198), k=6-10: 0.665 (n=237), k=11-20: 0.645 (n=94), k=21-10000: 0.798 (n=9)

## 2. Are predictions internally coherent?

- Post-terminal continuation rate (rollout emits another activity after a terminal one): **0.000**
- EOS precision / recall (next-event head): 0.995 / 1.000
- Mean predicted vs true suffix length: 12.78 vs 4.95; valid-termination rate 0.930
- Rollouts with the same activity ≥3× consecutively (loops): 0.000
- Δt: MAE 10.4 h, medAE 0.1 h, 80% interval coverage 0.81 (mean width 62 h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).
- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **0.66**

## 3. What is lost from long histories?

- Next-activity accuracy by prefix length: 1: 0.954 (n=1133), 2: 0.402 (n=1129), 3: 0.832 (n=708), 4-5: 0.709 (n=1191), 6-10: 0.774 (n=1369), 11-20: 0.770 (n=509), 21-10000: 0.759 (n=54)
- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: 0.713 (n=2777)
- Accuracy by distance (events) since the most recent informative event: 1-2: 0.700 (n=2235), 3-5: 0.598 (n=398), 6-10: 0.839 (n=31)
- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: 1-2: 0.196 (n=568), 3-5: 0.004 (n=475), 6-10: 0.000 (n=295), 11+: 0.000 (n=142). A KL that decays toward 0 with distance means the state has forgotten the event.

## 4. Concrete failure cases (3 worst greedy suffixes among true suffixes of length ≥ 3)

### case `1-740617279` — prefix length 2, DL-sim 0.00
- prefix: Accepted|In Progress → Accepted|In Progress
- true suffix: Accepted|Wait → Completed|Resolved → Completed|Closed
- greedy suffix: Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress (truncated)
- at the prefix end, top-3 next: `Queued|Awaiting Assignment` 0.80, `Accepted|Wait - User` 0.06, `Accepted|Wait` 0.05; true next `Accepted|Wait`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 0.4] h, true 1.7 h

### case `1-740693170` — prefix length 2, DL-sim 0.00
- prefix: Accepted|In Progress → Accepted|In Progress
- true suffix: Accepted|Wait → Accepted|Wait - User → Completed|Resolved → Completed|Closed
- greedy suffix: Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress (truncated)
- at the prefix end, top-3 next: `Queued|Awaiting Assignment` 0.65, `Accepted|Wait - User` 0.09, `Completed|In Call` 0.09; true next `Accepted|Wait`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 0.4] h, true 0.1 h

### case `1-740703682` — prefix length 3, DL-sim 0.02
- prefix: Accepted|In Progress → Accepted|In Progress → Queued|Awaiting Assignment
- true suffix: Accepted|In Progress → Completed|Resolved → Completed|Closed
- greedy suffix: Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment (truncated)
- at the prefix end, top-3 next: `Accepted|In Progress` 0.96, `Queued|Awaiting Assignment` 0.03, `Accepted|Assigned` 0.00; true next `Accepted|In Progress`
- first divergence at suffix step 1; Δt 80% interval at prefix end: [0.2, 115.6] h, true 63.4 h
