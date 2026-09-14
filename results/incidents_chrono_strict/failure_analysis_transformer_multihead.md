# Failure analysis — incidents_chrono_strict (transformer_multihead seed 0)

## 1. Do errors compound in multi-step prediction?

Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error | 0.32 | 0.58 | 0.65 | 0.74 | 0.75 | 0.78 | 0.72 | 0.69 | 0.76 | 0.69 | 0.64 | 0.74 | 0.67 | 0.67 | 0.50 |
| n | 1000 | 821 | 632 | 440 | 325 | 229 | 163 | 108 | 50 | 35 | 25 | 19 | 12 | 6 | 4 |

- Of 362 suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is **0.89** (vs overall per-step error 0.66); 93% of them are mostly wrong afterwards (derailment).
- DL-similarity by prefix length: k=1: 0.599 (n=165), k=2: 0.537 (n=137), k=3: 0.601 (n=160), k=4-5: 0.623 (n=198), k=6-10: 0.603 (n=237), k=11-20: 0.579 (n=94), k=21-10000: 0.614 (n=9)

## 2. Are predictions internally coherent?

- Post-terminal continuation rate (rollout emits another activity after a terminal one): **0.000**
- EOS precision / recall (next-event head): 0.995 / 1.000
- Mean predicted vs true suffix length: 7.77 vs 4.95; valid-termination rate 0.988
- Rollouts with the same activity ≥3× consecutively (loops): 0.000
- Δt: MAE 20.1 h, medAE 0.2 h, 80% interval coverage 0.70 (mean width 53 h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).
- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **0.66**

## 3. What is lost from long histories?

- Next-activity accuracy by prefix length: 1: 0.966 (n=1133), 2: 0.505 (n=1129), 3: 0.790 (n=708), 4-5: 0.641 (n=1191), 6-10: 0.701 (n=1369), 11-20: 0.676 (n=509), 21-10000: 0.704 (n=54)
- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: 0.743 (n=2777)
- Accuracy by distance (events) since the most recent informative event: 1-2: 0.615 (n=2235), 3-5: 0.548 (n=398), 6-10: 0.645 (n=31)
- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: 1-2: 0.107 (n=568), 3-5: 0.017 (n=475), 6-10: 0.006 (n=295), 11+: 0.013 (n=142). A KL that decays toward 0 with distance means the state has forgotten the event.

## 4. Concrete failure cases (3 worst greedy suffixes among true suffixes of length ≥ 3)

### case `1-740693170` — prefix length 2, DL-sim 0.00
- prefix: Accepted|In Progress → Accepted|In Progress
- true suffix: Accepted|Wait → Accepted|Wait - User → Completed|Resolved → Completed|Closed
- greedy suffix: Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress (truncated)
- at the prefix end, top-3 next: `Queued|Awaiting Assignment` 0.45, `Completed|Resolved` 0.28, `Accepted|Wait - User` 0.11; true next `Accepted|Wait`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 0.6] h, true 0.1 h

### case `1-740682191` — prefix length 15, DL-sim 0.02
- prefix: Accepted|In Progress → Accepted|In Progress → Accepted|In Progress → Queued|Awaiting Assignment → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Wait → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Wait - User → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment
- true suffix: Accepted|In Progress → Accepted|Wait → Completed|Resolved → Completed|Closed
- greedy suffix: Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment (truncated)
- at the prefix end, top-3 next: `Accepted|In Progress` 0.94, `Queued|Awaiting Assignment` 0.05, `Accepted|Wait - User` 0.01; true next `Accepted|In Progress`
- first divergence at suffix step 1; Δt 80% interval at prefix end: [0.1, 140.2] h, true 0.3 h

### case `1-740791756` — prefix length 12, DL-sim 0.02
- prefix: Accepted|In Progress → Accepted|In Progress → Completed|In Call → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Wait → Accepted|Assigned → Accepted|In Progress → Queued|Awaiting Assignment
- true suffix: Accepted|In Progress → Completed|Resolved → Completed|Closed
- greedy suffix: Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User → Accepted|In Progress → Accepted|Wait - User (truncated)
- at the prefix end, top-3 next: `Accepted|In Progress` 0.97, `Queued|Awaiting Assignment` 0.02, `Completed|Resolved` 0.00; true next `Accepted|In Progress`
- first divergence at suffix step 1; Δt 80% interval at prefix end: [0.1, 254.3] h, true 0.1 h
