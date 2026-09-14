# Failure analysis — incidents_chrono_strict (gru_multihead seed 0)

## 1. Do errors compound in multi-step prediction?

Per-step error rate of the greedy suffix (step k = k-th predicted event after the prefix; population shrinks as true suffixes end):

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| error | 0.35 | 0.63 | 0.62 | 0.75 | 0.72 | 0.66 | 0.72 | 0.76 | 0.72 | 0.72 | 0.76 | 0.73 | 0.72 | 0.77 | 0.85 |
| n | 1000 | 821 | 606 | 386 | 250 | 144 | 123 | 103 | 79 | 65 | 50 | 37 | 32 | 26 | 20 |

- Of 352 suffixes with a first wrong step that is not the last step, the mean error rate *after* the first wrong step is **0.85** (vs overall per-step error 0.70); 91% of them are mostly wrong afterwards (derailment).
- DL-similarity by prefix length: k=1: 0.608 (n=165), k=2: 0.573 (n=137), k=3: 0.440 (n=160), k=4-5: 0.303 (n=198), k=6-10: 0.451 (n=237), k=11-20: 0.406 (n=94), k=21-10000: 0.624 (n=9)

## 2. Are predictions internally coherent?

- Post-terminal continuation rate (rollout emits another activity after a terminal one): **0.000**
- EOS precision / recall (next-event head): 0.995 / 1.000
- Mean predicted vs true suffix length: 22.57 vs 4.95; valid-termination rate 0.611
- Rollouts with the same activity ≥3× consecutively (loops): 0.000
- Δt: MAE 21.0 h, medAE 0.2 h, 80% interval coverage 0.76 (mean width 44 h). Negative Δt is impossible by construction (softplus-free clamp at 0 in rollouts).
- Remaining-time monotonicity (fraction of consecutive positions where predicted remaining time does not increase): **0.67**

## 3. What is lost from long histories?

- Next-activity accuracy by prefix length: 1: 0.966 (n=1133), 2: 0.484 (n=1129), 3: 0.775 (n=708), 4-5: 0.631 (n=1191), 6-10: 0.725 (n=1369), 11-20: 0.719 (n=509), 21-10000: 0.722 (n=54)
- Accuracy when no informative event (REJECTED/Queued/Wait/Reminder/SAVED) has occurred yet: 0.738 (n=2777)
- Accuracy by distance (events) since the most recent informative event: 1-2: 0.634 (n=2235), 3-5: 0.505 (n=398), 6-10: 0.645 (n=31)
- State-separation probe: mean KL( p(next|prefix) ‖ p(next|prefix without the first informative event) ) by distance: 1-2: 0.159 (n=568), 3-5: 0.004 (n=475), 6-10: 0.000 (n=295), 11+: 0.000 (n=142). A KL that decays toward 0 with distance means the state has forgotten the event.

## 4. Concrete failure cases (3 worst greedy suffixes among true suffixes of length ≥ 3)

### case `1-740680611` — prefix length 9, DL-sim 0.00
- prefix: Accepted|In Progress → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Accepted|Assigned → Accepted|In Progress → Accepted|Wait - User → Completed|Resolved → Accepted|In Progress
- true suffix: Accepted|Wait - User → Completed|Resolved → Completed|Closed
- greedy suffix: Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress (truncated)
- at the prefix end, top-3 next: `Queued|Awaiting Assignment` 0.28, `Completed|Resolved` 0.28, `Accepted|Wait - User` 0.18; true next `Accepted|Wait - User`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 0.3] h, true 0.0 h

### case `1-738190721` — prefix length 5, DL-sim 0.00
- prefix: Queued|Awaiting Assignment → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress
- true suffix: Accepted|Wait → Accepted|Wait - User → Completed|Resolved → Completed|Closed
- greedy suffix: Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress (truncated)
- at the prefix end, top-3 next: `Queued|Awaiting Assignment` 0.35, `Completed|Resolved` 0.22, `Accepted|Assigned` 0.17; true next `Accepted|Wait`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 3.2] h, true 0.0 h

### case `1-740444716` — prefix length 4, DL-sim 0.00
- prefix: Accepted|In Progress → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress
- true suffix: Accepted|Wait - User → Completed|Resolved → Completed|Closed
- greedy suffix: Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress (truncated)
- at the prefix end, top-3 next: `Queued|Awaiting Assignment` 0.30, `Completed|Resolved` 0.21, `Accepted|Assigned` 0.17; true next `Accepted|Wait - User`
- first divergence at suffix step 0; Δt 80% interval at prefix end: [0.0, 1.5] h, true 0.2 h
