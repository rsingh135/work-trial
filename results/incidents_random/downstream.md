# Downstream uses of the state — incidents_random / transformer_multihead (3 seeds)

## 1. Selective prediction (abstain on highest predictive entropy)

| coverage | 100% | 90% | 80% | 70% | 50% | 30% |
|---|---|---|---|---|---|---|
| accuracy | 0.765 | 0.814 | 0.857 | 0.902 | 0.968 | 0.979 |

- single-model accuracy 0.756 → 3-seed ensemble 0.765

## 2. Epistemic uncertainty (seed-ensemble mutual information)

- mean predictive entropy 0.662 nats; mean mutual information 0.0289 nats (on wrong predictions 0.060364704579114914, on right 0.0192)
- unseen-label rate in this test set: 0.0000

Compare this row across the random / chronological / strict runs of the same log: epistemic uncertainty should rise under shift.

## 3. SLA-breach risk P(remaining > T) from the Laplace head

| threshold | T (hours) | positive rate | AUROC (ensemble / single) | Brier (ensemble / single) |
|---|---|---|---|---|
| train_median | 184 | 0.53 | 0.753 / 0.743 | 0.213 / 0.222 |
| train_p90 | 678 | 0.10 | 0.735 / 0.736 | 0.082 / 0.083 |

## 4. Anomaly score = per-case mean next-activity NLL

- Spearman(score, case length) = 0.65 (confound check: a high value would mean the score just flags long cases)
- Spearman(score, Markov-k2 score) = 0.04 (agreement with a counting baseline)

Top-5 most anomalous test cases:

- `1-740414265` (score 3.23, length 5): Queued|Awaiting Assignment → Accepted|In Progress → Completed|Resolved → Completed|Resolved → Completed|Resolved
- `1-737375500` (score 2.67, length 10): Accepted|In Progress → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|In Progress → Queued|Awaiting Assignment → Accepted|Assigned → Accepted|Wait - User → Accepted|Assigned → Accepted|Wait - User → Completed|Resolved
- `1-739794081` (score 2.52, length 4): Queued|Awaiting Assignment → Accepted|Wait - User → Accepted|Wait - User → Completed|Resolved
- `1-740585356` (score 2.38, length 5): Accepted|In Progress → Accepted|In Progress → Accepted|In Progress → Accepted|In Progress → Completed|In Call
- `1-739248306` (score 2.21, length 3): Accepted|In Progress → Accepted|In Progress → Completed|In Call
