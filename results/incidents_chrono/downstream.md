# Downstream uses of the state — incidents_chrono / transformer_multihead (3 seeds)

## 1. Selective prediction (abstain on highest predictive entropy)

| coverage | 100% | 90% | 80% | 70% | 50% | 30% |
|---|---|---|---|---|---|---|
| accuracy | 0.739 | 0.791 | 0.835 | 0.886 | 0.970 | 0.992 |

- single-model accuracy 0.736 → 3-seed ensemble 0.739

## 2. Epistemic uncertainty (seed-ensemble mutual information)

- mean predictive entropy 0.734 nats; mean mutual information 0.0253 nats (on wrong predictions 0.05039377883076668, on right 0.0164)
- unseen-label rate in this test set: 0.0000

Compare this row across the random / chronological / strict runs of the same log: epistemic uncertainty should rise under shift.

## 3. SLA-breach risk P(remaining > T) from the Laplace head

| threshold | T (hours) | positive rate | AUROC (ensemble / single) | Brier (ensemble / single) |
|---|---|---|---|---|
| train_median | 201 | 0.19 | 0.666 / 0.673 | 0.418 / 0.466 |

## 4. Anomaly score = per-case mean next-activity NLL

- Spearman(score, case length) = 0.04 (confound check: a high value would mean the score just flags long cases)
- Spearman(score, Markov-k2 score) = 0.47 (agreement with a counting baseline)

Top-5 most anomalous test cases:

- `1-740781545` (score 4.14, length 2): Accepted|In Progress → Completed|Resolved
- `1-740782031` (score 4.01, length 4): Accepted|In Progress → Accepted|Wait - User → Accepted|Wait - User → Accepted|Wait - Implementation
- `1-740462311` (score 3.63, length 2): Accepted|Wait → Completed|Resolved
- `1-740453701` (score 3.63, length 3): Completed|Resolved → Queued|Awaiting Assignment → Completed|Resolved
- `1-740421014` (score 3.19, length 3): Accepted|In Progress → Accepted|Wait - User → Completed|Resolved
