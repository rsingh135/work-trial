# Downstream uses of the state — incidents_chrono_strict / transformer_multihead (3 seeds)

## 1. Selective prediction (abstain on highest predictive entropy)

| coverage | 100% | 90% | 80% | 70% | 50% | 30% |
|---|---|---|---|---|---|---|
| accuracy | 0.715 | 0.772 | 0.825 | 0.885 | 0.960 | 0.974 |

- single-model accuracy 0.710 → 3-seed ensemble 0.715

## 2. Epistemic uncertainty (seed-ensemble mutual information)

- mean predictive entropy 0.888 nats; mean mutual information 0.0256 nats (on wrong predictions 0.0409637950360775, on right 0.0195)
- unseen-label rate in this test set: 0.0000

Compare this row across the random / chronological / strict runs of the same log: epistemic uncertainty should rise under shift.

## 3. SLA-breach risk P(remaining > T) from the Laplace head

| threshold | T (hours) | positive rate | AUROC (ensemble / single) | Brier (ensemble / single) |
|---|---|---|---|---|
| train_median | 30 | 0.81 | 0.616 / 0.613 | 0.213 / 0.206 |
| train_p90 | 330 | 0.04 | 0.679 / 0.658 | 0.040 / 0.042 |

## 4. Anomaly score = per-case mean next-activity NLL

- Spearman(score, case length) = 0.53 (confound check: a high value would mean the score just flags long cases)
- Spearman(score, Markov-k2 score) = 0.07 (agreement with a counting baseline)

Top-5 most anomalous test cases:

- `1-740781545` (score 5.91, length 2): Accepted|In Progress → Completed|Resolved
- `1-740812155` (score 5.14, length 2): Accepted|Wait - Implementation → Completed|Resolved
- `1-740462311` (score 4.88, length 2): Accepted|Wait → Completed|Resolved
- `1-740782031` (score 4.41, length 4): Accepted|In Progress → Accepted|Wait - User → Accepted|Wait - User → Accepted|Wait - Implementation
- `1-740328411` (score 3.77, length 3): Accepted|Assigned → Accepted|Wait - User → Completed|Resolved
