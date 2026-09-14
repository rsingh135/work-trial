# Downstream uses of the state — demo / gru_multihead (1 seeds)

## 1. Selective prediction (abstain on highest predictive entropy)

| coverage | 100% | 90% | 80% | 70% | 50% | 30% |
|---|---|---|---|---|---|---|
| accuracy | 0.908 | 0.921 | 0.953 | 0.980 | 0.990 | 1.000 |

- single-model accuracy 0.908 → 1-seed ensemble 0.908

## 2. Epistemic uncertainty (seed-ensemble mutual information)

- mean predictive entropy 0.371 nats; mean mutual information 0.0000 nats (on wrong predictions 0.0, on right 0.0000)
- unseen-label rate in this test set: 0.0000

Compare this row across the random / chronological / strict runs of the same log: epistemic uncertainty should rise under shift.

## 3. SLA-breach risk P(remaining > T) from the Laplace head

| threshold | T (hours) | positive rate | AUROC (ensemble / single) | Brier (ensemble / single) |
|---|---|---|---|---|
| train_median | 102 | 0.63 | 0.742 / 0.742 | 0.191 / 0.191 |
| train_p90 | 296 | 0.14 | 0.669 / 0.669 | 0.114 / 0.114 |

## 4. Anomaly score = per-case mean next-activity NLL

- Spearman(score, case length) = 0.71 (confound check: a high value would mean the score just flags long cases)
- Spearman(score, Markov-k2 score) = 0.84 (agreement with a counting baseline)

Top-5 most anomalous test cases:

- `declaration 88818` (score 1.97, length 3): Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE
- `declaration 94194` (score 1.84, length 3): Declaration SUBMITTED by EMPLOYEE → Declaration FINAL_APPROVED by SUPERVISOR → Declaration REJECTED by MISSING
- `declaration 96089` (score 1.83, length 3): Declaration SUBMITTED by EMPLOYEE → Declaration FINAL_APPROVED by SUPERVISOR → Declaration REJECTED by MISSING
- `declaration 93457` (score 1.83, length 3): Declaration SUBMITTED by EMPLOYEE → Declaration FINAL_APPROVED by SUPERVISOR → Declaration REJECTED by MISSING
- `declaration 101989` (score 1.59, length 8): Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by PRE_APPROVER → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled
