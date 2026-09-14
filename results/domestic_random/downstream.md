# Downstream uses of the state — domestic_random / transformer_multihead (3 seeds)

## 1. Selective prediction (abstain on highest predictive entropy)

| coverage | 100% | 90% | 80% | 70% | 50% | 30% |
|---|---|---|---|---|---|---|
| accuracy | 0.891 | 0.912 | 0.932 | 0.973 | 1.000 | 1.000 |

- single-model accuracy 0.891 → 3-seed ensemble 0.891

## 2. Epistemic uncertainty (seed-ensemble mutual information)

- mean predictive entropy 0.289 nats; mean mutual information 0.0051 nats (on wrong predictions 0.01440960168838501, on right 0.0040)
- unseen-label rate in this test set: 0.0000

Compare this row across the random / chronological / strict runs of the same log: epistemic uncertainty should rise under shift.

## 3. SLA-breach risk P(remaining > T) from the Laplace head

| threshold | T (hours) | positive rate | AUROC (ensemble / single) | Brier (ensemble / single) |
|---|---|---|---|---|
| train_median | 120 | 0.60 | 0.846 / 0.839 | 0.156 / 0.158 |
| train_p90 | 335 | 0.13 | 0.702 / 0.700 | 0.105 / 0.106 |

## 4. Anomaly score = per-case mean next-activity NLL

- Spearman(score, case length) = 0.17 (confound check: a high value would mean the score just flags long cases)
- Spearman(score, Markov-k2 score) = 0.88 (agreement with a counting baseline)

Top-5 most anomalous test cases:

- `declaration 93444` (score 3.54, length 3): Declaration SUBMITTED by EMPLOYEE → Declaration FINAL_APPROVED by SUPERVISOR → Declaration REJECTED by MISSING
- `declaration 95149` (score 2.09, length 3): Declaration SAVED by EMPLOYEE → Request Payment → Payment Handled
- `declaration 96098` (score 1.85, length 4): Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by PRE_APPROVER → Declaration FINAL_APPROVED by SUPERVISOR → Declaration REJECTED by MISSING
- `declaration 141310` (score 1.84, length 4): Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Payment Handled
- `declaration 94298` (score 1.66, length 6): Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by SUPERVISOR → Declaration REJECTED by EMPLOYEE
