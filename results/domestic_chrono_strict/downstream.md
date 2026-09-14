# Downstream uses of the state — domestic_chrono_strict / transformer_multihead (3 seeds)

## 1. Selective prediction (abstain on highest predictive entropy)

| coverage | 100% | 90% | 80% | 70% | 50% | 30% |
|---|---|---|---|---|---|---|
| accuracy | 0.915 | 0.913 | 0.912 | 0.958 | 0.999 | 1.000 |

- single-model accuracy 0.915 → 3-seed ensemble 0.915

## 2. Epistemic uncertainty (seed-ensemble mutual information)

- mean predictive entropy 0.325 nats; mean mutual information 0.0012 nats (on wrong predictions 0.0024241337087005377, on right 0.0011)
- unseen-label rate in this test set: 0.0000

Compare this row across the random / chronological / strict runs of the same log: epistemic uncertainty should rise under shift.

## 3. SLA-breach risk P(remaining > T) from the Laplace head

| threshold | T (hours) | positive rate | AUROC (ensemble / single) | Brier (ensemble / single) |
|---|---|---|---|---|
| train_median | 117 | 0.62 | 0.866 / 0.866 | 0.145 / 0.144 |
| train_p90 | 319 | 0.20 | 0.660 / 0.655 | 0.156 / 0.157 |

## 4. Anomaly score = per-case mean next-activity NLL

- Spearman(score, case length) = 0.79 (confound check: a high value would mean the score just flags long cases)
- Spearman(score, Markov-k2 score) = 0.86 (agreement with a counting baseline)

Top-5 most anomalous test cases:

- `declaration 140675` (score 2.48, length 3): Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration REJECTED by SUPERVISOR
- `declaration 142992` (score 2.48, length 4): Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Payment Handled
- `declaration 141310` (score 2.39, length 4): Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Payment Handled
- `declaration 138710` (score 2.35, length 4): Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Payment Handled
- `declaration 138147` (score 2.33, length 5): Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration APPROVED by BUDGET OWNER → Declaration FINAL_APPROVED by SUPERVISOR → Payment Handled
