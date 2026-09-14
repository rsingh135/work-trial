# Downstream uses of the state — international_random / transformer_multihead (3 seeds)

## 1. Selective prediction (abstain on highest predictive entropy)

| coverage | 100% | 90% | 80% | 70% | 50% | 30% |
|---|---|---|---|---|---|---|
| accuracy | 0.883 | 0.926 | 0.958 | 0.976 | 0.989 | 0.997 |

- single-model accuracy 0.883 → 3-seed ensemble 0.883

## 2. Epistemic uncertainty (seed-ensemble mutual information)

- mean predictive entropy 0.285 nats; mean mutual information 0.0053 nats (on wrong predictions 0.013960951939225197, on right 0.0042)
- unseen-label rate in this test set: 0.0000

Compare this row across the random / chronological / strict runs of the same log: epistemic uncertainty should rise under shift.

## 3. SLA-breach risk P(remaining > T) from the Laplace head

| threshold | T (hours) | positive rate | AUROC (ensemble / single) | Brier (ensemble / single) |
|---|---|---|---|---|
| train_median | 342 | 0.55 | 0.938 / 0.936 | 0.101 / 0.103 |
| train_p90 | 2058 | 0.11 | 0.906 / 0.905 | 0.072 / 0.072 |

## 4. Anomaly score = per-case mean next-activity NLL

- Spearman(score, case length) = 0.31 (confound check: a high value would mean the score just flags long cases)
- Spearman(score, Markov-k2 score) = 0.63 (agreement with a counting baseline)

Top-5 most anomalous test cases:

- `declaration 78711` (score 2.53, length 10): Permit SUBMITTED by EMPLOYEE → Permit FINAL_APPROVED by SUPERVISOR → Start trip → End trip → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration APPROVED by BUDGET OWNER → Declaration APPROVED by SUPERVISOR → Declaration REJECTED by DIRECTOR → Declaration REJECTED by EMPLOYEE
- `declaration 80762` (score 2.06, length 11): Permit SUBMITTED by EMPLOYEE → Permit APPROVED by PRE_APPROVER → Permit FINAL_APPROVED by DIRECTOR → Permit APPROVED by SUPERVISOR → Start trip → End trip → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled
- `declaration 74252` (score 1.96, length 7): Permit SUBMITTED by EMPLOYEE → Permit APPROVED by SUPERVISOR → Start trip → End trip → Permit FINAL_APPROVED by DIRECTOR → Declaration SAVED by EMPLOYEE → Permit REJECTED by MISSING
- `declaration 147335` (score 1.71, length 7): Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Start trip → Payment Handled → End trip
- `declaration 20797` (score 1.61, length 15): Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit REJECTED by SUPERVISOR → Permit REJECTED by EMPLOYEE → Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit APPROVED by SUPERVISOR → Permit FINAL_APPROVED by DIRECTOR → Declaration SUBMITTED by EMPLOYEE → Start trip → Declaration APPROVED by ADMINISTRATION → End trip → Declaration FINAL_APPROVED by SUPER
