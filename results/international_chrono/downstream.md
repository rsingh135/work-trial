# Downstream uses of the state — international_chrono / transformer_multihead (3 seeds)

## 1. Selective prediction (abstain on highest predictive entropy)

| coverage | 100% | 90% | 80% | 70% | 50% | 30% |
|---|---|---|---|---|---|---|
| accuracy | 0.893 | 0.934 | 0.963 | 0.981 | 0.990 | 0.996 |

- single-model accuracy 0.894 → 3-seed ensemble 0.893

## 2. Epistemic uncertainty (seed-ensemble mutual information)

- mean predictive entropy 0.300 nats; mean mutual information 0.0058 nats (on wrong predictions 0.014134012162685394, on right 0.0048)
- unseen-label rate in this test set: 0.0000

Compare this row across the random / chronological / strict runs of the same log: epistemic uncertainty should rise under shift.

## 3. SLA-breach risk P(remaining > T) from the Laplace head

| threshold | T (hours) | positive rate | AUROC (ensemble / single) | Brier (ensemble / single) |
|---|---|---|---|---|
| train_median | 338 | 0.57 | 0.912 / 0.909 | 0.122 / 0.124 |
| train_p90 | 2325 | 0.01 | 0.897 / 0.892 | 0.046 / 0.050 |

## 4. Anomaly score = per-case mean next-activity NLL

- Spearman(score, case length) = 0.46 (confound check: a high value would mean the score just flags long cases)
- Spearman(score, Markov-k2 score) = 0.78 (agreement with a counting baseline)

Top-5 most anomalous test cases:

- `declaration 13216` (score 2.27, length 8): Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit FINAL_APPROVED by SUPERVISOR → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Start trip → Declaration REJECTED by EMPLOYEE → End trip
- `declaration 13321` (score 2.14, length 13): Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit FINAL_APPROVED by SUPERVISOR → Declaration SUBMITTED by EMPLOYEE → Start trip → End trip → Declaration APPROVED by ADMINISTRATION → Declaration APPROVED by BUDGET OWNER → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Payment Handled → Send Reminder → Send Reminder
- `declaration 28818` (score 1.50, length 10): Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit APPROVED by BUDGET OWNER → Permit APPROVED by SUPERVISOR → Permit FINAL_APPROVED by DIRECTOR → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Start trip → Declaration REJECTED by EMPLOYEE → End trip
- `declaration 50730` (score 1.47, length 6): Start trip → Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → End trip → Permit FINAL_APPROVED by SUPERVISOR → Declaration SAVED by EMPLOYEE
- `declaration 71076` (score 1.47, length 13): Permit SUBMITTED by EMPLOYEE → Permit APPROVED by ADMINISTRATION → Permit FINAL_APPROVED by SUPERVISOR → Declaration SUBMITTED by EMPLOYEE → Declaration REJECTED by ADMINISTRATION → Declaration REJECTED by EMPLOYEE → Declaration SUBMITTED by EMPLOYEE → Declaration APPROVED by ADMINISTRATION → Declaration FINAL_APPROVED by SUPERVISOR → Request Payment → Start trip → Payment Handled → End trip
