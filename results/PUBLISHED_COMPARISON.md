# BPI 2013 — reproduction of the published protocol vs. published results

Protocol (Rama-Maneiro et al., arXiv:2009.13251 v4, as transcribed in the bundle): 5-fold CV over cases, 80/20 train/val within the training folds, end-of-case token appended to **every** trace, trace-level attributes excluded, metrics averaged over the five test folds. Every prefix (length 1 … T, the last one predicting the end token) is scored. Remaining-time MAE in days, suffix = normalised Damerau–Levenshtein similarity (greedy decoding; `sampled` = mean over 5 ancestral samples where available). Remaining difference to the paper: my models use *event*-level attributes (role, group, impact, product, resource) — the closest published row is therefore *Theis, with attributes*; the Markov baseline uses activity labels only.

## Closed Problems (5 folds)

| model | next-activity acc % (mean ± sd over folds) | suffix DL sim | remaining MAE days | NLL |
|---|---|---|---|---|
| **markov_k2 (this work)** | 63.08 ± 1.07 | 0.6006 ± 0.0115 | 135.27 ± 13.65 | 0.994 |
| **gbm (this work)** | 70.68 ± 1.14 | 0.6260 ± 0.0137 | 102.93 ± 11.09 | 1.036 |
| **gru_multihead (this work)** | 68.14 ± 1.47 | 0.6335 ± 0.0166 | 111.51 ± 9.62 | 0.898 |
| Tax (published) | 64.01 | 0.5824 | 172.849 | – |
| Hinkka (published) | 63.47 | – | – | – |
| Theis (no attrs) (published) | 59.48 | – | – | – |
| Evermann (published) | 58.83 | 0.6416 | – | – |
| Camargo (published) | 54.67 | 0.6641 | 257.086 | – |
| Theis (attrs) (published) | 54.65 | – | – | – |
| Francescomarino (published) | – | 0.5276 | 191.1 | – |

- folds: fold0 test n=298, fold1 test n=298, fold2 test n=297, fold3 test n=297, fold4 test n=297; git ec23fc5

## Incidents (5 folds)

| model | next-activity acc % (mean ± sd over folds) | suffix DL sim | remaining MAE days | NLL |
|---|---|---|---|---|
| **markov_k2 (this work)** | 59.04 ± 0.29 | 0.2791 ± 0.0058 | 12.47 ± 0.67 | 1.015 |
| **gbm (this work)** | 76.69 ± 0.20 | 0.5431 ± 0.0065 | 11.22 ± 0.77 | 0.707 |
| **gru_multihead (this work)** | 75.97 ± 0.52 | 0.5213 ± 0.0240 | 11.65 ± 0.58 | 0.744 |
| Hinkka (published) | 74.69 | – | – | – |
| Tax (published) | 70.09 | 0.3336 | 30.082 | – |
| Evermann (published) | 66.78 | 0.473 | – | – |
| Camargo (published) | 66.68 | 0.2607 | 28.132 | – |
| Theis (no attrs) (published) | 59.41 | – | – | – |
| Theis (attrs) (published) | 51.50 | – | – | – |
| Francescomarino (published) | – | 0.3607 | 35.405 | – |

- folds: fold0 test n=1511, fold1 test n=1511, fold2 test n=1511, fold3 test n=1511, fold4 test n=1510; git ec23fc5

Published numbers are transcriptions from the bundle; see `researcher_work_trial_bundle/results/PUBLISHED_BPI2013_RESULTS.md` for caveats (Table 7 caption ambiguity, supplementary next-timestamp results not transcribed).
