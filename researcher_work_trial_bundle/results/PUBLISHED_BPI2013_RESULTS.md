# Published BPI 2013 Reference Results

Source: E. Rama-Maneiro, J. C. Vidal, and M. Lama, *Deep Learning for
Predictive Business Process Monitoring: Review and Benchmark*, arXiv:2009.13251
v4. The source PDF is included in `papers/`.

These are transcriptions of the two BPI 2013 columns only. Consult the source
table and surrounding text before citing them.

## Evaluation protocol

- Five-fold cross-validation over complete cases.
- Within the four training folds, an 80/20 train/validation split.
- An end-of-case activity appended to every trace.
- Trace-level attributes excluded by the benchmark.
- Reported values are means across the five test folds.

Results from a single fold, a temporal split, different prefix selection, or a
different activity classifier are not directly comparable.

## Table 4 - Dataset statistics

| Dataset | Cases | Activities | Events | Mean / max case length | Mean / max event duration (days) | Mean / max case duration (days) | Variants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BPI 2013 Closed Problems | 1,487 | 7 | 6,660 | 4.48 / 35 | 51.42 / 2,254.84 | 178.88 / 2,254.85 | 327 |
| BPI 2013 Incidents | 7,554 | 13 | 65,533 | 8.68 / 123 | 1.57 / 722.25 | 12.08 / 771.35 | 2,278 |

## Table 5 - Next-activity accuracy (%)

Higher is better.

| Approach | Closed Problems | Incidents |
| --- | ---: | ---: |
| Camargo | 54.67 | 66.68 |
| Evermann | 58.83 | 66.78 |
| Hinkka | 63.47 | **74.69** |
| Khan | 43.58 | 51.91 |
| Mauro | 24.94 | 36.67 |
| Pasquadibisceglie | 47.45 | 46.03 |
| Tax | **64.01** | 70.09 |
| Theis, without attributes | 59.48 | 59.41 |
| Theis, with attributes | 54.65 | 51.50 |

## Table 7 - Activity-suffix normalized DL score

The caption calls this “DL distance,” while the metric definition describes a
normalized similarity between 0 and 1 and the highlighting treats higher values
as better. Preserve that caveat when reporting it.

| Approach / decoding | Closed Problems | Incidents |
| --- | ---: | ---: |
| Camargo, argmax | **0.6641** | 0.2607 |
| Camargo, random | 0.5357 | **0.5294** |
| Evermann | 0.6416 | 0.4730 |
| Francescomarino | 0.5276 | 0.3607 |
| Tax | 0.5824 | 0.3336 |

## Table 9 - Remaining-time MAE (days)

Lower is better.

| Approach / decoding | Closed Problems | Incidents |
| --- | ---: | ---: |
| Camargo, argmax | 257.086 | 28.132 |
| Camargo, random | 257.697 | 28.511 |
| Francescomarino | 191.100 | 35.405 |
| Navarin | **159.164** | **12.366** |
| Tax | 172.849 | 30.082 |

## What is not covered

- Open Problems is absent because its cases are unfinished.
- BPI 2020 is absent because the paper predates a standardized predictive
  benchmark over those five logs.
- Next-timestamp results are in the paper's supplementary material rather than
  its main result tables.
- These tables do not measure calibration, attribute/delta prediction, or
  recursive state consistency; those require new evaluation code.
