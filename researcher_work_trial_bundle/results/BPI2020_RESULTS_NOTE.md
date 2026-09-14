# BPI 2020 Reference-Result Note

There is no single canonical, protocol-matched result table covering all five
BPI 2020 logs for next activity, remaining time, and multi-step continuation.
Do not manufacture one by combining numbers from papers with different splits,
activity definitions, filtering, and prefix construction.

The most directly relevant published next-activity baseline is:

> D. Impedovo, G. Pirlo, and G. Semeraro, “Next Activity Prediction: An
> Application of Shallow Learning Techniques Against Deep Learning Over the BPI
> Challenge 2020,” *IEEE Access*, vol. 11, pp. 117947-117953, 2023.
> DOI: <https://doi.org/10.1109/ACCESS.2023.3325738>

Its protocol includes:

- All five BPI 2020 logs.
- Activity-label inputs only.
- A fixed prefix window of three events.
- A chronological 75/25 trace split, followed by additional filtering of
  overlapping training cases and a further discarded 25% portion.
- Decision Tree, Random Forest, AdaBoost, Gradient Boosting, KNN, SVM, and a
  three-block Bi-LSTM comparison.

The paper reports that Decision Tree and Gradient Boosting are strongest across
its logs and that the tested shallow methods outperform its Bi-LSTM baseline.
Those claims are useful motivation for including simple baselines, but the exact
accuracy values should only be compared under a reproduction of that protocol.

## Recommended work-trial treatment

Ask the candidate to establish a fresh BPI 2020 baseline table under the same
case-level split and target definitions used for their proposed model. Require:

- Most-common-next-activity and first-order Markov baselines.
- At least one learned direct predictor.
- Accuracy, macro-F1, and a proper probabilistic score.
- Per-log results rather than a pooled score.
- Explicit handling of cross-log overlap and temporal drift.

Published values may be reported in a separate “context only” column, never as
an apples-to-apples delta unless the protocol was reproduced.
