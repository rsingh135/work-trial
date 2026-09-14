# Reading Guide

## Required for Task 1

1. `bpi2013_challenge_summary.pdf`

   Read Section 2. It establishes that BPI 2013 has three logs, explains their
   business processes, and lists the process-owner questions.

2. `bpi2013_dataset_guide.pdf`

   Read Sections 1.1, 1.2, and 1.4. This is the compact domain and field guide.
   The longer VINST application manual is optional:
   <https://ceur-ws.org/Vol-1052/vinst_manual.pdf>

3. `bpi2020_attribute_explanation.pdf`

   Read both pages. It documents the source approval-flow fields and role IDs.

4. Official BPI 2020 challenge description

   <https://icpmconference.org/2020/bpi-challenge/>

   Read “Description of the Challenge,” “The Data,” and “The Process Flow.” It
   explains anonymization, overlap among logs, the 2017/2018 process change, and
   the relationship among permits, declarations, and requests.

5. `deep_learning_predictive_process_monitoring_benchmark_v4.pdf`

   Read Sections 2, 5.1, and 5.2. Tables 4, 5, 7, and 9 are the relevant BPI
   2013 statistics and reference results. The two BPI 2013 columns are
   transcribed in `results/PUBLISHED_BPI2013_RESULTS.md`.

## Contextual, not a canonical target

*Next Activity Prediction: An Application of Shallow Learning Techniques
Against Deep Learning Over the BPI Challenge 2020*, IEEE Access 11 (2023),
DOI: <https://doi.org/10.1109/ACCESS.2023.3325738>

This paper covers all five BPI 2020 logs, but its fixed three-event window,
activity-only inputs, temporal split, and particular train filtering make its
numbers protocol-specific. Read it as evidence that simple baselines matter,
not as an unconditional leaderboard.

## Required for Task 2

1. `appworld_acl2024.pdf`

   Read Sections 2, 3, and 4. Focus on state, API interactions, state-based
   evaluation, collateral damage, and the task/scenario split.

2. Current official documentation:
   <https://github.com/StonyBrookNLP/appworld/blob/main/README.md>

3. Parallel-world guidance, only if running multiple workers:
   <https://github.com/StonyBrookNLP/appworld/blob/main/guides/parallelizing_worlds.md>

The work trial does not require reading every BPI Challenge submission or every
AppWorld baseline paper. Additional papers should be selected in response to a
specific modeling claim.
