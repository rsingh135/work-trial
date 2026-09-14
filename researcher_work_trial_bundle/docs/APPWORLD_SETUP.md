# AppWorld Setup and Evaluation Hygiene

AppWorld is intentionally not copied into this bundle. Install it from the
official distribution so that its protected benchmark assets, versioning, and
licenses remain intact.

The PyPI version current when this bundle was assembled on 2026-08-13 was
`0.1.3.post1`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install 'appworld==0.1.3.post1'
appworld install
appworld download data
appworld verify tests
appworld verify tasks
```

Official project documentation:
<https://github.com/StonyBrookNLP/appworld/blob/main/README.md>

## Split policy

- `train`: collecting demonstrations, supervised learning, RL, and manual error
  analysis.
- `dev`: hyperparameter selection and manual error analysis.
- `test_normal` and `test_challenge`: aggregate final evaluation only. Do not
  inspect task-wise reports, tune prompts, or perform error analysis on them.

For the work trial, use a small pinned subset of `train` to produce learning
traces and a disjoint subset of `dev` for the paired before/after demonstration.
Do not require test-set execution.

## Useful native artifacts

AppWorld already records environment input/output, database changes, evaluation
reports, and checkpoints under each experiment output directory. The harness
should normalize or reference these artifacts rather than duplicate hidden
ground-truth data in a public trace format.

## Evaluation

- Task Goal Completion (TGC): proportion of tasks passing all evaluator tests.
- Scenario Goal Completion (SGC): proportion of scenarios for which all task
  variants pass. It requires complete task-variation groups.
- Also report evaluator assertions passed, collateral changes, invalid API
  calls, interactions, tokens, latency, and cost where available.

AppWorld permits parallel rollouts, but each process can hold only one active
world. Unique `experiment_name + task_id` combinations are also required to
avoid output collisions. See:
<https://github.com/StonyBrookNLP/appworld/blob/main/guides/parallelizing_worlds.md>

## Publication caution

The AppWorld maintainers ask users not to post unpacked code or data from its
protected `.bundle` files in plain text or images. Keep raw traces and database
diffs private unless their contents and applicable license have been reviewed.
