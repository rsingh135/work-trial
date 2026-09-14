# Work Trial: Sequential Modeling and Reusable Agent-Trace Learning

> Verbatim copy of the brief as provided. This file is never edited.

## Context

Many operational systems produce partial histories rather than clean, fixed-size examples. A useful model of these systems should be able to update as new events arrive, retain the information needed for several downstream questions, express uncertainty about what may happen next, and remain useful when the process or event vocabulary changes.

This work trial has two connected parts:

1. Design and prototype a model for forecasting business-process event traces.
2. Build a small, reusable loop that collects agent traces, trains a model from them, and uses the trained model inside the agent harness again.

We care more about sound formulation, experimental rigor, and a working end-to-end slice than exhaustive tuning or state-of-the-art benchmark results.

## Time frame

You will have ~3 working days to complete the exercise. Use your judgment to allocate time across design, implementation, experimentation, and analysis. We do not expect exhaustive tuning or full-scale benchmark runs; prioritize a sound formulation and a working end-to-end path. Clearly identify what you completed, what you deliberately left out, and what you would do next.

## Provided materials

You will receive a reference bundle containing:

- The three BPI Challenge 2013 logs and five BPI Challenge 2020 logs in XES format.
- Official dataset documentation and metadata.
- A short paper reading list.
- Published BPI 2013 reference results and their evaluation protocol.
- AppWorld installation and split-hygiene notes.

Treat the raw event logs as the source of truth. Do not assume that similarly named fields have identical semantics across logs.

## Part 1 — Model a partially observed business process

### Objective

Design a model that consumes the observed prefix of a case and maintains a representation useful for forecasting its continuation. The same underlying model should support multiple prediction questions rather than requiring an unrelated model for each one.

At minimum, support:

- The next activity or event class.
- The time until the next event or the remaining case duration.
- A multi-step continuation, such as an activity suffix or terminal outcome.

You may add other targets if they help the representation or evaluation—for example, event attributes, completion probability, or anomaly likelihood.

### Subtasks

#### 1. Data audit and task formulation

- Describe the observation available at each event and the targets derived from a case prefix.
- Identify missing data, censoring, high-cardinality fields, time irregularity, and possible sources of leakage.
- Explain which fields you include, exclude, decompose, or mask.
- State how the representation is updated when a new event arrives.

#### 2. Evaluation protocol

- Split at the case level; never place events from one case in multiple splits.
- Fit vocabularies, imputers, scalers, and feature selection on training data only.
- Establish a reproducible in-distribution split and at least one distribution-shift evaluation, such as a chronological split or transfer to another log.
- Use BPI 2013 Incidents and Closed Problems for comparisons with the packaged published results when—and only when—your protocol matches theirs.
- Make the ingestion pipeline work for all five BPI 2020 logs. For substantive experiments, select at least two and explain why they are informative.
- Treat BPI 2013 Open Problems as unfinished/right-censored unless you justify a different interpretation.

#### 3. Baseline

Implement at least one credible, inexpensive baseline. Examples include a frequency or transition baseline, gradient-boosted features over prefixes, or a small recurrent model. The baseline should use the same split and target construction as the proposed model.

#### 4. Proposed architecture

Provide a precise architecture specification, including:

- Inputs and event/attribute encoding.
- How history is compressed and updated.
- Training objectives and their weighting.
- How multi-step predictions are produced.
- How uncertainty is represented or measured.
- How schema differences and unseen values are handled.
- Expected compute, memory, and inference costs.

Implement enough of the architecture to test its central claim. If the complete design does not fit the available time, clearly separate implemented components from proposed ones.

#### 5. Experiments and analysis

Report at least:

| Capability | Primary metric | Useful secondary metric |
|---|---|---|
| Next event | Negative log-likelihood or cross-entropy | Macro-F1 and top-1 accuracy |
| Time prediction | MAE in an explicitly stated unit | Median absolute error |
| Multi-step continuation | Normalized edit similarity | Exact completion or valid-termination rate |
| Uncertainty | Calibration error or Brier score | Reliability plot |
| Distribution shift | Same task metrics on the shifted split | Relative degradation from in-distribution performance |

If you define a composite score, define it before examining final test results and still report every component separately. Include confidence intervals or variation across seeds where practical.

Analyze at least three failure cases. We are particularly interested in whether errors compound during multi-step prediction, whether predictions remain internally coherent, and which information is lost from long histories.

### Part 1 deliverable

A reviewer should be able to run one command that preprocesses a small sample, trains the baseline and proposed model, and produces a machine-readable results file. Full-scale runs may be separate.

## Part 2 — Build a reusable trace-to-training-to-agent loop

### Objective

Build a benchmark-agnostic harness that can:

- Run an agent in an interactive environment.
- Record complete, versioned trajectories.
- Convert those trajectories into training examples.
- Train or update a model from the examples.
- Insert the trained model back into the harness so that it affects subsequent agent behavior.
- Compare the resulting agent with the original baseline on held-out tasks.

Use AppWorld as the first environment adapter. Keep the interfaces sufficiently generic that another tool-use benchmark could be added without rewriting the collector, dataset builder, training entry point, or evaluator.

### Required trace contents

Each step should preserve, where available:

- Environment, benchmark, task, episode, and run identifiers.
- Dataset split and all relevant version information.
- Step index and timestamp.
- Current observation and available action/tool schemas.
- The agent action, structured tool call, and tool result.
- Environment reward, success state, or evaluator outputs.
- Errors, retries, termination reason, and token/latency/cost metadata.
- References needed to reconstruct how a training example was produced.

Document redaction rules for secrets or sensitive values. Avoid silently dropping failed actions: they are part of the trajectory and can be useful training data.

### Potential end-to-end demonstration

Using a small subset of AppWorld training tasks:

- Collect traces from a reproducible baseline agent.
- Validate and serialize the traces using a documented schema.
- Construct prefix/target training examples.
- Train a lightweight model that serves a clearly defined role—for example, selecting or ranking the next action, predicting action validity, estimating task progress, or proposing a continuation.
- Put that model back into the agent loop. An offline-only predictor does not complete this part of the task.
- Evaluate on held-out development tasks that were not used for training or prompt/model selection.

You do not need to train a large language model or run the full benchmark. A small learned component in a clean, reproducible loop is preferable to an expensive but poorly controlled run.

### Part 2 metrics

Report the metrics relevant to the model's chosen role, plus:

- AppWorld task-goal completion and scenario-goal completion where supported.
- Invalid action/tool-call rate.
- Steps and wall-clock time per episode.
- Trace validation success rate and number of usable training examples.
- Baseline-versus-trained-agent performance on the same held-out tasks.
- Run-to-run variation if the agent is stochastic.

Explain any cases in which the learned component improves an offline metric but does not improve end-to-end task success.

## Deliverables

Please submit:

- Source code and a reproducible environment specification.
- A concise design note, approximately three to five pages, covering both parts and the connection between them.
- A README with exact commands for data inspection, training, evaluation, and one small end-to-end demo.
- Machine-readable experiment configurations and results.
- A small set of example traces, with any sensitive content removed.
- Tests for the trace schema and at least the critical train/reinsert path.
- A short limitations section and a prioritized next-step plan.

Do not include downloaded AppWorld protected data or credentials in the submission.

## Closing Notes

This brief is by no means exhaustive or thorough. If there are more intriguing directions that this brief is limiting, please let us know and we will modify the scope. If there are any blocking issues resource wise, let us know as well. Use discretion when deploying card funds for compute / resources.

Document EVERYTHING. Please let us take a trip through your brain. Share your notion / gdoc / whatever you use to take notes as you are working on this.

Hope you enjoy the tasks!
