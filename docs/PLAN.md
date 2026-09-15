# Plan: Work Trial — Sequential Modeling + Reusable Agent-Trace Loop

## Context
Three-day work trial. Two parts: (1) multi-task forecasting model over business-process event prefixes (BPI 2013 + BPI 2020 XES logs), (2) benchmark-agnostic trace→train→reinsert loop with AppWorld as first adapter. Reviewer values sound formulation, rigor, working end-to-end slice over SOTA numbers. Everything documented. User will drop reference bundle into `work_trial/`; Anthropic API for agent; local M5 Pro compute; markdown notes in repo; private GitHub remote.

## Decisions already made (and why)
- **Python 3.12 via uv**, not system 3.14: torch/lightgbm/pm4py/appworld wheels lag 3.14. One `pyproject.toml`, `uv.lock` = reproducible env spec.
- **Stack**: PyTorch (proposed model), LightGBM + numpy (baselines), lxml streaming (XES), pydantic v2 (trace schema), pytest.
- **Notes**: `instructions.md` (verbatim brief, never edited), `PROGRESS.md` (dated decision log, reasoning per choice), `DESIGN_NOTE.md` (3–5 page final), `README.md` (exact commands).
- **Work style**: long autonomous stretches, commit per milestone, ask only when blocked (bundle missing, API key missing, protocol ambiguity that changes results).

## Repo layout
```
work_trial/
  instructions.md  PROGRESS.md  DESIGN_NOTE.md  README.md  pyproject.toml  Makefile
  bundle/                 # user-dropped; raw XES gitignored, docs kept
  bpm/                    # Part 1
    ingest/xes.py         # streaming XES→parquet, per-log schema map
    ingest/audit.py       # missingness, cardinality, censoring, time gaps → results/audit_*.json
    data/splits.py        # case-level random + chronological + cross-log
    data/features.py      # vocab/scalers fit on train only, UNK handling
    data/prefixes.py      # prefix→targets (next act, Δt, remaining, suffix, outcome, censor mask)
    models/markov.py      # baseline 1: k-order transition + per-activity median Δt + modal suffix
    models/gbm.py         # baseline 2: LightGBM over prefix aggregate features
    models/recurrent.py   # proposed: multi-head GRU with MDN time head
    train.py eval.py metrics.py calibrate.py analyze_failures.py
  agentloop/              # Part 2
    schema.py             # versioned pydantic Trace/Episode/Step
    redaction.py          # secret masking rules
    envs/base.py          # Env protocol: reset/step/action_schema/evaluate/task_ids(split)
    envs/appworld.py      # adapter
    agent/llm_agent.py    # ReAct code agent (Anthropic), N-candidate sampling hook
    agent/policy.py       # Policy protocol; LLMOnly vs LearnedReranker
    collect.py build_dataset.py train.py evaluate.py
  configs/*.yaml  results/*.json  traces/examples/*.jsonl
  tests/                  # schema, prefix construction, split hygiene, train→reinsert path
  scripts/demo_part1.sh scripts/demo_part2.sh
```

## Part 1 design
**Observation per event**: activity (BPI2013: `concept:name` + `lifecycle:transition` fused, e.g. `Accepted|In Progress`; BPI2020: `concept:name`), timestamp → Δt since prev, elapsed since case start, hour/weekday; low-card attrs (org:group, impact, org:role) as embeddings with train-fit vocab + UNK; high-card (org:resource, product) frequency-thresholded to UNK. Case-level attrs only if known at case start (else leakage). Audit decides final list per log; log it in PROGRESS.md.

**Targets from prefix of length k**: next activity; log(Δt) to next event; remaining time; suffix to end; terminal outcome (last activity class). BPI2013 Open Problems: right-censored → remaining-time/suffix/outcome masked, next-event targets kept.

**State update**: h_t = GRU(h_{t-1}, enc(e_t)). O(1) per new event, no replay. Central claim tested: one shared recurrent state serves all heads and stays coherent under autoregressive rollout.

**Proposed model** (implement fully; small): event encoder (concat embeddings + time feats) → 2-layer GRU (d=128) → heads: (a) next-activity softmax, (b) Δt via 3-component log-normal mixture (MDN) → NLL, gives uncertainty, (c) remaining-time log-normal, (d) end-of-case Bernoulli. Loss = weighted sum, weights fixed before test (uncertainty-weighted variant as ablation if time). Multi-step: autoregressive rollout feeding sampled/greedy event back; K Monte-Carlo rollouts → suffix distribution. Calibration: temperature scaling fit on val; report ECE/Brier + reliability plot. Schema shift: per-log vocab + shared UNK; transfer mode remaps activities via shared vocab, unseen → UNK. Costs: params ~200k, train minutes on CPU/MPS.

**Baselines**: (1) k-order Markov (k=1,2) + median Δt by activity + modal suffix; (2) LightGBM on aggregate prefix features. Same splits, same targets.

**Splits**: case-level seeded 70/15/15 (in-distribution); chronological by case start (shift #1); cross-log transfer BPI2020 RequestForPayment→DomesticDeclarations (shift #2; shared activity vocab, same org). Substantive BPI2020 logs: DomesticDeclarations (large, simple) + InternationalDeclarations (longer, permit-linked, more variants) — informative contrast. BPI2013 Incidents + Closed Problems vs published results only if bundle protocol reproducible; else report side-by-side with explicit protocol diff.

**Metrics**: NLL, acc, macro-F1; MAE + medAE in hours; normalized Damerau-Levenshtein similarity + exact suffix match; ECE + Brier; shift degradation %. Bootstrap CIs over cases, 3 seeds. Failure analysis: error vs suffix position (compounding), coherence checks (event after terminal, negative Δt), long-prefix degradation curve.

**One command**: `make demo` → sample preprocess, train both baselines + proposed for few epochs, write `results/demo_results.json`.

## Part 2 design
**Env protocol** (`envs/base.py`): `task_ids(split)`, `reset(task_id)`, `step(action)→StepResult(obs, reward, done, info)`, `action_schema()`, `evaluate()`. AppWorld adapter wraps `AppWorld(task_id, experiment_name)`, `execute(code)`, `evaluate()`; supervisor task instruction as initial obs.

**Trace schema**: `schema_version`, env/benchmark/task/episode/run ids, split, versions (appworld, model id, prompt hash, git sha, seed), per step: index, ts, observation, action_schema ref, raw action, structured call, result, error, retry idx, reward/eval, tokens/latency/cost; termination reason; final evaluator output; `provenance` linking each training example to (episode, step). JSONL; validated before serialization. Redaction: mask values of keys matching `password|token|secret|api_key|access_token`, hash originals; documented in README. Failed actions retained.

**Learned component role**: **action reranker / validity scorer**. At each step LLM samples N=3 candidates (temp 0.7); model scores P(no error ∧ episode eventually succeeds | prefix features, candidate). Pick argmax. Model: hashed n-gram features over (last obs, last error, candidate code, API names) → logistic regression/LightGBM. Seconds to train, CPU, reproducible. Offline metric: AUC/logloss on held-out episodes. Online: reinserted via `Policy` hook.

**Data hygiene**: collect on subset of AppWorld `train` tasks (~30–40, 2–3 stochastic runs each); prompt/hparams chosen on train-internal holdout; final eval on `dev` tasks (~20) never touched. Never `test`.

**Metrics**: TGC, SGC, invalid-action rate, steps, wall-clock/episode, trace validation rate, #examples, baseline vs reranked over 3 runs, offline AUC vs online gain discussion.

**Cost**: claude-haiku-4-5, est. <$30 total.

## Sequence
1. Skeleton: `instructions.md`, `PROGRESS.md`, uv env, git init, `gh repo create --private work-trial`.
2. Wait/poll for bundle → read docs, protocol notes, AppWorld notes. Start `appworld` install in background.
3. XES ingest + audit all 8 logs → audit JSON + PROGRESS entries.
4. Splits, prefixes, metrics, Markov + GBM baselines, demo command.
5. Proposed model, training, ID + chrono + transfer runs, calibration, failure analysis, CIs.
6. Part 2: schema + tests, env adapter, agent, collect, build dataset, train reranker, reinsert, eval.
7. README, DESIGN_NOTE, results JSON, example traces, limitations/next steps, final push.

## Verification
- `uv run pytest` green (schema, prefix targets, split hygiene, train→reinsert).
- `make demo` produces `results/demo_results.json` from scratch.
- `scripts/demo_part2.sh` runs 2–3 AppWorld tasks end-to-end with reranker active.
- Every decision has a PROGRESS.md entry with reasoning.

## Needs from user
- Drop bundle in `work_trial/bundle/`.
- `export ANTHROPIC_API_KEY=...` in shell before Part 2 collection.
- OK to create private GitHub repo `rsingh135/work-trial` (approving this plan = OK).
