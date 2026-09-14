# Work trial: sequential modelling of process traces + a reusable agent-trace loop

Two connected parts (brief: `instructions.md`, decision log: `PROGRESS.md`, design note: `DESIGN_NOTE.md`):

* **Part 1** (`bpm/`): a multi-head recurrent model over business-process event prefixes (BPI 2013 + BPI 2020 XES logs) that predicts the next activity, the time to the next event, the remaining case time, and multi-step suffixes from one shared state, compared against Markov and gradient-boosting baselines under in-distribution, chronological and cross-log shift.
* **Part 2** (`agentloop/`): a benchmark-agnostic trace → dataset → train → reinsert → evaluate loop, with AppWorld as the first environment adapter and a mock environment for offline testing.

## Environment

```bash
uv sync --extra dev            # Python 3.12, torch, sklearn, pydantic v2 … (uv.lock is the reproducible spec)
make appworld-setup            # AppWorld in its own Python 3.11 venv (it pins pydantic<2) + ~200 MB data download
export ANTHROPIC_API_KEY=...   # only needed for Part 2 runs on AppWorld with a real LLM
```

Raw logs are read from the reference bundle at `researcher_work_trial_bundle/data/{bpi2013,bpi2020}/*.xes.gz` (paths in `bpm/ingest/registry.py`); the bundle data is not committed. Plain-language model explainer: `docs/WORLD_MODEL_EXPLAINED.md`.

## Part 1 — commands

| What | Command | Output |
|---|---|---|
| Data audit (all 8 logs) | `uv run python -m bpm.ingest.audit` | `results/audit/SUMMARY.md`, per-log JSON |
| **One-command demo** (sample, baselines + proposed, ~2 min) | `make demo` (= `uv run python -m bpm.run configs/demo.yaml`) | `results/demo.json` |
| Full-scale run for one config | `uv run python -m bpm.run configs/international_random.yaml` | `results/<name>.json`, `results/<name>/{split,encoder}.json`, `gru_seed0.pt` |
| All full-scale runs (hours) | `scripts/run_all_part1.sh` | `results/*.json`, logs in `results/logs/` |
| Aggregate tables + figures | `uv run python -m bpm.report` | `results/SUMMARY.md`, `results/figures/*.png` |
| Failure analysis for a run | `uv run python -m bpm.analyze_failures results/international_random.json` | `results/international_random/failure_analysis.md` |
| Published-protocol reproduction (5-fold CV, BPI2013) | `scripts/run_published_bpi2013.sh` then `uv run python -m bpm.published_compare` | `results/PUBLISHED_COMPARISON.md` |
| Tests | `uv run pytest -q` | |

Config keys are documented at the top of `bpm/run.py`; `--override key=value` patches any of them (e.g. `--override max_cases=500 seeds=[0]`).

Result JSON layout: `config`, `git_sha`, `split` (type, fingerprint, sizes), `data`, then `models.<name>.runs[]` each with `train_log` and `metrics` (`next_activity`, `next_dt_hours`, `remaining_hours`, `suffix`, `uncertainty` incl. reliability bins, `next_activity_by_prefix_len`), plus `seed_summary` and optional `transfer`.

## Part 2 — commands

| What | Command |
|---|---|
| **Offline end-to-end demo** (mock env + scripted LLM, no key) | `make demo-part2` (= `scripts/demo_part2_offline.sh`) → `results/part2_mock/summary.json` |
| Collect traces | `uv run python -m agentloop.collect --env appworld --split train --n-tasks 45 --out traces/appworld_train_v1 --cost-budget 10` |
| Build training examples | `uv run python -m agentloop.build_dataset traces/appworld_train_v1 --out datasets/appworld_v1` |
| Train the action scorer | `uv run python -m agentloop.train datasets/appworld_v1 --out models/appworld_v1` |
| Baseline vs. reranked agent on held-out dev tasks | `uv run python -m agentloop.evaluate --env appworld --split dev --n-tasks 30 --runs 2 --policy-model models/appworld_v1/model.pkl --out results/part2_appworld_v1` |
| Whole AppWorld pipeline | `scripts/run_part2_appworld.sh` (env vars `N_TRAIN_TASKS N_DEV_TASKS RUNS MODEL BUDGET`) |
| Dry run of the AppWorld adapter with canned actions (no key) | `uv run python -m agentloop.collect --env appworld --split train --n-tasks 3 --client canned --out traces/appworld_canned` |

Trace layout: `<out>/episodes.jsonl` (one validated, redacted `Episode` per line; schema in `agentloop/schema.py`), `<out>/schemas/<sha256>.json` (action/tool schemas referenced by `action_schema_ref`), `<out>/manifest.json` (config, versions, validation rate, cost), `<out>/invalid.jsonl` (records that failed validation — never dropped silently). Example traces: `traces/examples/`.

### Redaction rules (`agentloop/redaction.py`, rules v1)
Values of keys matching `password|passwd|secret|api_key|apikey|access_token|auth_token|token|verification_code|otp` in JSON-like output, keyword arguments of the same names in agent code, and JWT-shaped blobs are replaced by `<REDACTED:<8 hex of salted sha256>>` at serialisation. Supervisor persona fields (name/e-mail/phone) in AppWorld are synthetic and referenced by task instructions, so they are kept; add them to `SECRET_KEYS` for real data. The agent process always sees unredacted values.

### Adding another benchmark
Implement `agentloop/envs/base.py::Env` (`task_ids`, `scenario_id`, `reset`, `step`, `action_schema`, `evaluate`, `close`, `version`) and register it in `agentloop/collect.py::make_env`. The collector, dataset builder, trainer, policies and evaluator do not change; the action dict just needs a `type` and, for the current featuriser, a `code` string (or extend `agentloop/agent/features.py`).

## Layout

```
bpm/ingest/{xes,registry,audit}.py   streaming XES parser, per-log schema decisions, audit
bpm/data/{cases,encoding}.py         case objects, case-level splits, train-fit encoder + targets
bpm/models/{markov,gbm,recurrent}.py baselines + proposed model behind one interface (base.py)
bpm/{evaluate,metrics,run,report,analyze_failures}.py
agentloop/{schema,redaction}.py      trace schema v1.0.0, redaction
agentloop/envs/{base,appworld,mock}.py
agentloop/agent/{llm_agent,policy,features}.py
agentloop/{collect,build_dataset,train,evaluate}.py
configs/  results/  traces/examples/  tests/  scripts/
```

Limitations and next steps: see `DESIGN_NOTE.md` (last two sections).
