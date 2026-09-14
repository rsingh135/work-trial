# Progress & Decision Log

Running log of what was done, what was decided, and why. Newest entries at the bottom. Brief lives in `instructions.md` (never edited).

Conventions: each entry = date, what, **why**. Decisions that change results get a `DECISION:` tag so a reviewer can grep them.

---

## 2026-09-13 — Kickoff

**Setup answers from reviewer/user:**
- Reference bundle will be dropped into `work_trial/bundle/`.
- Agent LLM: Anthropic API (`claude-haiku-4-5` for collection + eval; cheap, stochastic, sufficient for a baseline agent).
- Notes: markdown in repo (this file + `DESIGN_NOTE.md` + `README.md`).
- Infra: local git + private GitHub repo; compute on local M5 Pro / 48 GB. No cloud GPU.

**DECISION: Python 3.12 via `uv`, not the system 3.14.** torch / lightgbm / appworld wheels lag 3.14; one `pyproject.toml` + `uv.lock` is the reproducible env spec.

**DECISION: Stack.** PyTorch for the proposed model, LightGBM + numpy for the tabular baseline, `lxml` streaming for XES (logs are up to ~65k events; streaming keeps memory flat and avoids pm4py's heavy import), pydantic v2 for the trace schema, pytest.

**DECISION: Order of work.** Part 1 data audit first (it determines feature choices for everything downstream), AppWorld install in the background (it is the slowest external dependency and might surface blocking issues early).

**Plan file**: `~/.claude/plans/i-have-this-work-noble-karp.md` (copied into repo as `PLAN.md` for the reviewer).
