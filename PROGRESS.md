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

## 2026-09-13 — Bundle received; environment; data audit

**Bundle contents actually received**: the 8 XES logs only (dropped in repo root, moved to `bundle/logs/`). Not received: dataset docs, paper list, published BPI2013 results + protocol, AppWorld notes. Proceeding from public knowledge; will compare to published BPI2013 numbers only if I can reconstruct the exact protocol from the papers (Tax et al. 2017 / Evermann et al. 2017 / Camargo et al. 2019 all report on BPI2013 with *different* splits). Flagged to user.

**Environment issues + decisions**
- `appworld` pins `pydantic<2`; my trace schema wants pydantic v2. **DECISION:** AppWorld lives in its own venv (`.venv-appworld`, Python 3.11) and the harness talks to it out-of-process (subprocess/HTTP adapter). Side benefit: forces the env interface to be truly benchmark-agnostic — the collector never imports AppWorld.
- LightGBM needs Homebrew `libomp` on macOS. **DECISION:** use `sklearn.ensemble.HistGradientBoosting*` instead — same model class, zero system deps, reviewer can `uv sync` and go.
- Python 3.12 via uv; torch 2.14 with MPS available.

**XES ingestion** (`bpm/ingest/xes.py`): lxml `iterparse` streaming, all attrs kept as strings with `ev_`/`case_` prefixes, only `time:timestamp` parsed. All 8 logs parse in <2 s each; case/event counts match the published dataset descriptions exactly (e.g. Incidents 7554/65533, Domestic 10500/56437).

**Audit findings that change the design** (`results/audit/SUMMARY.md`, per-log JSON):
1. **BPI2013 Incidents is a snapshot, not a stream.** 96% of cases start in Apr–May 2012; 5621/5716 `Completed|Closed` events occur in May 2012 (batch auto-close). 25% of cases end at `Completed|In Call` with no Closed event. ⇒ a chronological split within Incidents has a compressed time axis (still done, by case start), remaining-time is dominated by the auto-close lag, and `In Call`-ending cases must be treated as *incomplete*.
2. **Open Problems**: 43% of cases actually end with `Completed|Closed`; per the brief still treated as fully right-censored (a "Closed" open problem can be reopened). Next-event targets on observed prefixes are kept; suffix/remaining/outcome masked.
3. **BPI2020 censoring is activity-level, not log-level**: International has 593 cases ending at `End trip`, Permit has 991 ending at `Send Reminder` and 453 at `End trip` — process not finished. Only 1–3 cases per log end within 30 days of the log end, so time-censoring is negligible.
   **DECISION (uniform completeness rule):** a case is *complete* iff its last activity ∈ the registry's `terminal` set (per log, derived from the audit's last-activity distribution). Suffix / remaining-time / outcome targets are built only from complete cases; next-activity and Δt targets from every prefix. Open Problems: `censored=True` ⇒ never complete.
4. **Leakage confirmed by measurement**: PermitLog `dec_id_*`, `DeclarationNumber_*` have NMI=1.0 with the terminal activity (they are populated as declarations arrive); `TotalDeclared/Overspent*` are end-of-case aggregates; International `AdjustedAmount` is post-hoc. All excluded in `bpm/ingest/registry.py` with reasons. BPI2020 event `id` encodes sub-process origin (`st_step`/`dd_`/`rv_`) → excluded.
5. **Time irregularity**: `Start trip`/`End trip` are date-only (100% midnight local) → their Δt is artificially quantised; hour-of-day feature is meaningless for them. 1% zero-Δt events in International/Permit. Timestamp offsets mix +01/+02 (DST) → all time features computed in `Europe/Amsterdam` local time.
6. **High cardinality**: `org:resource` 1440 values (Incidents), `product` 704, `org:group` 649. Handled via train-fit vocab + min-frequency → UNK. In BPI2020 `org:resource` is anonymised to {STAFF MEMBER, SYSTEM} — trivially low-card; same field name, completely different semantics across families (the brief's warning, observed).
7. **Missingness**: `org:role` 11–36% UNKNOWN/UNDEFINED depending on log → kept as an explicit category, not imputed.
8. **Schema quirk**: `oranization country` (sic) exists only in Open Problems. Excluded (as is its correctly-spelled sibling).
9. **Variants**: Domestic/RfP top-5 variants cover ~90% of cases (easy); Permit 32%, Incidents 41% (hard). ⇒ substantive BPI2020 experiments on **Domestic** (large, simple, near-deterministic — tests calibration/time) and **International** (longer, permit+trip prefix, 753 variants — tests multi-step and long-history retention); **RfP→Domestic** transfer as the cross-log shift (shared org, disjoint activity strings that share structure `X APPROVED by Y`).
