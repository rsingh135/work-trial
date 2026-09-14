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

## 2026-09-14 — Part 1 pipeline complete; Part 2 loop built offline

**Part 1 code** (`bpm/`): case loader with per-log schema → train-fit encoder → three models behind one interface (`SequenceModel`: `predict_case`, `rollout`, `rollout_many`) → one evaluator (`bpm/evaluate.py`) → `bpm/run.py` config runner writing `results/<name>.json`. Demo (`configs/demo.yaml`) runs in ~25 s.

Bugs caught by end-to-end demo (documented because they would have silently corrupted results):
- pandas 3 stores tz-aware timestamps at µs resolution; my `astype("int64")/1e9` gave epoch-*milliseconds*, so every Δt was 1000× too small (MAE "0.1 h" on a log whose median Δt is ~40 h). Fixed with an explicit `total_seconds()` and a regression test.
- sklearn HGB treats my `-1` "no earlier event" marker as a categorical value out of range → NLL 7.6. Fixed with NaN (native missing support).
- sklearn HGB single-row `predict_proba` inside a rollout loop thrashed 18 threads (18k calls). Fixed with lock-step batched rollouts (`rollout_many`) — evaluation of 1000 prefixes × 6 rollouts now takes seconds.

**DECISION: completeness rule + EOS token.** End-of-case is predicted as an `<EOS>` class in the next-activity head (no separate Bernoulli head) — one distribution, coherent by construction. Suffix/remaining-time targets only from complete cases (terminal set per log). Incidents `In Call`-ending cases, International `End trip`-ending cases etc. are therefore *incomplete* and excluded from those targets.

**DECISION: decomposed activity components.** Activity embedding = whole-label embedding + Σ component embeddings (`Declaration|APPROVED|SUPERVISOR`, or BPI2013 `status|substatus`). Unseen labels → UNK whole-label but still carry action/actor components. This is the mechanism tested in the RfP→Domestic transfer experiment.

**DECISION: attribute dropout (p=0.15)** during training so rollouts with unknown future attributes (UNK) are in-distribution.

**Loss weights fixed a priori**: CE + 0.5·MDN-NLL(Δt) + 0.5·Gaussian-NLL(remaining). Not tuned on test.

**Part 2 code** (`agentloop/`): pydantic v2 schema v1.0.0, redaction, `Env` protocol, AppWorld HTTP adapter (server auto-launched from `.venv-appworld`), mock env, LLM agent (Anthropic + scripted client), policies, collect/build/train/evaluate CLIs. 12 tests pass, incl. an end-to-end collect→build→train→reinsert test on the mock env.

**DECISION: label only the executed action.** AppWorld `save_state/load_state` checkpoints DBs but not the Python shell namespace, so "counterfactually executing" unchosen candidates would leak variables between candidates and corrupt labels. So: collection uses the baseline (1 sample/step, cheapest), each step yields one (context, code) → valid example; the reranker samples N=3 only at deployment. Unexecuted candidates are still recorded in the trace with their scores.

**DECISION: raise_on_failure=True** in the AppWorld adapter — otherwise API failures (401 etc.) come back as ordinary output and would be labelled *valid*.

**Finding (mock env, before any API spend): validity ≠ progress.** A validity-only reranker cut the invalid-action rate 23%→2% but *halved* task success (72%→33%) because "complete_task(answer='wrong')" never errors. Added a second head P(success | context, action) trained on the episode-level label; scoring mode `product` restored TGC to 100% on the mock. This is the concrete illustration of "offline metric improves, end-to-end does not" that the brief asks for; the AppWorld run will report all three modes.

**Cost plan for AppWorld (needs ANTHROPIC_API_KEY)**: `claude-haiku-4-5`, prompt caching on the system prompt, max 25 steps, outputs truncated to 2500 chars. Collection: ~45 train tasks × 1 run ≈ $5–8; eval: 30 dev tasks × 2 policies × 2 runs, reranker at N=3 ≈ $10–15. Hard `--cost-budget` guard in the collector.

## 2026-09-14 (later) — Batch run hung; root causes fixed

Symptoms: the full-scale batch sat on `domestic_random` GBM for 8 hours; a concurrent demo run also stalled. Alone, the same GBM stage takes 13 s.
- **Cause 1 — OpenMP oversubscription.** torch and sklearn each load an OpenMP runtime; with default "all 18 cores" per library and a second process on the box, sklearn's HistGradientBoosting calls degraded from seconds to effectively hanging. **Fix:** `OMP_NUM_THREADS=8` set in `bpm/run.py` before imports, `torch.set_num_threads(8)`; and never run two heavy Part 1 processes concurrently (documented in README).
- **Cause 2 — per-row sklearn calls.** `predict_case` was called once per case (3k tiny `predict_proba` calls). **Fix:** `predict_cases` batched interface (one call per split) for both GBM and GRU.
- **Cause 3 — gradient-free batches.** After the NaN guard, a length-sorted batch made only of single-event *incomplete* cases has no targets at all → loss is a constant without `grad_fn` → `backward()` raises. **Fix:** skip such batches. (Single-event cases still contribute at eval time as prefixes with no target — i.e. not at all — which is correct.)
Net effect: full Domestic (10.5k cases) demo config end-to-end in 7 s; a full-scale config with 3 seeds in a few minutes.

**Coherence finding surfaced by the failure-analysis script on the demo:** predicted remaining time is *not monotone* along a case (only ~60% of consecutive positions decrease) because each position is predicted independently from the state. Candidate fix (not implemented): predict remaining time as Δt-head expectation summed over a rollout, or add a monotonicity penalty. Recorded as a limitation.

## 2026-09-14 — First full batch: three more fixes, then a clean rerun

First batch (all 11 configs, minutes each after the thread fix) surfaced:
- Sampled rollouts: a mixture tail sample gave Δt = e^hundreds → inf timestamp → NaN logits. **Fix:** clamp sampled log-Δt to log1p(10 years).
- sklearn caps categorical cardinality at 255; Incidents `org:group` has 472 train values. **Fix:** GBM feeds such fields as ordinal ints (documented baseline weakness).
- Fully censored Open Problems has no remaining-time targets → GBM regressor crashed. **Fix:** skip the head; predictions NaN → evaluator reports "no complete cases".
- **Model-quality bug (the important one).** On PermitLog the remaining-time Gaussian head's validation NLL *diverged* (1.06 → 5.6) while next-activity CE was still improving (0.68 → 0.58); early stopping on the total loss stopped at epoch 7, leaving the GRU at NLL 0.686 vs GBM 0.488. Cause: heteroscedastic Gaussian with a small σ floor collapses on the training tail. **DECISION:** remaining-time head is now a **Laplace** likelihood on log1p seconds (heavy-tailed; point prediction = median, which matches the MAE metric) with scale floor e^-1; MDN σ floor raised to e^-2. Permit GRU: NLL 0.479, DL 0.805, ECE 0.017; training stable to epoch 45 with LR decay. Loss weights unchanged (0.5/0.5).
Because this changes the GRU everywhere, **all Part 1 numbers are from the rerun with this code** (git sha in each results JSON). Pre-fix observations kept for the record: Domestic random — all three models saturate (NLL 0.30–0.32); Domestic chrono — GRU 0.264 < GBM 0.278 < Markov 0.290; International chrono — GRU 0.287 < GBM 0.298 < Markov 0.522; Closed Problems (1.5k cases) — Markov best (1.03), GRU overfits (1.11).

## 2026-09-14 — Part 1 full results (clean rerun, git sha in each JSON)

Tables: `results/SUMMARY.md`; figures: `results/figures/`; failure analyses: `results/{domestic_random,international_random,incidents_random,permit_random}/failure_analysis.md`. Design-note §1.5 has the narrative. What I take from it:
- **Where the recurrent state earns its keep: temporal shift.** Incidents random→chrono NLL: GRU +5%, GBM +26%. Domestic chrono: GRU 0.249 vs GBM 0.278. The prefix-feature GBM is tied or better in-distribution on the big logs, and better on attribute-rich Incidents (0.693 vs 0.759).
- **Small logs**: GRU best on Closed (0.954 vs Markov 1.032, GBM 1.123) and Open (1.151 vs 1.30) — the GBM overfits with ~1k cases.
- **Calibration**: GRU raw ECE ≤0.02 on every large log without temperature scaling; Δt 80% interval coverage 0.77–0.82.
- **Transfer is negative**: fine-tune-from-RfP (0.476) < scratch (0.366) on 5% Domestic. Reason: per-label output layer; shared components only on the input side. Next step: component-factorised output head. Kept in the report as a real negative result rather than dropped.
- **Failure modes**: compounding (93–95% of suffixes stay wrong after the first wrong step); Incidents greedy rollouts loop on `Accepted|In Progress` (35% fail to terminate within 50 steps); recurrent state forgets an early rejection within ~3 events (KL probe 0.2–0.27 → ≤0.01).

**Ablation launched** (`results/ablations/`): Incidents GRU with attribute dropout 0 → does the GBM gap on Incidents come from the p=0.15 attribute masking?

**Not done / consciously left**: uncertainty-weighted losses, Transformer variant, hazard remaining-time head, more transfer pairs. Listed in DESIGN_NOTE §3–4.

**Ablations on Incidents (GRU vs GBM gap, `results/ablations/`)**: attribute dropout 0 → NLL 0.758 (no change); d_model 256 / dropout 0.2 / lr 1e-3 → see log line below. Neither closes the gap to GBM (0.693), so it is not masking or capacity; the likely cause is the GBM's direct access to the *current* event's 472-value `org:group` and `product` as splittable features vs. 8-dim embeddings squeezed through a 128-dim state. Left as a documented limitation.
    [ablation_incidents_gru_d256] gru_multihead seed=0 fit=60s  nll=0.765 acc=0.742 dtMAE=27.0h remMAE=278.6h DL=0.470 ece=0

## 2026-09-14 — Reference bundle received (`researcher_work_trial_bundle/`); now the single source of truth

Checked: `shasum -c CHECKSUMS.sha256` passes; all eight XES files are **byte-identical** to the copies I had been using, so no Part 1 number changes because of the data. Old `bundle/` removed; `bpm/ingest/registry.py` points at the bundle paths. AppWorld `verify tests` (100% pass) and `verify tasks` (147 tasks) run clean with the pinned `0.1.3.post1`, matching `docs/APPWORLD_SETUP.md`.

What the bundle changes / confirms:
- `docs/LOG_CATALOG.md` confirms my choices: official activity classifier = `concept:name + lifecycle:transition` (7/13 activities); `oranization country` typo; Open Problems right-censored; high-card ids need train-only vocab. It also warns that the five BPI2020 logs are *related views* (PermitLog embeds declarations and requests) and that 2017 (pilot) vs 2018 (rollout) is a real process change — which is exactly what my chronological splits picked up (later period more regular). My transfer pair RfP→Domestic does not share case ids (distinct declaration/request objects), but I now state in the design note that PermitLog-involving transfer would leak.
- `results/PUBLISHED_BPI2013_RESULTS.md` gives the exact protocol (Rama-Maneiro et al.: 5-fold CV over cases, 80/20 train/val inside training folds, EOS appended to every trace, trace attributes excluded, mean over folds, accuracy %, DL similarity, remaining-time MAE in **days**). My random 70/15/15 split is *not* that protocol, so no comparison was made before. **DECISION:** add a `cv` split type + `assume_complete` (EOS on every trace, as the benchmark does — including the 25% of Incidents cases that end at `In Call`) and run all five folds for Closed Problems and Incidents with all three models (`configs/published_*`, `scripts/run_published_bpi2013.sh`). Differences that remain and are stated: my models use event attributes (the benchmark's "Theis with attributes" row is the closest comparator), and the benchmark reports the mean of fold-level *per-prefix* accuracies over all prefixes incl. the EOS position — same as mine.
- `results/BPI2020_RESULTS_NOTE.md`: no canonical BPI2020 table; Impedovo et al. 2023 report shallow > Bi-LSTM under a 3-event window. Consistent with my finding that GBM ties the GRU in-distribution. Their numbers are cited as context only, not compared.
- `docs/APPWORLD_SETUP.md`: split policy (train for collection, dev for the paired comparison, never test) matches what I built; SGC requires complete variant groups — my task selector already picks whole scenarios; "unique experiment_name + task_id" — my adapter uses `harness_<run_id>`; adds "collateral changes / evaluator assertions passed" to the metrics list → the evaluator's raw pass/fail list is already stored per episode, summarised into the Part 2 report.
- Reading guide: the BPI2013 dataset guide (§1.1–1.4) and BPI2020 attribute explanation are consistent with the registry; nothing to change.

## 2026-09-14 — Published-protocol reproduction (BPI2013, 5-fold CV)

`results/PUBLISHED_COMPARISON.md`. Fold means: Closed acc GBM 70.7 / GRU 68.1 / Markov 63.1 % vs best published 64.0 %; Incidents acc 76.7 / 76.0 / 59.0 % vs 74.7 %; Incidents remaining MAE 11.2 / 11.7 days vs 12.4; suffix DL Incidents 0.54 / 0.52 vs 0.53. Markov (activity-only) inside the published range on both logs → protocol reproduction sane. The learned models' margin over the published rows comes largely from event attributes, stated in the note; not framed as SOTA. Published fold files excluded from `results/SUMMARY.md` (separate report).

## 2026-09-14 — Self-review as the assigner ("what would an exceptional candidate do?")

Prompted by the reviewer's question whether the task is "pretty easy". Framing: a world-model startup for enterprise traces needs (i) one state that answers many questions, (ii) robustness to vocabulary/process drift, (iii) online updating, (iv) a flywheel from agent traces. Basic submission = LSTM + random split + accuracy + offline classifier. Solid = what I had. Exceptional = prove properties of the *state*. Gaps I found in my own work and the plan (in execution order):
1. Central claim under-tested → add a causal-Transformer backbone (attention over prefix = KV-cache state) behind identical heads; same forgetting probe. Compare GRU vs Transformer vs stateless GBM.
2. Chronological split leaks (67% Incidents train cases overlap test window) → `strict` chrono option: truncate train cases at cutoff, mark incomplete.
3. Robustness asserted not shown → attribute-removal test (all event attrs → UNK at test time), GRU vs GBM.
4. Online update asserted not shown → streaming test: incremental `step()` state == batch forward; streaming demo.
5. Uncertainty unused → selective prediction curve, seed-ensemble epistemic uncertainty under shift, SLA-breach probability P(remaining > T) with AUROC/Brier, per-case anomaly score (NLL) with long-case confound check. All from the same state.
6. Greedy decoding loops on Incidents → sample-medoid decoding.
7. Transfer negative for a mechanical reason (per-label output rows) → component-factorised, input-tied output head; rerun RfP→Domestic; unseen-label rate on 2017→2018 chrono tests.
8. Part 2 learner is a throwaway → treat agent traces as event logs (episode=case, app.api=activity with object/action components, error=attribute) and use the Part 1 model class as the learned component (next-API, P(error), P(success), remaining steps). LogReg stays as the cheap baseline learner.
