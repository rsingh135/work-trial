# Design note — sequential modelling of process traces and a reusable agent-trace loop

*Companion documents: `PROGRESS.md` (dated decision log with reasoning), `README.md` (exact commands), `results/SUMMARY.md` (all numbers), `results/audit/SUMMARY.md` (data audit).*

## 0. The connection between the two parts

Both parts are the same problem at different scales: a system emits a **partial history** (a case prefix; an agent trajectory prefix), a model keeps a **state** that summarises it, and several **downstream questions** are asked of that state (what happens next, when, how does it end; which action is valid, will the episode succeed). Part 1 studies the modelling question in a setting with ground truth and clean evaluation; Part 2 builds the plumbing that turns live interaction into exactly the prefix→target examples Part 1 consumes, and closes the loop by putting the learned predictor back into the process that generated the data. The trace schema of Part 2 was designed so that an AppWorld episode *is* an event log: `(episode, step, action, result, error, timestamp)` maps onto `(case, event index, activity, attributes, outcome, timestamp)`, and the same "prefix at position t → targets" construction (`bpm/data/encoding.py`, `agentloop/build_dataset.py`) is used in both.

## 1. Part 1 — modelling a partially observed process

### 1.1 Data audit and task formulation
Eight XES logs were ingested with a streaming parser that keeps every attribute as a string and parses only `time:timestamp` (`bpm/ingest/xes.py`); how each log is *read* is a hand-written registry with reasons per field (`bpm/ingest/registry.py`), and the audit (`bpm/ingest/audit.py`) computes what the design claims. Findings that shaped the design:

* **Observation per event.** Activity label (BPI2013: `concept:name|lifecycle:transition` — the log's own classifier; BPI2020: `concept:name`), timestamp, low-cardinality event attributes (`org:role`, BPI2013 `org:group`, `impact`, `organization involved`), high-cardinality attributes kept but frequency-thresholded to `<UNK>` (`org:resource` 1440 values, `product` 704), and case-start attributes only when known at case start (`Amount`, `RequestedAmount`, organisational unit). `org:role` is 11–36 % UNKNOWN/UNDEFINED depending on the log; it is kept as an explicit category, not imputed.
* **Same name, different semantics.** `org:resource` is a person name in BPI2013 and an anonymised role token (`STAFF MEMBER`/`SYSTEM`) in BPI2020; Open Problems has a misspelled `oranization country` key that exists nowhere else. The registry treats every log separately.
* **Censoring is activity-level, not just log-level.** BPI2013 Incidents is a *snapshot*: 96 % of cases start in Apr–May 2012 and 98 % of `Closed` events fall in May 2012 (batch auto-close); 25 % of cases end at `Completed|In Call` with no close. International Declarations has 593 cases ending at `End trip`, PermitLog 991 at `Send Reminder`. Open Problems is right-censored by definition. **Rule:** a case is *complete* iff its last activity is in the registry's terminal set; suffix / remaining-time / outcome targets are built only from complete cases, next-activity and Δt targets from every prefix.
* **Leakage measured, not guessed.** Normalised mutual information between case attributes and the terminal activity: PermitLog `dec_id_*`/`DeclarationNumber_*` = 1.0 (populated as declarations arrive), `TotalDeclared`/`Overspent*` are end-of-case aggregates, International `AdjustedAmount` is post-hoc, BPI2020 event `id` encodes the sub-process (`st_step`/`dd_`/`rv_`). All excluded.
* **Time irregularity.** `Start trip`/`End trip` are date-only (100 % local midnight) so their Δt is quantised; mixed +01/+02 offsets → all clock features computed in `Europe/Amsterdam`; 1 % zero-Δt events; Δt spans 8 orders of magnitude → modelled on `log1p(seconds)`.

**Targets from a prefix of length t** (positions 0..t observed): next activity including `<EOS>`; `log1p` seconds to the next event; seconds to case end (complete cases); the activity suffix to the end. **State update:** `h_t = GRU(h_{t-1}, enc(e_t))`, `h_0 = f(case-start attributes)` — one O(1) update per arriving event, no replay (`RecurrentModel.step`).

### 1.2 Evaluation protocol
Case-level splits only (asserted disjoint, fingerprinted and persisted as JSON). Vocabularies, component tables, categorical thresholds and time scalers are fitted on the training split. **In-distribution:** seeded 70/15/15 random split. **Shift 1:** chronological split by case start (train earliest, test latest; the fraction of train cases still open when the test window starts is recorded — 67 % for Incidents because of the snapshot structure, ≤10 % for BPI2020). **Shift 2:** cross-log transfer RequestForPayment → DomesticDeclarations (zero-shot through a union label vocabulary; few-shot fine-tune on 5 % of target cases vs. training from scratch on the same 5 %). All five BPI2020 logs run through the pipeline (`permit_random`, `prepaid_random` configs); substantive experiments use **Domestic** (large, 90 % of cases in 5 variants — tests calibration and time modelling) and **International** (longer, permit + trip prefix, 753 variants — tests multi-step prediction and long-history retention). Metrics: NLL / accuracy / macro-F1, MAE and median AE in hours, normalised Damerau–Levenshtein similarity, exact match and valid-termination rate, ECE (raw and temperature-scaled on validation) and Brier with reliability bins, 95 % case-level bootstrap CIs and three seeds.

**Published BPI2013 results.** The bundle's `results/PUBLISHED_BPI2013_RESULTS.md` transcribes Rama-Maneiro et al. (arXiv:2009.13251) with its protocol: 5-fold CV over cases, 80/20 train/validation inside the training folds, an end-of-case token appended to every trace, trace attributes excluded, fold means. I reproduced that protocol exactly (`split: cv`, `assume_complete: true`, `configs/published_*`, ten runs) and report it separately from my own splits in `results/PUBLISHED_COMPARISON.md` (§1.5). One difference is stated rather than hidden: my GBM and GRU use event-level attributes, so the closest published comparator is "Theis, with attributes"; my Markov baseline is activity-only and lands inside the published range, which is a sanity check on the protocol reproduction. BPI2020 has no canonical table (bundle note); Impedovo et al. 2023 is cited as context only.

### 1.3 Baselines
1. **k-order Markov** (k = 2 with back-off to k = 1, 0): transition tables for the next activity, median Δt per context, median remaining time per (last activity, position). Greedy/sampled rollouts from the same tables.
2. **Gradient-boosted trees** (`sklearn` HistGradientBoosting, no native deps): one row per prefix position with last-3 activities, bag-of-activities, the 7 time features, current event attributes and case attributes; three boosters (next activity, Δt, remaining). Autoregressive rollout by re-featurising the synthetic prefix.
Both use the identical split, encoder and target construction as the proposed model.

### 1.4 Proposed architecture (implemented in full, `bpm/models/recurrent.py`)
* **Event encoder.** `enc(e_t) = MLP([E_act[a_t]; Σ_j E_comp[comp_j(a_t)]; E_f[attr_f,t] ∀f; W_time·time_t])`, `time_t` = standardised `log1p Δt`, `log1p elapsed`, sin/cos hour, sin/cos weekday, `log1p position`. The **decomposed components** (`obj|ACTION|ACTOR` for BPI2020, `status|substatus` for BPI2013) live in shared embedding tables so a label never seen in training still has a partial representation.
* **History compression.** 2-layer GRU, d = 128, `h_0` from case-start attributes.
* **Heads and objectives.** Next activity: softmax over the vocabulary *including `<EOS>`* (end-of-case is a class, so termination is coherent by construction). Δt: 3-component Gaussian mixture on `log1p` seconds (a log-normal mixture in seconds) → mixture NLL; remaining time: Gaussian on `log1p` seconds → NLL. Total `CE + 0.5·NLL_Δt + 0.5·NLL_rem`, weights fixed before any test run. **Attribute dropout** (p = 0.15) replaces event attributes by `<UNK>` during training so rollouts, where future attributes are unknown, are in-distribution.
* **Multi-step.** Autoregressive rollout from `h_t`: choose an activity (greedy or sampled), sample/median Δt from the mixture, feed back the activity, its components, `<UNK>` attributes and time features derived from the predicted Δt. Prefixes of different lengths are packed so hundreds of rollouts advance in lock-step.
* **Uncertainty.** Categorical distribution + entropy for the next activity; mixture quantiles give a Δt 80 % interval; post-hoc temperature scaling fitted on validation.
* **Schema differences / unseen values.** Per-log registry → encoder; unseen labels → `<UNK>` whole-label embedding but live components; unseen attribute values → `<UNK>`; transfer mode pre-registers target labels in the output layer (rows untrained until fine-tuning).
* **Costs.** 214 k parameters; training on a laptop CPU takes seconds per epoch on the largest log (a full-scale config with three seeds runs in minutes); inference is one GRU step per event (~0.2 ms) plus a 30-iteration bisection for mixture quantiles; state is 2×128 floats per open case.

Proposed but not implemented: uncertainty-weighted multi-task loss; a Transformer variant with KV-cache as the state (O(t) memory) for comparison; hazard-style remaining-time head with explicit censoring likelihood (currently censored cases are masked, which is unbiased only if censoring is independent of the process state — not true for the Incidents snapshot).

### 1.5 Results and analysis
*(all numbers from the final batch — 4 models × 14 configs, 3 seeds; every metric with CIs in `results/SUMMARY.md`; git sha in each results JSON)*

**Headline: next-activity NLL on the test split** (± = std over 3 seeds where run):

| log / split | Markov k=2 | GBM | GRU | Transformer |
|---|---|---|---|---|
| Domestic, random | 0.324 | 0.303 | 0.305 ± .002 | 0.317 |
| Domestic, chronological (strict) | 0.291 | 0.279 | **0.249** ± .008 | 0.271 |
| International, random | 0.583 | 0.318 | 0.318 ± .001 | 0.323 |
| International, chronological (strict) | 0.52 | 0.30 | 0.30 | 0.30 |
| Incidents, random | 1.022 | **0.693** | 0.759 ± .011 | 0.741 |
| Incidents, chronological (leaky: 67 % of train cases overlap the test window) | 1.080 | 0.875 | **0.794** ± .025 | 0.791 |
| Incidents, chronological **strict** (train truncated at cutoff: 3,477 of 5,287 cases) | 1.101 | 1.349 | 0.812 (seeds: 0.81 / 1.13 / 1.22) | **0.799** (0.80 / 0.85 / 0.82) |
| Closed Problems | 1.032 | 1.123 | **0.954** ± .009 | 1.02 |
| Open Problems (censored; next-event only) | 1.305 | 1.295 | **1.151** ± .010 | 1.17 |
| PermitLog | 0.818 | 0.488 | **0.479** | 0.49 |
| PrepaidTravelCost | 0.527 | **0.304** | 0.328 | 0.33 |

1. **In distribution, a stateless prefix-feature GBM is as good as a learned state.** GBM and the two sequence models are within CI on the large BPI2020 logs; the GBM wins on attribute-rich Incidents (0.693 vs 0.741/0.759) and Prepaid; the sequence models win on Permit (longest, most variable cases) and on the two small BPI2013 logs where the GBM overfits.
2. **Under a *strict* temporal shift the state wins decisively.** On Incidents with train histories truncated at the cutoff (the honest protocol; the non-strict split leaks 67 % of train cases into the test window), GBM degrades +95 % in NLL (0.693 → 1.349, accuracy 0.77 → 0.57, suffix DL 0.57 → 0.28) while the GRU degrades +7 % and the Transformer +8 %, and the Transformer's suffix similarity *rises* to 0.595. On Domestic strict, GRU 0.249 vs GBM 0.279. (All BPI2020 chronological NLLs *improve* vs random because the 2018 rollout process is more regular than the 2017 pilot — the bundle's log catalog documents this process change.) One caveat: with truncated training the GRU's seed variance explodes (0.81–1.22; the validation CE is identical across seeds, the spread is in how each seed handles end-of-case on a test set that is 57 % complete while 66 % of training cases lost their ending) — the Transformer is steadier (0.80–0.85).
3. **Robustness to attribute-schema removal (the attribute-dropout claim, now measured).** Re-evaluating each trained model with every event attribute masked: Incidents GBM 0.693 → 1.169 (+69 %), GRU 0.759 → 0.934 (+23 %), Transformer 0.741 → 0.941. Without attributes the sequence models beat the tree. On BPI2020 logs attributes carry little (role/anonymised resource) and nothing moves.
4. **Decoding matters more than architecture for multi-step coherence.** Greedy rollouts loop on Incidents (`Accepted|In Progress` is 46 % of events): 35 % of GRU suffixes never terminate within 50 steps (DL 0.456). Taking the *medoid* of 5 sampled continuations terminates 100 % of the time and lifts DL to 0.571 (GRU) / 0.569 (Transformer), above the GBM's 0.567. On the BPI2020 logs greedy and medoid are within 0.01.
5. **Time.** Δt MAE is tail-dominated (Domestic MAE 40 h vs median AE 2.7 h); GBM (absolute-error loss) and the mixture median are within CI everywhere and beat the naive per-activity median by ~30 %. The mixture's 80 % interval covers 77–82 % of true Δt on every log.
6. **Calibration.** GRU raw ECE ≤ 0.02 on all large logs without temperature scaling (GBM needs scaling: Closed 0.162 → 0.046). Under strict shift the GRU keeps ECE 0.040 on Incidents where the GBM goes to 0.155.
7. **Transfer (RfP → Domestic) with a fair budget is neutral, and the factorised head helps a little.** With the same 60-epoch budget as scratch, fine-tuning the RfP model on 5 % of Domestic cases gives 0.364 vs 0.366 from scratch (the earlier "negative" result was a 15- vs 60-epoch artefact, corrected). The component-factorised output head improves zero-shot from 5.75 to 5.15 NLL (accuracy 0.50 → 0.54, suffix DL 0.58 → 0.64) and fine-tune to 0.348 — real but small: RfP→Domestic shares actions/actors but Domestic is easy enough that 367 cases suffice from scratch.
8. **Published protocol (5-fold CV, `results/PUBLISHED_COMPARISON.md`).** Closed Problems accuracy GBM 70.7 / GRU 68.1 / Markov 63.1 % vs best published 64.0 %; Incidents 76.7 / 76.0 / 59.0 % vs 74.7 %; Incidents remaining MAE 11.2 / 11.7 days vs 12.4; suffix DL 0.54 / 0.52 vs 0.53. The activity-only Markov baseline lands inside the published range, which validates the protocol reproduction; the learned models' margin comes largely from event attributes.

**Downstream questions from the same state** (`results/<run>/downstream.md`, GRU 3-seed ensemble):

| run | accuracy at 100 / 80 / 50 % coverage | SLA-breach AUROC (T = train median / p90) | ensemble MI (nats) | predictive entropy |
|---|---|---|---|---|
| Domestic random | 0.891 / 0.932 / 1.000 | 0.846 / 0.702 | 0.005 | 0.289 |
| Domestic chrono strict | 0.915 / 0.912 / 0.999 | 0.866 / 0.660 | 0.001 | 0.325 |
| International random | 0.883 / 0.958 / 0.989 | 0.938 / 0.906 | 0.005 | 0.285 |
| International chrono strict | 0.894 / 0.961 / 0.991 | 0.912 / 0.903 | 0.007 | 0.314 |
| Incidents random | 0.765 / 0.857 / 0.968 | 0.753 / 0.735 | 0.029 | 0.662 |
| Incidents chrono strict | 0.715 / 0.825 / 0.960 | 0.616 / 0.679 | 0.026 | 0.888 |

* *Selective prediction works*: abstaining on the 20 % most uncertain prefixes lifts Incidents accuracy from 0.77 to 0.86, and the remaining half of predictions are 96–100 % correct on every log. This is the operator-facing use of calibration.
* *SLA-breach risk* from the Laplace head is a usable classifier on the declaration logs (AUROC 0.85–0.94 at the median threshold) and weak on Incidents (0.75 → 0.62 under shift), where the batch auto-close makes the end time nearly independent of the case.
* *Seed-ensemble mutual information does not detect the shift*: it stays ≈ 0.005 on Domestic and ≈ 0.027 on Incidents across random / chronological / strict splits, while total entropy rises (0.66 → 0.89 on Incidents). Three seeds of one architecture agree even when they are jointly wrong — a negative result for cheap epistemic uncertainty here.
* *Anomaly score* (per-case mean NLL) flags abandoned and multiply-rejected cases at the top on every log, but its Spearman correlation with case length ranges from 0.04 to 0.79 across runs, so it needs length-conditioning before it is an alert.

**Failure analysis** (`results/<run>/failure_analysis*.md`, GRU and Transformer):

1. *Errors compound identically for both state designs.* Per-step error along greedy suffixes rises monotonically (International 0.12 → 0.51 over 8 steps; Permit 0.14 → 0.87 over 12); after the first wrong event 90–93 % of the rest of the suffix is wrong for both GRU and Transformer. The model commits to a plausible alternative variant and never recovers; sampling recovers the exact suffix in 91 % of Domestic prefixes vs 77 % greedy.
2. *Coherence.* Termination is coherent where the process has a real end (post-terminal continuation 2–4 % on BPI2020, EOS precision/recall ≥ 0.99), and fails on the Incidents snapshot under greedy decoding (fixed by medoid decoding, above). Remaining-time predictions are non-monotone along a case at 13–33 % of positions — each position is predicted independently.
3. *Long histories: the GRU forgets, the Transformer remembers, and it does not matter.* The state-separation probe (KL between next-event distributions with and without the first informative event in the prefix) decays for the GRU from 0.20–0.26 nats at distance 1–2 to ≤ 0.01 beyond 3 events on every log; the Transformer retains 0.03–0.04 nats at 6–10+ events (Incidents: 0.027 vs 0.002; International: 0.043 vs 0.001). Yet accuracy conditional on distance since that event is the same for both (Incidents 3–5 events: 0.60 vs 0.58). The information an attention state keeps about an old rejection is real but not predictive in these logs: the remaining error is dominated by events triggered outside the log (checked directly: after `FINAL_APPROVED by SUPERVISOR`, 0.9 % of 9,145 prefixes are rejected next, with no prefix variant or amount separating them — the 99 % majority rule is the ceiling).
4. *Concrete cases* are listed in the per-run files (e.g. Domestic `declaration 86632`: model 0.95 on `Request Payment`, true next `REJECTED by MISSING`; Δt interval [5.9, 226] h contains the true 119 h).

## 2. Part 2 — the trace → train → agent loop

### 2.1 Harness
`Env` protocol (`task_ids/split`, `reset`, `step(action dict)`, `action_schema`, `evaluate`, `version`) → `LLMAgent` (one Python code block per turn; the prompt assumes nothing about the benchmark beyond that) → `Policy` (baseline: first sample; reranker: N samples scored by the learned component) → `Episode` (pydantic v2, schema 1.0.0) → redaction at serialisation → JSONL + action-schema store + manifest. AppWorld is driven **out of process** over its HTTP server from a separate venv (it pins pydantic<2), which also guarantees that the collector never imports a benchmark. A `MockEnv` + scripted LLM exercise the whole path offline, and a canned-action client dry-runs the real AppWorld adapter without an API key (this is how password redaction was verified on genuine output).

### 2.2 Trace contents
Every step keeps: env/benchmark/task/scenario/episode/run ids; split; versions (env + data version, model id, prompt hash, harness git SHA, policy id + version, seed); step index and timestamp; the observation the agent saw; a hash reference to the full action/tool schema; *all* sampled candidates with raw text, parsed action, API calls, per-candidate token usage and policy score, plus the chosen index; the tool result; a typed error (syntax / api / runtime / parse / llm / budget); retry index; the env's success flag; per-step usage (tokens incl. cache reads/writes, cost, latency); termination reason; the evaluator's pass/fail list. Failed actions are ordinary steps. Training examples carry a `Provenance` (trace file, episode, step, candidate, schema and builder versions). Redaction rules are versioned and counted per episode (README).

### 2.3 Learned component and its role
**Action scorer**: `P(valid | context, code)` where valid = the executed code raised no error, plus a secondary head `P(episode success | context, code)`. Features are hashed n-grams over the code, the previous error, the last observation, API-call tokens crossed with "previous step errored", and a few numeric features; the model is logistic regression (seconds to train, deterministic, no vocabulary to ship). Reinsertion: the reranker samples N = 3 candidates from the same LLM at the same temperature and executes the arg-max of `P(valid)·P(success)`; ties fall back to the first sample, so an uninformative model reduces exactly to the baseline.

**Why one label per step.** AppWorld can checkpoint databases but not the Python shell namespace, so "counterfactually" executing the unchosen candidates would let variables leak between them and corrupt labels. Collection therefore uses the baseline agent (one sample per step); unexecuted candidates at deployment are recorded with their scores but unlabelled.

### 2.4 Hygiene
Collection on a subset of AppWorld `train` tasks (whole scenarios, all variants); the offline metric is measured on *held-out scenarios* within that set; the baseline-vs-reranked comparison runs on `dev` tasks never used for training or any prompt/hyper-parameter decision; `test_*` is never touched. Train and dev share no scenario ids (verified: 30 vs 19 scenarios, overlap 0).

### 2.5 Results
**Offline loop (mock environment, `make demo-part2`, no API key):** 48 collected episodes → 143 examples (validation rate 1.00, 18 % invalid actions) → group-held-out AUROC 1.00 for validity, 0.96 for success → on held-out dev tasks over 3 runs:

| policy | task-goal completion | scenario-goal completion | invalid-action rate | steps / episode |
|---|---|---|---|---|
| baseline (1 sample) | 0.72 | 0.17 | 0.233 | 3.31 |
| reranker, validity only | 0.33 | 0.08 | 0.024 | 2.33 |
| reranker, validity × success | **1.00** | **1.00** | 0.019 | 3.00 |

**AppWorld (real LLM):** *pending — requires `ANTHROPIC_API_KEY`; `scripts/run_part2_appworld.sh` runs collection on 45 train tasks, training, and the baseline-vs-reranked comparison on 30 dev tasks × 2 runs, and fills this section from `results/part2_appworld_v1/summary.json`.* The adapter itself has been exercised end-to-end with canned actions (3 tasks, 4 steps each, validation rate 1.00, 22 redactions per episode, `api_error` correctly labelled on a bad login).

**Offline vs. end-to-end (mock environment, before any API spend).** A validity-only scorer cut the invalid-action rate from 23 % to 2 % but *halved* task success (72 % → 33 %): the scripted agent's "bad" options include `complete_task(answer='wrong')`, which never errors, so the scorer preferred an early wrong completion over a failing lookup. Adding the success head and scoring by the product restored 100 % task success at a 2 % invalid rate. Validity is necessary, not sufficient; a scorer must see a progress signal.

## 3. Limitations
* Bundle docs / published BPI2013 results were not received; the comparison with published numbers is protocol-described, not executed.
* Remaining-time predictions are not constrained to be monotone along a case (13–33 % of consecutive positions increase); censoring is handled by masking, not by a likelihood, and under the strict split the GRU's seed variance shows that truncated training histories are a real problem for a fixed-size state.
* Seed-ensemble mutual information did not detect distribution shift; a proper epistemic estimate (deep ensembles across architectures, or a density model on the state) is future work.
* Cross-log transfer is limited to one pair; PermitLog-involving pairs would leak (the log catalog documents that PermitLog embeds declarations and requests).
* Part 2 learned component is linear on hashed features — deliberately cheap; it cannot represent long-range plan state. AppWorld numbers are on a small task subset with two runs, so run-to-run variation is wide.
* Cost accounting uses list prices; the harness records raw token counts so it can be recomputed.

## 4. Prioritised next steps
1. Hazard/survival remaining-time head with censoring likelihood; monotone remaining time via Δt-rollout expectation.
2. Transformer-with-KV-cache variant to test whether the GRU's fixed-size state is what loses long-history information (the state-separation probe in `failure_analysis.md` is the diagnostic).
3. Part 2: replace the hashed-feature scorer with a small fine-tuned encoder over (context, code); add a progress-estimation head trained from the evaluator's per-test passes rather than a single episode label.
4. Counterfactual candidate labelling by forking the whole environment process (not just the DBs) so every sampled candidate gets a validity label.
5. A second adapter (e.g. a JSON-tool-call benchmark) to prove the interfaces empirically rather than by construction.
