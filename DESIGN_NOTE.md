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

**Published BPI2013 results.** The bundle received contained the logs only (no published-results file or protocol). The published BPI2013 next-event numbers I know of (Evermann et al. 2017; Tax et al. 2017 derivatives; Camargo et al. 2019) use different prefix conventions and splits from each other, so a like-for-like comparison would be a guess; I report my Incidents / Closed Problems numbers with the exact protocol so a reader with the packaged results can align them. See the limitations section.

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
*(filled from `results/SUMMARY.md`; see that file for every number with CIs)*

**Headline numbers** (test split, NLL of the next activity with 95 % case-bootstrap CI; ± = std over 3 seeds; full table with every metric in `results/SUMMARY.md`):

| log / split | Markov k=2 | GBM | GRU (proposed) | GRU suffix DL-sim | GRU ECE raw |
|---|---|---|---|---|---|
| Domestic, random | 0.324 | 0.303 | 0.305 ± 0.002 | 0.937 | 0.008 |
| Domestic, **chronological** | 0.290 | 0.278 | **0.249 ± 0.008** | 0.944 | 0.038 |
| International, random | 0.583 | 0.318 | 0.318 ± 0.001 | 0.888 | 0.013 |
| International, **chronological** | 0.522 | 0.298 | 0.298 ± 0.005 | **0.906** | 0.020 |
| Incidents (BPI2013), random | 1.022 | **0.693** | 0.759 ± 0.011 | 0.456 | 0.020 |
| Incidents, **chronological** | 1.080 | 0.875 | **0.794 ± 0.025** | **0.525** | **0.037** |
| Closed Problems (BPI2013) | 1.032 | 1.123 | **0.954 ± 0.009** | 0.598 | 0.077 |
| Open Problems (censored; next-event only) | 1.305 | 1.295 | **1.151 ± 0.010** | – | 0.090 |
| PermitLog | 0.818 | 0.488 | **0.479** | 0.835 | 0.017 |
| PrepaidTravelCost | 0.527 | **0.304** | 0.328 | 0.916 | 0.014 |

* **In distribution** the GBM and the GRU are statistically tied on the two large BPI2020 logs (CIs overlap; Domestic is saturated — even the Markov model reaches 0.32 because 90 % of cases follow five variants). The GBM wins on the attribute-rich Incidents log (0.693 vs 0.759) and on Prepaid; the GRU wins on Permit (the log with the longest, most variable cases), and clearly on the two small BPI2013 logs where the GBM overfits (Closed: 0.954 vs 1.123).
* **Under temporal shift the GRU degrades least.** Incidents random→chronological: GRU +5 % NLL, GBM +26 %, Markov +6 %; the GRU's suffix similarity *rises* (0.456→0.525) while the GBM's halves (0.567→0.289). Domestic chronological: GRU 0.249 vs GBM 0.278 vs Markov 0.290 with the largest macro-F1 (0.665). Every model's calibration deteriorates under shift (Domestic ECE 0.008→0.04–0.05 for all three); the GRU keeps the lowest ECE on Incidents-chrono (0.037 vs 0.075). Note that all three models *improve* in NLL on the BPI2020 chronological splits: the later period of the declaration process is more regular than the earlier one, so "chronological" is a shift in both directions.
* **Time.** Δt MAE is dominated by the heavy tail (Domestic: MAE 40 h, median AE 2.7 h); GBM (trained with an absolute-error loss) and GRU (mixture median) are within CI everywhere; both beat the naive per-activity median by ~30 %. The GRU's 80 % predictive interval covers 77–82 % of true Δt on every log — the mixture head is usably calibrated.
* **Calibration.** Raw ECE of the GRU is ≤ 0.02 on all large logs without temperature scaling; scaling helps the GBM (Closed: 0.162→0.046) far more than the GRU, i.e. the GRU's probabilities are already close to honest.
* **Cross-log transfer (RfP → Domestic) is a negative result.** Zero-shot NLL 5.75 (output rows for Domestic labels are untrained; accuracy at prefix length 1 is 0 because `Declaration SUBMITTED by EMPLOYEE` never occurs in RfP). Fine-tuning on 5 % of Domestic cases (367) reaches 0.476, *worse* than training from scratch on the same 367 cases (0.366). The shared trunk is anchored to RfP statistics and 15 epochs at half learning rate do not undo it, while Domestic is easy enough that 367 cases suffice. The decomposed-component embeddings help only on the *input* side; the output layer is still per-label. The fix (next step 2) is a component-factorised output head.

**Failure analysis** (`results/<run>/failure_analysis.md`, GRU seed 0; four logs):

1. *Errors compound.* Per-step error along greedy suffixes rises monotonically — International 0.12 → 0.51 over 8 steps, Permit 0.14 → 0.87 over 12 — and once the first wrong event is emitted, 93–95 % of the remaining suffix is wrong (mean post-divergence error 0.86–0.94 vs 0.41–0.65 overall). The rollout does not recover; it commits to a plausible alternative variant. Sampling five suffixes recovers the exact suffix in 91 % of Domestic prefixes vs 77 % greedy, so the *distribution* is informative even when the mode is wrong.
2. *Coherence.* Termination is coherent where the process has a clear end: post-terminal continuation 2–4 % on BPI2020, 0 % on Incidents, EOS precision/recall ≥ 0.99. But on Incidents the GRU **fails to terminate** 35 % of greedy rollouts (mean predicted suffix 19.8 events vs 8.0 true; valid-termination 0.655 vs GBM 1.000): `Accepted|In Progress` is 46 % of all events, and its self-transition stays the arg-max at every step, so greedy decoding loops on the modal status. The failure is specific to greedy decoding on a snapshot log where the true end (`Closed`) is a batch event weakly tied to the case history. Remaining-time predictions are non-monotone along a case at 13–33 % of positions (each position is predicted independently; nothing enforces "one event later means less time left").
3. *Long histories are forgotten.* The state-separation probe — KL between the next-event distribution with and without the first informative event (a rejection / queueing / reminder) in the prefix — decays from 0.20–0.27 nats when the event is 1–2 steps back to ≤ 0.01 at 3–5 steps and ≈ 0 beyond, on every log. Accuracy conditional on distance since that event drops correspondingly on Incidents (0.71 at 1–2 → 0.60 at 3–5). The GRU state carries a short effective memory: it summarises "where the case is now", not "what happened to it". This is the concrete argument for the attention-based variant (next step 2), which can look back at the rejection directly.
4. *Concrete cases* (in the per-run files): e.g. Domestic `declaration 86632` — prefix `SUBMITTED → APPROVED by PRE_APPROVER → FINAL_APPROVED`, model puts 0.95 on `Request Payment`; true next is `REJECTED by MISSING` followed by a full re-approval loop. Nothing in the prefix distinguishes this case (the rejection is triggered by information outside the log); the model is *confidently wrong for the right reason*, and its Δt interval [5.9, 226] h does contain the true 119 h.

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
* Remaining-time predictions are not constrained to be monotone along a case (≈60 % of consecutive positions decrease on Domestic); censoring is handled by masking, not by a likelihood.
* The chronological split on Incidents is weak (snapshot log); cross-log transfer is limited to one pair.
* Part 2 learned component is linear on hashed features — deliberately cheap; it cannot represent long-range plan state. AppWorld numbers are on a small task subset with two runs, so run-to-run variation is wide.
* Cost accounting uses list prices; the harness records raw token counts so it can be recomputed.

## 4. Prioritised next steps
1. Hazard/survival remaining-time head with censoring likelihood; monotone remaining time via Δt-rollout expectation.
2. Transformer-with-KV-cache variant to test whether the GRU's fixed-size state is what loses long-history information (the state-separation probe in `failure_analysis.md` is the diagnostic).
3. Part 2: replace the hashed-feature scorer with a small fine-tuned encoder over (context, code); add a progress-estimation head trained from the evaluator's per-test passes rather than a single episode label.
4. Counterfactual candidate labelling by forking the whole environment process (not just the DBs) so every sampled candidate gets a validity label.
5. A second adapter (e.g. a JSON-tool-call benchmark) to prove the interfaces empirically rather than by construction.
