# Design note — sequential modelling of process traces and a reusable agent-trace loop

*Concise version (~4 pages). Every number here has a source file under `results/`; the long-form appendix with all tables is `docs/DESIGN_NOTE_FULL.md`, the dated decision log is `PROGRESS.md`.*

## 0. One problem at two scales

A system emits a **partial history** (a case prefix; an agent trajectory prefix); a model keeps a **state** that summarises it; several **questions** are asked of that state. Part 1 studies the modelling question where ground truth is clean (eight BPI event logs). Part 2 builds the plumbing that turns live agent interaction into the same prefix→target examples and puts the learned predictor back into the agent. The link is literal: an AppWorld episode is converted into a Part 1 `Case` (API call = event, error = attribute, outcome = terminal event) and the Part 1 model is one of the two learned components reinserted into the agent (`agentloop/trace_model.py`).

## 1. Part 1 — a compact case state that answers several questions

**Data audit → decisions.** A streaming XES parser keeps every column raw; `bpm/ingest/audit.py` computes what the note claims; `bpm/ingest/registry.py` records per log which fields are activity, input, excluded (with reasons) and which activities are real endings. Findings that shaped the design: BPI2013 Incidents is a *snapshot* (96 % of cases start in six weeks of 2012, 98 % of `Closed` events bulk-closed in May) and 25 % of its tickets never close; International has 593 cases ending at `End trip`; PermitLog columns `dec_id_*`/`TotalDeclared` have mutual information 1.0 with the outcome (populated after the fact, excluded); `Start/End trip` are date-only; `org:resource` is a person in BPI2013 and an anonymised token in BPI2020. **Rule:** a case is complete iff its last activity is in the log's terminal set; suffix and remaining-time targets come only from complete cases, next-event targets from every prefix; Open Problems is censored wholesale.

**Targets from a prefix of t+1 events:** next activity (with an explicit `<EOS>` class), log-seconds to the next event, seconds to the end, and the activity suffix.

**Model (`bpm/models/recurrent.py`).** Event encoder = concat[activity embedding; Σ embeddings of the label's *decomposed components* (`Declaration|APPROVED|SUPERVISOR`, or `status|substatus`); one embedding per attribute with rare values → UNK; 7 time features on log scale in local time] → MLP → 128-d. Two interchangeable state designs behind identical heads: a 2-layer **GRU** (fixed-size note, O(1) update per event, initial state from case-start attributes) and a 2-layer causal **Transformer** (attention over all past events; the KV-cache is the state). Heads: softmax over activities incl. `<EOS>` (termination is coherent by construction), a 3-component Gaussian mixture on log Δt (median + 10–90 % interval), a Laplace on log remaining time (median point estimate; survival function gives SLA-breach risk). Loss = CE + 0.5·NLL_Δt + 0.5·NLL_rem, weights fixed before any test run. Attributes are randomly blanked (p = 0.15) during training so rollouts with unknown future attributes are in-distribution. Multi-step prediction feeds predictions back in; two decoders: greedy and the *medoid* of 5 samples. 214 k parameters; seconds per epoch on a laptop CPU; a streaming `step()` is tested equal to the batched forward.

**Baselines with identical preprocessing:** a k = 2 Markov table with back-off and per-context median times; gradient-boosted trees (sklearn) over last-3 activities, activity counts, time features and current attributes, with autoregressive rollout.

**Protocol.** Case-level splits, vocabularies and scalers fitted on training only, fingerprinted split files. Random 70/15/15; chronological by case start in a *leaky* variant (whole cases kept; 67 % of Incidents train cases overlap the test window) and a **strict** variant (train events after the cutoff deleted, truncated cases marked incomplete); cross-log transfer RfP → Domestic; the published 5-fold CV protocol reproduced exactly for BPI2013. Metrics: NLL / accuracy / macro-F1; MAE and median AE in hours with interval coverage; normalised Damerau–Levenshtein suffix similarity, exact match, termination rate; ECE and Brier raw and temperature-scaled; case-level bootstrap CIs, three seeds; and an attribute-removal probe (all attributes masked at test time).

**Results** (next-activity NLL, test split; full tables in `results/SUMMARY.md`):

| log / split | Markov | GBM | GRU | Transformer |
|---|---|---|---|---|
| Domestic random / strict-chrono | 0.324 / 0.291 | 0.303 / 0.279 | 0.305 / **0.249** | 0.317 / 0.271 |
| International random / strict-chrono | 0.583 / 0.52 | 0.318 / 0.30 | 0.318 / 0.30 | 0.323 / 0.30 |
| Incidents random / leaky-chrono / **strict-chrono** | 1.02 / 1.08 / 1.10 | **0.693** / 0.875 / 1.349 | 0.759 / 0.794 / 0.812 | 0.741 / 0.791 / **0.799** |
| Closed / Open (censored) / Permit / Prepaid | 1.03 / 1.31 / 0.82 / 0.53 | 1.12 / 1.30 / 0.49 / **0.30** | **0.95 / 1.15 / 0.48** / 0.33 | 1.02 / 1.17 / 0.49 / 0.33 |

1. *In distribution a stateless tree ties a learned state* on the large logs and beats it on attribute-rich Incidents; the state models win on the small logs and the longest-case log.
2. *Under an honest temporal shift the state wins decisively*: Incidents strict, GBM +95 % NLL (accuracy 0.77 → 0.57, suffix similarity halved), GRU +7 %, Transformer +8 % with the best suffix similarity (0.595). The leaky split hides this (GBM +26 %).
3. *Robustness to attribute removal*: Incidents GBM +69 %, GRU +23 %; the state models then beat the tree.
4. *Decoding beats architecture for coherence*: greedy loops on Incidents (35 % of rollouts never terminate); medoid decoding terminates 100 % and lifts suffix similarity 0.456 → 0.571, above the GBM.
5. *Calibration*: GRU raw ECE ≤ 0.02 on every large log; Δt 80 % intervals cover 77–82 %. Selective prediction lifts Incidents accuracy 0.77 → 0.86 at 80 % coverage. Seed-ensemble mutual information does **not** detect shift (kept as a negative result).
6. *Published protocol* (5-fold CV): Closed Problems accuracy GBM 70.7 / GRU 68.1 % vs best published 64.0 %; Incidents 76.7 / 76.0 % vs 74.7 %; Incidents remaining-time MAE 11.2 / 11.7 days vs 12.4. The activity-only Markov baseline lands inside the published range, validating the reproduction; the margin comes largely from event attributes.
7. *Transfer*: fine-tune ≈ scratch (0.364 vs 0.366) at equal budget; a component-factorised output head improves zero-shot 5.75 → 5.15 NLL. Small but real.

**Failure analysis** (`results/<run>/failure_analysis*.md`). Errors compound: after the first wrong event 90–93 % of the rest of a greedy suffix is wrong, for both state designs. The GRU forgets an early rejection within ~3 events (KL probe → ≤ 0.01 nats); the Transformer remembers it (0.03–0.04 at 6–10 events) yet is no more accurate: the residual error is caused by information outside the log (after `FINAL_APPROVED`, 99.1 % of 9,145 prefixes proceed normally and nothing in the prefix separates the rest). Remaining-time predictions are non-monotone at 13–33 % of positions.

## 2. Part 2 — trace → dataset → model → agent, benchmark-agnostic

**Harness.** One environment contract (`agentloop/envs/base.py`: `task_ids`, `reset`, `step`, `action_schema`, `evaluate`, `version`) with actions as typed dicts. AppWorld runs as its own server in its own venv and is driven over HTTP, so nothing in the loop imports the benchmark; a 60-line mock world implements the same contract and the whole pipeline runs on it offline and in tests. The agent (Claude, one code block per turn) proposes N actions; a **policy** picks one; the environment executes it; the evaluator is called after every step.

**Traces** (`agentloop/schema.py`, v1.1.0). One validated, redacted JSON line per episode: ids, split, versions (env, model, prompt hash, git sha, policy, seed), and per step the observation, *every* candidate (raw reply, parsed action, tokens, score, counterfactual outcome), the executed one, result, typed error, per-step checker output and reward, cost. Failed actions stay. Secrets become salted hashes (rules documented in the README). A training example carries provenance to (file, episode, step, candidate). Validation rate 1.00 across all ~800 real episodes.

**Learners** (the reinsertion seam is `policy.choose(context, candidates)`): a hashed-n-gram logistic scorer over the candidate's code with heads *valid / success / progress / regress*, and the Part 1 model trained on the trajectories (validity = P(api|ok)/(P(ok)+P(err)), plausibility, success by rollout). Rule-based **gates** from the published schema (API exists; no "done" before a productive call) serve as controls. Following FDM-1, labels come from the environment: the checker after every step (dense progress), and every candidate executed in a **forked** copy (database checkpoint + Python-namespace rollback, 0.3 s for three; verified equal to replay) → counterfactual labels for all candidates (5.5 k examples from 90 episodes) and an **oracle** policy that picks the candidate the checker rewards, the ceiling for any learned picker.

**Results, Claude Haiku 4.5, 30 held-out dev tasks × 2 runs** (`results/part2_appworld_v*_*/summary.json`):

| version | baseline | gates only | learned picker | hybrid | oracle |
|---|---|---|---|---|---|
| v1 (one label / step) | 3.3 % | — | 6.7 % | 3.3 % | — |
| v2 (dense labels, forks, gates) | 3.3 % | 3.3 % | 6.7 % | 3.3 % | **5.0 %** |
| v3 (+ harness fixes) | 1.7 % | 0.0 % | 0.0 % | 3.3 % | 1.7 % |
| **v4 Opus 5**: raw → world-frame → +picker → +oracle | 90.0 % → 96.7 % | — | 96.7 % | — | 100 % |

Offline the learners are strong (held-out scenarios: validity AUROC 0.97, progress 0.99, regress 1.00); end-to-end nothing moves (differences of 1–2 successes in 60). The oracle explains why: even with the real environment as judge, best-of-3 completes 5 %, because in 1,239 of 1,283 steps *no* sampled candidate advances the checker. The candidate generator, not the picker, is the bottleneck. The offline-vs-online gap the brief asks about appeared twice in the data: a validity-only picker prefers a bare `complete_task()` because it never errors (19/60 episodes end within two steps), and the API-level model cannot see argument mistakes the text model can. Harness bugs found in the traces (truncated replies at my 1,024-token cap, hand-retyped passwords) were fixed in v3 and helped on training tasks (credential errors 95 → 4) but not on dev.

**v4: change the generator (Claude Opus 5), with and without a PERSIST-style world-frame.** `agentloop/agent/worldframe.py` keeps a persistent structured state (apps, known APIs with signatures, token variables, entities fetched, last errors, notes) seeded from the task text and the published schema, rendered every turn in place of the long raw-output window (arXiv:2603.03482 applied to tool use). Collection on the 90 train tasks: 70/90 solved (Haiku: 11/90), 9.2 steps, 0.5 invalid actions per episode. Same 30 held-out dev tasks × 2 runs:

| policy (Opus 5) | TGC % | SGC % | invalid % | steps | input tok/step | $ / episode |
|---|---|---|---|---|---|---|
| raw history (prompt v3) | 90.0 ± 0.0 | 90.0 | 4.2 | 9.8 | 3,296 | 0.196 |
| **world-frame (prompt v4)** | **96.7 ± 0.0** | 90.0 | 9.5 | 9.5 | 2,278 | **0.142** |
| world-frame + gates + progress picker (3 samples) | 96.7 ± 0.0 | 90.0 | 4.9 | 8.8 | — | 0.399 |
| world-frame + gates + **oracle lookahead** (3 samples, forked) | **100.0 ± 0.0** | **100.0** | 5.8 | 8.6 | — | 0.393 |

The generator was the whole Haiku story: the same harness, tasks and seeds go from ≤ 6.7 % to 90 % by swapping the model. The persistent state then adds four successes in 60 (it rescues an entire scenario the raw agent fails in all six attempts), cuts the context by a third and the cost per episode by 28 %, at the price of more exploratory errors (9.5 % vs 4.2 % invalid actions). On top of that agent the learned picker keeps completion at 96.7 % while halving invalid actions (9.5 → 4.9 %) and shortening episodes (9.5 → 8.8 steps), for 2.8× the cost; the environment-forking oracle reaches 100 % of tasks and scenarios. Read together with Haiku: selection among candidates is worth exactly the gap between the generator's menu and the checker's optimum — nothing when the menu is empty (Haiku, ceiling 5 %), efficiency when the menu is good (Opus), and the last two scenarios only with the environment in the loop. Zero run-to-run variance in every Opus arm; total recorded API spend for Part 2, all versions: $270.

## 3. Limitations

Part 1's proposed model does not beat a well-built tree in-distribution; its advantage is robustness. Censoring is handled by masking, not by a likelihood, and the GRU's seed variance under strict training shows the cost. Cross-log transfer covers one pair. Part 2's learned components are deliberately small and did not change task success with Haiku; the AppWorld numbers are on 30 tasks × 2 runs, so ±3 points is noise. Cost estimates were optimistic (learned-policy episodes cost 2–4× the baseline). The published-results comparison uses event attributes the paper's systems mostly did not.

## 4. Next steps, in order

1. Finish v4 and report Opus raw vs world-frame vs picker vs oracle on the same tasks.
2. Survival-style remaining-time head so truncated histories are modelled, not masked.
3. Argument-aware events for the trace world model; a learned next-state predictor as a fork-free progress estimator.
4. Shift-sensitive epistemic uncertainty (cross-architecture ensembles or a density model on the state).
5. Continual monthly retraining on the time stream; cost per *solved* task as the headline agent metric.
6. A second environment adapter to prove the contract empirically.
