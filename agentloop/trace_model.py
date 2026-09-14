"""The connection between the two parts, made literal: an agent trajectory *is* an event log.

    episode  → case          step        → event
    app.api call (+ ok/err)  → activity   error type, #calls, parse errors → event attributes
    step timestamp           → timestamp  episode outcome → terminal pseudo-event "outcome.success|ok" / "outcome.failure|ok"

So the Part 1 sequence model (``bpm.models.recurrent.RecurrentModel``, GRU or Transformer backbone,
decomposed activity components app / api / status) is trained on agent traces unchanged, and its
next-activity head becomes the learned component of Part 2:

    P(next = "app.api|ok")  and  P(next = "app.api|err")  for every API → for a candidate action
    whose first API call is a:
        validity(a)     = P(a|ok) / (P(a|ok) + P(a|err))          "will this call error here?"
        plausibility(a) = P(a|ok) + P(a|err)                       "is this what trajectories do next?"
        success(a)      = P(outcome.success reached | prefix + a)  via rollout of the same model
    score = validity · plausibility · success   (mode selectable)

Compared with the hashed-feature logistic regression in ``agentloop/train.py`` (the cheap baseline
learner), this model sees the whole trajectory as a sequence and predicts several things at once —
the same "one state, many questions" claim as Part 1, now on agent data.

CLI:
    uv run python -m agentloop.trace_model train traces/appworld_train_v1 --out models/tracemodel_v1
    (policy: --policy tracemodel --policy-model models/tracemodel_v1 in collect/evaluate)
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

from agentloop.schema import Episode
from bpm.data.cases import Case, split_random
from bpm.data.encoding import EOS, Encoder
from bpm.models.recurrent import RecurrentModel

_API = re.compile(r"apis\.([a-z_]+)\.([a-z_]+)\(")
OUTCOME_OK, OUTCOME_FAIL = "outcome.success|ok", "outcome.failure|ok"
START = "start.episode|ok"  # pseudo-event so the model can score the *first* action too


def action_activity(code: str, error_type: str | None) -> str:
    calls = _API.findall(code or "")
    base = f"{calls[0][0]}.{calls[0][1]}" if calls else ("python.no_api" if code else "none.no_code")
    return f"{base}|{'err' if error_type else 'ok'}"


def episode_to_case(ep: Episode) -> Case:
    t0 = ep.started_at.timestamp()
    acts, ts, err_types, n_calls, prev_err = [START], [t0], ["none"], ["0"], ["none"]
    last_err = "none"
    for s in ep.steps:
        cand = s.candidates[s.chosen_index]
        code = cand.action.get("code", "")
        et = s.error.type if s.error else None
        acts.append(action_activity(code, et))
        ts.append(s.timestamp.timestamp())
        err_types.append(et or "none")
        n_calls.append(str(min(len(_API.findall(code or "")), 5)))
        prev_err.append(last_err)
        last_err = et or "none"
    ok = bool(ep.final_eval and ep.final_eval.success)
    acts.append(OUTCOME_OK if ok else OUTCOME_FAIL)
    ts.append((ep.ended_at or ep.steps[-1].timestamp).timestamp() if ep.steps else t0 + 1)
    err_types.append("none"); n_calls.append("0"); prev_err.append(last_err)
    ts = np.maximum.accumulate(np.array(ts, dtype=float))
    T = len(acts)
    return Case(ep.episode_id, acts, ts, np.zeros(T, int), np.zeros(T, int),
                {"error_type": err_types, "n_calls": n_calls, "prev_error": prev_err}, {"benchmark": ep.benchmark}, {}, True, "agent")


def load_episodes(trace_dirs: list[Path]) -> list[Episode]:
    eps = []
    for d in trace_dirs:
        for line in (d / "episodes.jsonl").read_text().splitlines():
            if line.strip():
                eps.append(Episode.from_jsonl_line(line))
    return eps


class TraceModel:
    """Trained Part 1 model over agent traces + encoder, packaged for the policy."""

    def __init__(self, model: RecurrentModel, encoder: Encoder, version: str):
        self.model, self.encoder, self.version = model, encoder, version

    def prefix_case(self, context: dict) -> Case:
        hist = context.get("history", [])
        t0 = (hist[0]["timestamp"] - 1.0) if hist else 0.0
        acts = [START] + [action_activity(h.get("code", ""), h.get("error_type")) for h in hist]
        ts = np.maximum.accumulate(np.array([t0] + [h.get("timestamp", 0.0) for h in hist], dtype=float))
        T = len(acts)
        err = ["none"] + [h.get("error_type") or "none" for h in hist]
        prev = ["none"] + err[:-1]
        nc = ["0"] + [str(min(len(_API.findall(h.get("code", "") or "")), 5)) for h in hist]
        return Case("live", acts, ts, np.zeros(T, int), np.zeros(T, int), {"error_type": err, "n_calls": nc, "prev_error": prev},
                    {"benchmark": context.get("benchmark", "appworld")}, {}, False, "agent")

    def score(self, context: dict, codes: list[str], mode: str = "product", n_roll: int = 8) -> tuple[np.ndarray, dict]:
        case = self.prefix_case(context)
        enc = self.encoder.transform(case)
        probs = self.model.predict_case(enc).next_probs[-1]
        vocab = self.encoder.act_vocab
        val, pla, suc = [], [], []
        for code in codes:
            base = action_activity(code, None).split("|")[0]
            p_ok = probs[vocab.get(base + "|ok", 1)] if base + "|ok" in vocab else 0.0
            p_err = probs[vocab.get(base + "|err", 1)] if base + "|err" in vocab else 0.0
            unk = base + "|ok" not in vocab and base + "|err" not in vocab
            val.append(0.5 if unk else (p_ok + 1e-6) / (p_ok + p_err + 2e-6))
            pla.append(probs[1] if unk else p_ok + p_err)  # UNK mass for never-seen APIs
        if mode in ("product", "success"):
            # success: append the candidate's ok-event and roll the model forward to the outcome pseudo-event
            for code in codes:
                base = action_activity(code, None).split("|")[0]
                lbl = base + "|ok" if base + "|ok" in vocab else "<UNK>"
                ext = Case("live", case.activities + [lbl], np.append(case.timestamps, case.timestamps[-1] + 1.0), np.zeros(len(case) + 1, int),
                           np.zeros(len(case) + 1, int), {k: v + ["none" if k != "n_calls" else "1"] for k, v in case.event_cat.items()}, case.case_cat, {}, False, "agent")
                e2 = self.encoder.transform(ext)
                rolls = self.model.rollout(e2, len(ext) - 1, 30, mode="sample", n=n_roll, seed=0)
                ok_id = vocab.get(OUTCOME_OK)
                suc.append(float(np.mean([ok_id in r for r in rolls])))
        else:
            suc = [1.0] * len(codes)
        val, pla, suc = np.array(val), np.array(pla), np.array(suc)
        s = {"valid": val, "plausible": pla, "success": suc, "product": val * pla * suc}[mode]
        return s, {"valid": val.tolist(), "plausible": pla.tolist(), "success": suc.tolist()}

    def save(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        self.model.save(out / "model.pt")
        self.encoder.to_json(out / "encoder.json")
        (out / "meta.json").write_text(json.dumps({"version": self.version}))

    @staticmethod
    def load(path: Path) -> "TraceModel":
        path = Path(path)
        enc = Encoder.from_json(path / "encoder.json")
        return TraceModel(RecurrentModel.load(path / "model.pt", enc), enc, json.loads((path / "meta.json").read_text())["version"])


class TraceModelPolicy:
    """Sequence world model as the learned component. Optionally *hybrid*: multiply by the
    token-level validity scorer (agentloop/train.py bundle), because the activity abstraction
    (app.api) cannot see argument-level mistakes that the code text reveals."""
    policy_id = "trace_world_model"

    def __init__(self, model_path, n_candidates: int = 3, score_mode: str = "product", validity_model: str | None = None):
        self.tm = TraceModel.load(Path(model_path))
        self.version = self.tm.version
        self.n_candidates = n_candidates
        self.score_mode = score_mode
        self.validity = None
        if validity_model:
            with open(validity_model, "rb") as fh:
                b = pickle.load(fh)
            self.validity = (b["model"], b.get("model_success"), b["featurizer"])
            self.policy_id = "trace_world_model+validity"
            self.version += "+" + b.get("version", "?")

    def choose(self, context: dict, codes: list[str]):
        if not codes:
            return 0, [], {}
        s, parts = self.tm.score(context, codes, self.score_mode)
        if self.validity is not None:
            clf, clf_s, feat = self.validity
            X = feat.transform([context] * len(codes), codes)
            pv = clf.predict_proba(X)[:, 1]
            parts["token_validity"] = pv.tolist()
            if clf_s is not None:  # token-level success head sees argument-level mistakes the API abstraction cannot
                ps = clf_s.predict_proba(X)[:, 1]
                parts["token_success"] = ps.tolist()
                pv = pv * ps
            s = s * pv
        best = int(np.argmax(s))
        return best, [float(x) for x in s], {"scores": [float(x) for x in s], "overrode_first": best != 0, "score_mode": self.score_mode, **parts}


def train(trace_dirs: list[Path], out: Path, heldout_frac: float = 0.25, seed: int = 0, backbone: str = "gru", epochs: int = 60) -> dict:
    eps = load_episodes(trace_dirs)
    cases = [episode_to_case(e) for e in eps if e.steps]
    # group split by scenario so offline metrics are on unseen scenarios
    groups = sorted({e.scenario_id or e.task_id for e in eps})
    rng = np.random.default_rng(seed); rng.shuffle(groups)
    held = set(groups[: max(1, int(len(groups) * heldout_frac))]) if len(groups) > 1 else set()
    scen = {e.episode_id: (e.scenario_id or e.task_id) for e in eps}
    tr = [c for c in cases if scen[c.case_id] not in held]
    ho = [c for c in cases if scen[c.case_id] in held]
    va = ho if ho else tr[: max(1, len(tr) // 5)]
    enc = Encoder("agent", ["error_type", "n_calls", "prev_error"], ["benchmark"], [], min_count_cat=1).fit(tr)
    etr, eva, eho = enc.transform_all(tr), enc.transform_all(va), enc.transform_all(ho)
    model = RecurrentModel(d_model=64, n_layers=1, epochs=epochs, patience=8, batch_size=16, p_attr_drop=0.1, seed=seed, backbone=backbone)
    log = model.fit(etr, eva, enc)
    rep = {"n_episodes": len(eps), "n_train_cases": len(tr), "n_heldout_cases": len(ho), "vocab": enc.n_activities, "train_log": {k: v for k, v in log.items() if k != "history"}}
    if eho:
        rep["heldout"] = offline_metrics(model, enc, eho)
    tm = TraceModel(model, enc, f"tracemodel-{backbone}-{int(np.random.default_rng(seed).integers(1e6))}")
    tm.save(out)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


def offline_metrics(model: RecurrentModel, enc: Encoder, cases) -> dict:
    """Next-API top-1, validity AUROC (P(err) of the true next call), and outcome prediction via rollout."""
    from sklearn.metrics import roc_auc_score
    preds = model.predict_cases(cases)
    top1, y_err, p_err, y_ok, p_ok = [], [], [], [], []
    id2 = enc.id2act
    for c, pr in zip(cases, preds):
        for t in range(len(c) - 1):
            y = c.next_act[t]
            if y < 0:
                continue
            base = id2[y].split("|")[0]
            top1.append(float(pr.next_probs[t].argmax() == y))
            if base.startswith("outcome."):
                continue
            po = pr.next_probs[t][enc.act_vocab.get(base + "|ok", 1)]
            pe = pr.next_probs[t][enc.act_vocab.get(base + "|err", 1)]
            y_err.append(int(id2[y].endswith("|err"))); p_err.append(pe / (po + pe + 1e-9))
        # outcome: probability mass on outcome.success at the last real step vs truth
        y_ok.append(int(c.activities[-1] == OUTCOME_OK)); p_ok.append(float(pr.next_probs[len(c) - 2][enc.act_vocab.get(OUTCOME_OK, 1)]) if len(c) >= 2 else 0.5)
    out = {"next_api_top1": float(np.mean(top1)) if top1 else None, "n_steps": len(top1)}
    if len(set(y_err)) == 2:
        out["validity_auroc"] = float(roc_auc_score(y_err, p_err))
    if len(set(y_ok)) == 2:
        out["outcome_auroc_final_step"] = float(roc_auc_score(y_ok, p_ok))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("trace_dirs", nargs="+"); t.add_argument("--out", required=True); t.add_argument("--backbone", default="gru")
    t.add_argument("--epochs", type=int, default=60); t.add_argument("--seed", type=int, default=0); t.add_argument("--heldout-frac", type=float, default=0.25)
    a = ap.parse_args(argv)
    rep = train([Path(d) for d in a.trace_dirs], Path(a.out), a.heldout_frac, a.seed, a.backbone, a.epochs)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
