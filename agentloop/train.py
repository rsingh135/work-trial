"""Train the action-validity scorer from built examples; report offline metrics on the
group-held-out split.

    uv run python -m agentloop.train datasets/validity --out models/validity_v1

Model: logistic regression on hashed features (seconds on CPU, deterministic). The bundle
(pickled featurizer + classifier + metadata) is what ``RerankerPolicy`` loads — that is the
reinsertion point.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from agentloop.agent.features import Featurizer
from agentloop.schema import TrainingExample


def load(path: Path) -> list[TrainingExample]:
    return [TrainingExample.model_validate_json(l) for l in path.read_text().splitlines() if l.strip()]


def _xy(feat: Featurizer, exs: list[TrainingExample]):
    ctx = [{**e.context, "step_index": int(e.context.get("step_index", 0)), "n_prev_errors": int(e.context.get("n_prev_errors", 0))} for e in exs]
    X = feat.transform(ctx, [e.action.get("code", "") for e in exs])
    y = np.array([e.label_valid for e in exs])
    ys = np.array([-1 if e.label_success is None else e.label_success for e in exs])
    yp = np.array([-1 if e.label_progress is None else e.label_progress for e in exs])
    yr = np.array([-1 if e.label_regress is None else e.label_regress for e in exs])
    return X, y, ys, yp, yr


def offline_metrics(y, p) -> dict:
    out = {"n": int(len(y)), "pos_rate": float(y.mean()) if len(y) else None}
    if len(y) and 0 < y.mean() < 1:
        out.update(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)), logloss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
                   accuracy=float(((p > 0.5) == y).mean()), majority_accuracy=float(max(y.mean(), 1 - y.mean())))
    return out


def train(data_dir: Path, out: Path, C: float = 1.0, seed: int = 0) -> dict:
    t0 = time.time()
    tr, ho = load(data_dir / "train.jsonl"), load(data_dir / "heldout.jsonl")
    feat = Featurizer()
    Xtr, ytr, str_, ptr, rtr = _xy(feat, tr)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(Xtr, ytr)
    rep = {"n_train": len(tr), "n_heldout": len(ho), "C": C, "seed": seed, "featurizer_version": feat.version,
           "train_metrics": {"valid": offline_metrics(ytr, clf.predict_proba(Xtr)[:, 1])}}
    # secondary head: P(episode succeeds | context, action) — a noisy "progress" signal (episode-level label
    # assigned to every step), used to break the validity scorer's blindness to *harmful-but-valid* actions
    clf_s = None
    m = str_ >= 0
    if m.sum() and 0 < str_[m].mean() < 1:
        clf_s = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed).fit(Xtr[m], str_[m])
        rep["train_metrics"]["success"] = offline_metrics(str_[m], clf_s.predict_proba(Xtr[m])[:, 1])
    # dense heads from per-step evaluator deltas (progress) and collateral/premature-done (regress); also fitted on
    # counterfactual candidates when the traces contain them
    def _fit(yv, name):
        m = yv >= 0
        if m.sum() and 0 < yv[m].mean() < 1:
            c = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed).fit(Xtr[m], yv[m])
            rep["train_metrics"][name] = offline_metrics(yv[m], c.predict_proba(Xtr[m])[:, 1])
            return c
        return None
    clf_p, clf_r = _fit(ptr, "progress"), _fit(rtr, "regress")
    rep["n_counterfactual_train"] = int(sum(1 for e in tr if e.source == "counterfactual"))
    if ho:
        Xho, yho, sho, pho, rho = _xy(feat, ho)
        rep["heldout_metrics"] = {"valid": offline_metrics(yho, clf.predict_proba(Xho)[:, 1])}
        for name, c, yv in (("success", clf_s, sho), ("progress", clf_p, pho), ("regress", clf_r, rho)):
            m = yv >= 0
            if c is not None and m.sum() and 0 < yv[m].mean() < 1:
                rep["heldout_metrics"][name] = offline_metrics(yv[m], c.predict_proba(Xho[m])[:, 1])
    rep["train_seconds"] = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    version = f"logreg-h{feat.version}-{int(t0)}"
    with open(out / "model.pkl", "wb") as fh:
        pickle.dump({"model": clf, "model_success": clf_s, "model_progress": clf_p, "model_regress": clf_r, "featurizer": feat, "version": version,
                     "data_dir": str(data_dir), "report": rep}, fh)
    rep["version"] = version
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(json.dumps(train(Path(a.data_dir), Path(a.out), a.C, a.seed), indent=1))


if __name__ == "__main__":
    main()
