"""Downstream questions answered from the *same* trained state, the way an operator would use them:

  1. Selective prediction — accuracy vs. coverage when abstaining on the most uncertain prefixes.
  2. Epistemic uncertainty — seed-ensemble disagreement (mutual information); reported per run so
     in-distribution vs. shifted test sets can be compared.
  3. SLA-breach risk — P(remaining time > T) from the Laplace head (closed form), scored as a
     classifier with AUROC / Brier at T = train median and train p90 of remaining time.
  4. Anomaly score — per-case mean next-activity NLL; top anomalous variants; case-length confound
     (Spearman) and agreement with a Markov anomaly score.

    uv run python -m bpm.downstream results/international_random.json [--model gru_multihead]
→ results/<name>/downstream.{md,json}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score

from bpm.data.cases import Split, load_cases
from bpm.data.encoding import Encoder
from bpm.ingest.registry import get_schema
from bpm.models.markov import MarkovModel
from bpm.models.recurrent import RecurrentModel, laplace_sf
from bpm.run import truncate, truncate_at_cutoff


def _entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(-1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("--model", default="gru_multihead")
    a = ap.parse_args(argv)
    r = json.loads(Path(a.results_json).read_text())
    name = r["config"]["name"]
    d = Path("results") / name
    schema = get_schema(r["log"])
    encoder = Encoder.from_json(d / "encoder.json")
    split = Split.from_json(d / "split.json")
    cases = truncate(load_cases(schema, max_cases=r["config"].get("max_cases")), r["config"].get("max_len"))
    if r["config"].get("assume_complete"):
        for c in cases:
            c.complete = True
    by = {c.case_id: c for c in cases}
    tr = [by[i] for i in split.train]
    if split.meta.get("strict") and split.meta.get("cutoff_ts"):
        tr = truncate_at_cutoff(tr, split.meta["cutoff_ts"])
    enc_tr, enc_te = encoder.transform_all(tr), encoder.transform_all([by[i] for i in split.test])
    paths = sorted(d.glob(f"{a.model}_seed*.pt"))
    models = [RecurrentModel.load(p, encoder) for p in paths]
    out: dict = {"run": name, "model": a.model, "n_seeds": len(models)}
    L = [f"# Downstream uses of the state — {name} / {a.model} ({len(models)} seeds)", ""]

    # ---------------- gather per-seed predictions
    preds = [m.predict_cases(enc_te) for m in models]
    P, Y, G, RP, RT, RG = [], [], [], [], [], []
    for i, c in enumerate(enc_te):
        m = c.next_act >= 0
        P.append(np.stack([pr[i].next_probs[m] for pr in preds]))  # [S, n, V]
        Y.append(c.next_act[m]); G += [c.case_id] * int(m.sum())
        mm = np.isfinite(c.remaining) & (np.arange(len(c)) < len(c) - 1)
        if mm.any():
            RP.append(np.stack([pr[i].remaining_params[mm] for pr in preds]))  # [S, n, 2]
            RT.append(c.remaining[mm]); RG += [c.case_id] * int(mm.sum())
    P = np.concatenate(P, 1); Y = np.concatenate(Y); G = np.array(G)
    p_mean = P.mean(0)
    correct = (p_mean.argmax(1) == Y).astype(float)
    ent = _entropy(p_mean)
    mi = ent - _entropy(P).mean(0)  # mutual information = epistemic part of the entropy

    # ---------------- 1. selective prediction
    order = np.argsort(ent)
    sel = {}
    for cov in (1.0, 0.9, 0.8, 0.7, 0.5, 0.3):
        k = int(len(order) * cov)
        sel[f"{int(cov * 100)}%"] = float(correct[order[:k]].mean())
    single = (P[0].argmax(1) == Y).astype(float)
    out["selective_prediction"] = {"ensemble_accuracy_at_coverage": sel, "single_model_accuracy": float(single.mean()), "ensemble_accuracy": float(correct.mean()),
                                   "aurc_proxy": float(np.mean([correct[order[: int(len(order) * c)]].mean() for c in np.linspace(0.1, 1.0, 10)]))}
    L += ["## 1. Selective prediction (abstain on highest predictive entropy)", "",
          "| coverage | " + " | ".join(sel) + " |", "|---|" + "---|" * len(sel), "| accuracy | " + " | ".join(f"{v:.3f}" for v in sel.values()) + " |", "",
          f"- single-model accuracy {single.mean():.3f} → {len(models)}-seed ensemble {correct.mean():.3f}", ""]

    # ---------------- 2. epistemic uncertainty
    out["uncertainty"] = {"mean_entropy_nats": float(ent.mean()), "mean_mutual_information_nats": float(mi.mean()),
                          "mi_on_wrong_vs_right": [float(mi[correct == 0].mean()) if (correct == 0).any() else None, float(mi[correct == 1].mean())],
                          "unseen_label_rate_test": float(np.mean([a == 1 for c in enc_te for a in c.act]))}
    L += ["## 2. Epistemic uncertainty (seed-ensemble mutual information)", "",
          f"- mean predictive entropy {ent.mean():.3f} nats; mean mutual information {mi.mean():.4f} nats "
          f"(on wrong predictions {out['uncertainty']['mi_on_wrong_vs_right'][0]}, on right {out['uncertainty']['mi_on_wrong_vs_right'][1]:.4f})",
          f"- unseen-label rate in this test set: {out['uncertainty']['unseen_label_rate_test']:.4f}", "",
          "Compare this row across the random / chronological / strict runs of the same log: epistemic uncertainty should rise under shift.", ""]

    # ---------------- 3. SLA breach
    if RP:
        RP = np.concatenate(RP, 1); RT = np.concatenate(RT); RG = np.array(RG)
        rem_train = np.concatenate([c.remaining[np.isfinite(c.remaining)] for c in enc_tr])
        sla = {}
        for label, T in (("train_median", float(np.median(rem_train))), ("train_p90", float(np.percentile(rem_train, 90)))):
            y = (RT > T).astype(int)
            if 0 < y.mean() < 1:
                p_seeds = np.stack([laplace_sf(RP[s, :, 0], RP[s, :, 1], np.log1p(T)) for s in range(RP.shape[0])])
                p = p_seeds.mean(0)
                sla[label] = {"T_hours": T / 3600, "positive_rate": float(y.mean()), "auroc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
                              "auroc_single": float(roc_auc_score(y, p_seeds[0])), "brier_single": float(brier_score_loss(y, p_seeds[0]))}
        out["sla_breach"] = sla
        L += ["## 3. SLA-breach risk P(remaining > T) from the Laplace head", "",
              "| threshold | T (hours) | positive rate | AUROC (ensemble / single) | Brier (ensemble / single) |", "|---|---|---|---|---|"]
        for k, v in sla.items():
            L.append(f"| {k} | {v['T_hours']:.0f} | {v['positive_rate']:.2f} | {v['auroc']:.3f} / {v['auroc_single']:.3f} | {v['brier']:.3f} / {v['brier_single']:.3f} |")
        L.append("")

    # ---------------- 4. anomaly score
    nll = -np.log(np.clip(p_mean[np.arange(len(Y)), Y], 1e-12, 1))
    per_case = {}
    for g, v in zip(G, nll):
        per_case.setdefault(g, []).append(v)
    score = {g: float(np.mean(v)) for g, v in per_case.items()}
    lens = {c.case_id: len(c) for c in enc_te}
    ids = list(score)
    rho = spearmanr([score[i] for i in ids], [lens[i] for i in ids]).correlation
    mk = MarkovModel(order=2); mk.fit(enc_tr, [], encoder)
    mk_score = {}
    for c in enc_te:
        pr = mk.predict_case(c)
        m = c.next_act >= 0
        mk_score[c.case_id] = float(-np.log(np.clip(pr.next_probs[m][np.arange(m.sum()), c.next_act[m]], 1e-12, 1)).mean()) if m.any() else 0.0
    rho_mk = spearmanr([score[i] for i in ids], [mk_score[i] for i in ids]).correlation
    top = sorted(ids, key=lambda i: -score[i])[:10]
    by_enc = {c.case_id: c for c in enc_te}
    variants = Counter(tuple(by_enc[i].activities) for i in top)
    out["anomaly"] = {"spearman_score_vs_case_length": float(rho), "spearman_vs_markov_score": float(rho_mk),
                      "top10": [{"case_id": i, "score": score[i], "length": lens[i], "variant": " > ".join(by_enc[i].activities)[:300]} for i in top]}
    L += ["## 4. Anomaly score = per-case mean next-activity NLL", "",
          f"- Spearman(score, case length) = {rho:.2f} (confound check: a high value would mean the score just flags long cases)",
          f"- Spearman(score, Markov-k2 score) = {rho_mk:.2f} (agreement with a counting baseline)", "", "Top-5 most anomalous test cases:", ""]
    for i in top[:5]:
        L.append(f"- `{i}` (score {score[i]:.2f}, length {lens[i]}): " + " → ".join(by_enc[i].activities)[:400])
    L.append("")
    (d / "downstream.json").write_text(json.dumps(out, indent=1))
    (d / "downstream.md").write_text("\n".join(L))
    print(f"wrote {d / 'downstream.md'}")


if __name__ == "__main__":
    main()
