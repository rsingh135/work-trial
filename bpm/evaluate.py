"""Shared evaluation for every model: next event, time, multi-step, uncertainty, breakdowns."""

from __future__ import annotations

import time as _time

import numpy as np

from bpm import metrics as M
from bpm.data.encoding import EOS, EncodedCase, Encoder
from bpm.models.base import CasePreds, SequenceModel

PREFIX_BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 10_000)]


def _bucket(k: int) -> str:
    for lo, hi in PREFIX_BUCKETS:
        if lo <= k <= hi:
            return f"{lo}-{hi}" if lo != hi else str(lo)
    return "?"


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    """Grid-search a temperature on log-probs minimising NLL (post-hoc calibration on validation)."""
    logp = np.log(np.clip(probs, 1e-12, 1))
    best_t, best = 1.0, float("inf")
    for t in np.concatenate([np.linspace(0.3, 1.0, 15), np.linspace(1.0, 4.0, 31)]):
        z = logp / t
        z = z - z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        v = M.nll(p, y)
        if v < best:
            best, best_t = v, float(t)
    return best_t


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-12, 1)) / t
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def _collect(model: SequenceModel, cases: list[EncodedCase]):
    """Run predict_case over cases; return flat arrays for next-activity, Δt, remaining."""
    P, Y, G, K = [], [], [], []
    DT_P, DT_T, DT_Q, DT_G = [], [], [], []
    RM_P, RM_T, RM_G = [], [], []
    preds = model.predict_cases(cases)
    for c, pr in zip(cases, preds):
        m = c.next_act >= 0
        P.append(pr.next_probs[m]); Y.append(c.next_act[m]); G += [c.case_id] * int(m.sum()); K += (np.arange(len(c))[m] + 1).tolist()
        m = np.isfinite(c.next_dt)
        DT_P.append(pr.next_dt_s[m]); DT_T.append(np.expm1(c.next_dt[m])); DT_G += [c.case_id] * int(m.sum())
        if pr.next_dt_q is not None:
            DT_Q.append(pr.next_dt_q[m])
        m = np.isfinite(c.remaining) & (np.arange(len(c)) < len(c) - 1)
        RM_P.append(pr.remaining_s[m]); RM_T.append(c.remaining[m]); RM_G += [c.case_id] * int(m.sum())
    cat = lambda xs: np.concatenate(xs) if xs and sum(len(x) for x in xs) else np.zeros(0)
    return dict(P=cat(P), Y=cat(Y).astype(int), G=np.array(G), K=np.array(K),
                DT_P=cat(DT_P), DT_T=cat(DT_T), DT_Q=cat(DT_Q) if DT_Q else None, DT_G=np.array(DT_G),
                RM_P=cat(RM_P), RM_T=cat(RM_T), RM_G=np.array(RM_G))


def evaluate(model: SequenceModel, test: list[EncodedCase], val: list[EncodedCase], encoder: Encoder,
             terminal_ids: set[int], n_suffix_prefixes: int = 1000, max_suffix_len: int = 50, n_samples: int = 5,
             seed: int = 0, n_boot: int = 300) -> dict:
    t0 = _time.time()
    res: dict = {}
    te = _collect(model, test)
    va = _collect(model, val)

    # ---------------- next activity (+ calibration)
    P, Y, G = te["P"], te["Y"], te["G"]
    V = encoder.n_activities
    res["next_activity"] = {
        "n_prefixes": int(len(Y)),
        "nll": M.cluster_bootstrap(-np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, 1)), G, n_boot=n_boot, seed=seed).as_dict(),
        "accuracy": M.cluster_bootstrap((P.argmax(1) == Y).astype(float), G, n_boot=n_boot, seed=seed).as_dict(),
        "macro_f1": M.macro_f1(P.argmax(1), Y, V),
        "top3_accuracy": float(np.mean([y in np.argsort(-p)[:3] for p, y in zip(P, Y)])),
        "eos_recall": float(((P.argmax(1) == EOS) & (Y == EOS)).sum() / max((Y == EOS).sum(), 1)),
        "eos_precision": float(((P.argmax(1) == EOS) & (Y == EOS)).sum() / max((P.argmax(1) == EOS).sum(), 1)),
    }
    e_raw, rel_raw = M.ece(P, Y)
    T = fit_temperature(va["P"], va["Y"]) if len(va["Y"]) else 1.0
    Pc = apply_temperature(P, T)
    e_cal, rel_cal = M.ece(Pc, Y)
    res["uncertainty"] = {
        "ece_raw": e_raw, "brier_raw": M.brier(P, Y), "nll_raw": M.nll(P, Y),
        "temperature": T, "ece_calibrated": e_cal, "brier_calibrated": M.brier(Pc, Y), "nll_calibrated": M.nll(Pc, Y),
        "mean_entropy_bits": float(np.mean(-(P * np.log2(np.clip(P, 1e-12, 1))).sum(1))),
        "reliability_raw": rel_raw, "reliability_calibrated": rel_cal,
    }

    # ---------------- time
    res["next_dt_hours"] = {
        "n": int(len(te["DT_T"])),
        "mae": M.cluster_bootstrap(np.abs(te["DT_P"] - te["DT_T"]) / M.HOUR, te["DT_G"], n_boot=n_boot, seed=seed).as_dict(),
        "medae": M.medae_hours(te["DT_P"], te["DT_T"]),
        "mae_log1p": float(np.mean(np.abs(np.log1p(np.maximum(te["DT_P"], 0)) - np.log1p(te["DT_T"])))),
        "naive_median_mae": M.mae_hours(np.full_like(te["DT_T"], np.median(np.concatenate([np.expm1(c.next_dt[np.isfinite(c.next_dt)]) for c in val]) if val else 0)), te["DT_T"]),
    }
    if te["DT_Q"] is not None and len(te["DT_Q"]):
        cov = np.mean((te["DT_T"] >= te["DT_Q"][:, 0]) & (te["DT_T"] <= te["DT_Q"][:, 1]))
        res["next_dt_hours"]["q10_q90_coverage"] = float(cov)
        res["next_dt_hours"]["q10_q90_mean_width_hours"] = float(np.mean(te["DT_Q"][:, 1] - te["DT_Q"][:, 0]) / M.HOUR)
    if len(te["RM_T"]):
        res["remaining_hours"] = {
            "n": int(len(te["RM_T"])),
            "mae": M.cluster_bootstrap(np.abs(te["RM_P"] - te["RM_T"]) / M.HOUR, te["RM_G"], n_boot=n_boot, seed=seed).as_dict(),
            "medae": M.medae_hours(te["RM_P"], te["RM_T"]),
        }
    else:
        res["remaining_hours"] = {"n": 0, "note": "no complete cases (censored log)"}

    # ---------------- multi-step continuation (complete cases only)
    rng = np.random.default_rng(seed)
    cand = [(i, t) for i, c in enumerate(test) if c.complete for t in range(0, len(c) - 1)]
    if cand:
        pick = rng.choice(len(cand), size=min(n_suffix_prefixes, len(cand)), replace=False)
        sims, exact, term, sims_s, any_exact, sims_by_k, post_term, lens_pred, lens_true, per_pos_err = [], [], [], [], [], {}, [], [], [], {}
        medoid_sims, medoid_exact, medoid_term = [], [], []
        groups = []
        items = [(test[cand[j][0]], cand[j][1]) for j in pick]
        greedy = model.rollout_many(items, max_suffix_len, mode="greedy", n=1, seed=seed)
        sampled = model.rollout_many(items, max_suffix_len, mode="sample", n=n_samples, seed=seed + 1) if n_samples > 0 else None
        for r, (c, t) in enumerate(items):
            true = c.act[t + 1:].tolist()
            g = greedy[r][0]
            s = M.dl_similarity(g, true)
            sims.append(s); exact.append(float(g == true)); term.append(float(len(g) < max_suffix_len))
            lens_pred.append(len(g)); lens_true.append(len(true)); groups.append(c.case_id)
            sims_by_k.setdefault(_bucket(t + 1), []).append(s)
            # coherence: did the rollout continue after emitting a terminal activity?
            post_term.append(float(any(a in terminal_ids for a in g[:-1])))
            # error compounding: per-step accuracy along the greedy suffix
            for step, (pa, ta) in enumerate(zip(g, true)):
                per_pos_err.setdefault(step, []).append(float(pa != ta))
            if sampled is not None:
                ss = sampled[r]
                sims_s.append(np.mean([M.dl_similarity(x, true) for x in ss]))
                any_exact.append(float(any(x == true for x in ss)))
                # medoid decoding: the sample most similar to the other samples (a consensus continuation)
                med = max(range(len(ss)), key=lambda i: sum(M.dl_similarity(ss[i], ss[j]) for j in range(len(ss)) if j != i))
                medoid_sims.append(M.dl_similarity(ss[med], true)); medoid_exact.append(float(ss[med] == true))
                medoid_term.append(float(len(ss[med]) < max_suffix_len))
        groups = np.array(groups)
        res["suffix"] = {
            "n_prefixes": int(len(pick)), "max_len": max_suffix_len,
            "dl_similarity": M.cluster_bootstrap(np.array(sims), groups, n_boot=n_boot, seed=seed).as_dict(),
            "exact_match": float(np.mean(exact)),
            "valid_termination_rate": float(np.mean(term)),
            "post_terminal_continuation_rate": float(np.mean(post_term)),
            "mean_len_pred": float(np.mean(lens_pred)), "mean_len_true": float(np.mean(lens_true)),
            "dl_by_prefix_len": {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in sorted(sims_by_k.items(), key=lambda kv: int(kv[0].split("-")[0]))},
            "greedy_step_error_rate": {str(k): {"err": float(np.mean(v)), "n": len(v)} for k, v in sorted(per_pos_err.items()) if k < 15},
        }
        if n_samples > 0:
            res["suffix"]["sampled_mean_dl"] = float(np.mean(sims_s))
            res["suffix"][f"any_of_{n_samples}_exact"] = float(np.mean(any_exact))
            res["suffix"]["medoid_dl"] = float(np.mean(medoid_sims))
            res["suffix"]["medoid_exact"] = float(np.mean(medoid_exact))
            res["suffix"]["medoid_valid_termination_rate"] = float(np.mean(medoid_term))
    else:
        res["suffix"] = {"n_prefixes": 0, "note": "no complete cases in test (censored log)"}

    # ---------------- breakdown by prefix length (next activity)
    K = te["K"]
    by_k = {}
    for lo, hi in PREFIX_BUCKETS:
        m = (K >= lo) & (K <= hi)
        if m.sum() == 0:
            continue
        by_k[_bucket(lo)] = {"n": int(m.sum()), "accuracy": M.accuracy(P[m], Y[m]), "nll": M.nll(P[m], Y[m])}
    res["next_activity_by_prefix_len"] = by_k
    res["eval_seconds"] = _time.time() - t0
    return res
