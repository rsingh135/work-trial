"""Baseline 2: gradient-boosted trees over hand-built prefix features.

One feature vector per prefix position:
  last-k activity ids (k=3, as categorical), bag-of-activity counts so far, the 7 time
  features of the current event, event categorical ids of the current event, case
  categoricals + numerics.
Three separate boosters share the features: next activity (classifier), log1p Δt
(regressor), log1p remaining (regressor). Rollout is autoregressive: append the predicted
activity with the predicted Δt and recompute the features. Uses sklearn's
HistGradientBoosting (no native-library dependency).
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from bpm.data.encoding import EOS, UNK, EncodedCase, Encoder
from bpm.models.base import CasePreds, SequenceModel


class GBMModel(SequenceModel):
    name = "gbm"

    def __init__(self, k_last: int = 3, max_iter: int = 200, learning_rate: float = 0.05, seed: int = 0):
        self.k_last = k_last
        self.max_iter = max_iter
        self.lr = learning_rate
        self.seed = seed

    # ------------------------------------------------------------- features
    def _feat_row(self, enc: EncodedCase, t: int, acts: np.ndarray, time_t: np.ndarray, cat_t: np.ndarray) -> np.ndarray:
        last = np.full(self.k_last, np.nan)  # NaN = 'no such earlier event' (sklearn treats NaN as missing)
        for j in range(self.k_last):
            if t - j >= 0:
                last[j] = acts[t - j]
        bag = np.bincount(acts[: t + 1], minlength=self.V).astype(float)
        return np.concatenate([last, bag, time_t, cat_t.astype(float), enc.case_cat.astype(float), enc.case_num, [t]])

    def _features(self, enc: EncodedCase) -> np.ndarray:
        return np.stack([self._feat_row(enc, t, enc.act, enc.time[t], enc.cat[t]) for t in range(len(enc))])

    def _cat_mask(self, enc: EncodedCase) -> list[bool]:
        n_last, n_bag, n_time = self.k_last, self.V, enc.time.shape[1]
        n_cat, n_ccat, n_cnum = enc.cat.shape[1], len(enc.case_cat), len(enc.case_num)
        return [True] * n_last + [False] * n_bag + [False] * n_time + [True] * n_cat + [True] * n_ccat + [False] * n_cnum + [False]

    def fit(self, train, val, encoder: Encoder) -> dict:
        self.V = encoder.n_activities
        X, ya, ydt, yrem = [], [], [], []
        for c in train:
            F = self._features(c)
            X.append(F)
            ya.append(c.next_act)
            ydt.append(c.next_dt)
            yrem.append(c.remaining)
        X = np.concatenate(X)
        ya, ydt, yrem = np.concatenate(ya), np.concatenate(ydt), np.concatenate(yrem)
        cat_mask = self._cat_mask(train[0])
        # sklearn needs categorical values < max_bins; ids here are small (vocab-sized)
        common = dict(max_iter=self.max_iter, learning_rate=self.lr, early_stopping=True, validation_fraction=0.1,
                      random_state=self.seed, categorical_features=cat_mask, max_bins=255, l2_regularization=1.0, min_samples_leaf=20)
        m = ya >= 0
        # classifier: sklearn's internal early-stopping split is stratified and fails on classes with <2
        # members (rare activities), so use a fixed iteration budget instead
        clf_kw = {**common, "early_stopping": False, "max_iter": min(self.max_iter, 120)}
        self.clf = HistGradientBoostingClassifier(**clf_kw).fit(X[m], ya[m])
        self.classes_ = self.clf.classes_
        m = np.isfinite(ydt)
        self.reg_dt = HistGradientBoostingRegressor(loss="absolute_error", **common).fit(X[m], ydt[m])
        m = np.isfinite(yrem)
        self.reg_rem = HistGradientBoostingRegressor(loss="absolute_error", **common).fit(X[m], np.log1p(yrem[m]))
        return {"n_rows": int(len(X)), "clf_iters": int(self.clf.n_iter_), "dt_iters": int(self.reg_dt.n_iter_), "rem_iters": int(self.reg_rem.n_iter_)}

    def _probs_full(self, X: np.ndarray) -> np.ndarray:
        p = self.clf.predict_proba(X)
        out = np.full((len(X), self.V), 1e-6)
        out[:, self.classes_] = p
        return out / out.sum(1, keepdims=True)

    def predict_case(self, enc: EncodedCase) -> CasePreds:
        X = self._features(enc)
        probs = self._probs_full(X)
        dt = np.expm1(self.reg_dt.predict(X))
        rem = np.expm1(self.reg_rem.predict(X))
        return CasePreds(probs, dt, rem)

    def predict_cases(self, cases):
        """One predict call for all positions of all cases (sklearn per-call overhead dominates otherwise)."""
        if not cases:
            return []
        X = np.concatenate([self._features(c) for c in cases])
        probs = self._probs_full(X)
        dt = np.expm1(self.reg_dt.predict(X))
        rem = np.expm1(self.reg_rem.predict(X))
        out, i = [], 0
        for c in cases:
            n = len(c)
            out.append(CasePreds(probs[i:i + n], dt[i:i + n], rem[i:i + n]))
            i += n
        return out

    def rollout(self, enc, t, max_len, mode="greedy", n=1, seed=0):
        return self.rollout_many([(enc, t)], max_len, mode=mode, n=n, seed=seed)[0]

    def rollout_many(self, items, max_len, mode="greedy", n=1, seed=0):
        """All rollouts advance in lock-step so each step is ONE predict_proba call (sklearn's
        per-call overhead dominates otherwise)."""
        rng = np.random.default_rng(seed)
        R = [(enc, t, k) for enc, t in items for k in range(n)]
        acts = [enc.act[: t + 1].tolist() for enc, t, _ in R]
        ts = np.array([float(enc.timestamps[t]) for enc, t, _ in R])
        start = np.array([float(enc.timestamps[0]) for enc, t, _ in R])
        time_t = [enc.time[t].copy() for enc, t, _ in R]
        cat_t = [enc.cat[t].copy() for enc, t, _ in R]
        suffix = [[] for _ in R]
        alive = np.ones(len(R), dtype=bool)
        unk_cat = np.full(R[0][0].cat.shape[1], UNK) if R else None
        for step in range(max_len):
            idx = np.flatnonzero(alive)
            if len(idx) == 0:
                break
            X = np.stack([self._feat_row(R[i][0], len(acts[i]) - 1, np.array(acts[i]), time_t[i], cat_t[i]) for i in idx])
            P = self._probs_full(X)
            DT = np.maximum(np.expm1(self.reg_dt.predict(X)), 0.0)
            for row, i in enumerate(idx):
                p = P[row]
                a = int(p.argmax()) if mode == "greedy" else int(rng.choice(self.V, p=p))
                if a == EOS:
                    alive[i] = False
                    continue
                suffix[i].append(a)
                acts[i].append(a)
                ts[i] += DT[row]
                time_t[i] = _synthetic_time_feats(self._enc_scalers, DT[row], ts[i] - start[i], ts[i], len(acts[i]) - 1)
                cat_t[i] = unk_cat
        out = [[None] * n for _ in items]
        j = 0
        for i in range(len(items)):
            for k in range(n):
                out[i][k] = suffix[j]; j += 1
        return out

    def set_encoder(self, encoder: Encoder):
        self._enc_scalers = encoder


def _synthetic_time_feats(encoder: Encoder, dt: float, elapsed: float, ts: float, pos: int) -> np.ndarray:
    # local hour / weekday from the synthetic timestamp (UTC → approximate local with fixed +1h; the
    # sin/cos features are coarse enough that DST error is immaterial for a baseline)
    hour = ((ts + 3600) / 3600) % 24
    wd = (int((ts + 3600) // 86400) + 3) % 7  # 1970-01-01 was a Thursday (=3)
    return np.array(
        [
            encoder.dt_scaler(np.log1p(dt)),
            encoder.elapsed_scaler(np.log1p(max(elapsed, 0))),
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * wd / 7),
            math.cos(2 * math.pi * wd / 7),
            encoder.pos_scaler(np.log1p(pos)),
        ],
        dtype=np.float32,
    )
