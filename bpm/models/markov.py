"""Baseline 1: k-order Markov transition model with back-off, plus per-state median times.

* next activity: P(a_{t+1} | a_{t-k+1..t}) with additive smoothing, backing off to shorter
  contexts (and finally the unigram) when a context was unseen in training.
* next Δt: median of log1p(Δt) following the same context (back-off likewise).
* remaining time: median remaining seconds given (last activity, position bucket).
* rollout: greedy argmax / ancestral sampling from the same tables.

Deliberately cheap; it is the "how far does counting get you" reference.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from bpm.data.encoding import EOS, EncodedCase, Encoder
from bpm.models.base import CasePreds, SequenceModel


class MarkovModel(SequenceModel):
    def __init__(self, order: int = 2, alpha: float = 0.1):
        self.order = order
        self.alpha = alpha
        self.name = f"markov_k{order}"

    def fit(self, train, val, encoder: Encoder) -> dict:
        V = encoder.n_activities
        self.V = V
        self.trans: list[dict[tuple, np.ndarray]] = [dict() for _ in range(self.order + 1)]
        counts: list[dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(self.order + 1)]
        dts: list[dict[tuple, list]] = [defaultdict(list) for _ in range(self.order + 1)]
        rem: dict[tuple, list] = defaultdict(list)
        for c in train:
            acts = c.act.tolist()
            for t in range(len(acts)):
                y = c.next_act[t]
                for k in range(self.order + 1):
                    ctx = tuple(acts[max(0, t - k + 1) : t + 1]) if k > 0 else ()
                    if k > 0 and len(ctx) < k:
                        continue
                    if y >= 0:
                        counts[k][ctx][y] += 1
                    if np.isfinite(c.next_dt[t]):
                        dts[k][ctx].append(float(c.next_dt[t]))
                if np.isfinite(c.remaining[t]):
                    rem[(acts[t], min(t, 10))].append(float(c.remaining[t]))
        for k in range(self.order + 1):
            for ctx, cnt in counts[k].items():
                p = np.full(V, self.alpha)
                for a, n in cnt.items():
                    p[a] += n
                self.trans[k][ctx] = p / p.sum()
        self.dt_med = [{ctx: float(np.median(v)) for ctx, v in dts[k].items()} for k in range(self.order + 1)]
        self.rem_med = {key: float(np.median(v)) for key, v in rem.items()}
        self.rem_global = float(np.median([x for v in rem.values() for x in v])) if rem else 0.0
        self.dt_global = float(np.median([x for v in dts[0].values() for x in v])) if dts[0] else 0.0
        return {"contexts": {k: len(self.trans[k]) for k in range(self.order + 1)}}

    def _probs(self, acts: list[int], t: int) -> np.ndarray:
        for k in range(self.order, -1, -1):
            ctx = tuple(acts[max(0, t - k + 1) : t + 1]) if k > 0 else ()
            if k > 0 and len(ctx) < k:
                continue
            if ctx in self.trans[k]:
                return self.trans[k][ctx]
        return np.full(self.V, 1.0 / self.V)

    def _dt(self, acts: list[int], t: int) -> float:
        for k in range(self.order, -1, -1):
            ctx = tuple(acts[max(0, t - k + 1) : t + 1]) if k > 0 else ()
            if k > 0 and len(ctx) < k:
                continue
            if ctx in self.dt_med[k]:
                return float(np.expm1(self.dt_med[k][ctx]))
        return float(np.expm1(self.dt_global))

    def predict_case(self, enc: EncodedCase) -> CasePreds:
        acts = enc.act.tolist()
        T = len(acts)
        probs = np.stack([self._probs(acts, t) for t in range(T)])
        dt = np.array([self._dt(acts, t) for t in range(T)])
        rem = np.array([self.rem_med.get((acts[t], min(t, 10)), self.rem_global) for t in range(T)])
        return CasePreds(probs, dt, rem)

    def rollout(self, enc, t, max_len, mode="greedy", n=1, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            acts = enc.act[: t + 1].tolist()
            suffix = []
            for _ in range(max_len):
                p = self._probs(acts, len(acts) - 1)
                a = int(p.argmax()) if mode == "greedy" else int(rng.choice(self.V, p=p))
                if a == EOS:
                    break
                suffix.append(a)
                acts.append(a)
            out.append(suffix)
        return out
