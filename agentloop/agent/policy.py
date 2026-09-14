"""Policies: how the agent picks among sampled candidate actions.

``FirstCandidatePolicy``  baseline — one sample, take it (n_candidates=1).
``RerankerPolicy``        learned component reinserted into the loop: sample n candidates from
                          the same LLM at the same temperature, score each with the trained
                          action-validity model, execute the argmax. Ties → earliest sample,
                          so with an uninformative model it reduces exactly to the baseline.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from agentloop.agent.features import Featurizer


class FirstCandidatePolicy:
    policy_id = "first_candidate"
    version = "1"
    n_candidates = 1

    def choose(self, context: dict, codes: list[str]) -> tuple[int, list[float | None], dict[str, Any]]:
        return 0, [None] * len(codes), {}


class RerankerPolicy:
    policy_id = "validity_reranker"

    def __init__(self, model_path: str | Path, n_candidates: int = 3, min_margin: float = 0.0, score_mode: str = "product"):
        """score_mode: 'valid' = P(no error); 'success' = P(episode success); 'product' = both multiplied
        (falls back to 'valid' when the bundle has no success head)."""
        with open(model_path, "rb") as fh:
            bundle = pickle.load(fh)
        self.clf = bundle["model"]
        self.clf_success = bundle.get("model_success")
        self.score_mode = score_mode if (self.clf_success is not None or score_mode == "valid") else "valid"
        self.featurizer: Featurizer = bundle["featurizer"]
        self.version = bundle.get("version", "unknown")
        self.n_candidates = n_candidates
        self.min_margin = min_margin  # only override the first sample if the best beats it by this much

    def score(self, context: dict, codes: list[str]) -> np.ndarray:
        X = self.featurizer.transform([context] * len(codes), codes)
        pv = self.clf.predict_proba(X)[:, 1]
        if self.score_mode == "valid":
            return pv
        ps = self.clf_success.predict_proba(X)[:, 1]
        return ps if self.score_mode == "success" else pv * ps

    def choose(self, context: dict, codes: list[str]) -> tuple[int, list[float | None], dict[str, Any]]:
        if not codes:
            return 0, [], {}
        s = self.score(context, codes)
        best = int(np.argmax(s))
        if s[best] - s[0] < self.min_margin:
            best = 0
        return best, [float(x) for x in s], {"scores": [float(x) for x in s], "overrode_first": best != 0, "score_mode": self.score_mode}
