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


# ----------------------------------------------------------------------------------------------
# Rule-based gates derived from information the environment already publishes. They are the
# "does the world model of the action space say this is possible" checks, and serve as controls
# for how much of a learned policy's gain is attributable to them.
# ----------------------------------------------------------------------------------------------
import re as _re

_API_RE = _re.compile(r"apis\.([a-z_]+)\.([a-z_]+)\(")
_META_APPS = ("api_docs", "supervisor")


def _api_calls(code: str):
    return _API_RE.findall(code or "")


def schema_violations(code: str, schema: dict | None) -> list[str]:
    """API calls in ``code`` whose app.api does not exist in the published action schema."""
    if not schema or "apps" not in schema:
        return []
    apps = schema["apps"]
    return [f"{a}.{b}" for a, b in _api_calls(code) if a not in apps or b not in apps[a]]


def premature_completion(code: str, history: list[dict]) -> bool:
    """``complete_task`` before any productive (non-docs, non-supervisor) call succeeded."""
    if "complete_task" not in (code or ""):
        return False
    for h in history:
        if h.get("error_type"):
            continue
        if any(a not in _META_APPS for a, _ in _api_calls(h.get("code", ""))):
            return False
    return True


class GatedPolicy:
    """Wraps any policy: candidates that violate the schema or complete prematurely are pushed to
    the back (they are still recorded and scored); the inner policy chooses among the survivors.
    If nothing survives, the inner policy sees all candidates unchanged."""

    def __init__(self, inner, schema_gate: bool = True, progress_gate: bool = True):
        self.inner = inner
        self.schema_gate, self.progress_gate = schema_gate, progress_gate
        self.policy_id = f"{inner.policy_id}+gate" if inner.policy_id != "first_candidate" else "gate_only"
        self.version = getattr(inner, "version", "1") + "+gate1"
        self.n_candidates = max(getattr(inner, "n_candidates", 1), 3)  # a gate needs alternatives to pick from

    def choose(self, context: dict, codes: list[str]):
        schema, hist = context.get("action_schema"), context.get("history", [])
        bad = [bool(self.schema_gate and schema_violations(c, schema)) or bool(self.progress_gate and premature_completion(c, hist)) for c in codes]
        keep = [i for i, b in enumerate(bad) if not b]
        if not keep or len(keep) == len(codes):
            idx, scores, meta = self.inner.choose(context, codes)
            return idx, scores, {**meta, "gate_rejected": [i for i, b in enumerate(bad) if b], "gate_fallback": not keep}
        sub_idx, sub_scores, meta = self.inner.choose(context, [codes[i] for i in keep])
        scores = [None] * len(codes)
        for k, sc in zip(keep, sub_scores):
            scores[k] = sc
        chosen = keep[sub_idx]
        return chosen, scores, {**meta, "gate_rejected": [i for i, b in enumerate(bad) if b], "overrode_first": chosen != 0}
