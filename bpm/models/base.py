"""Common model interface so baselines and the proposed model are evaluated identically."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from bpm.data.encoding import EncodedCase, Encoder


@dataclass
class CasePreds:
    """Per-position predictions for one case (position t = after observing events 0..t)."""

    next_probs: np.ndarray  # [T, V] distribution over next activity (incl. EOS)
    next_dt_s: np.ndarray  # [T] point prediction, seconds to next event
    remaining_s: np.ndarray  # [T] point prediction, seconds to case end
    next_dt_q: np.ndarray | None = None  # [T, 2] optional (q10, q90) seconds — predictive interval
    remaining_params: np.ndarray | None = None  # [T, 2] optional Laplace (mu, log_b) on log1p(seconds) — for P(remaining > T)


class SequenceModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train: list[EncodedCase], val: list[EncodedCase], encoder: Encoder) -> dict:
        """Train; return a JSON-serialisable training log."""

    @abstractmethod
    def predict_case(self, enc: EncodedCase) -> CasePreds:
        ...

    @abstractmethod
    def rollout(
        self, enc: EncodedCase, t: int, max_len: int, mode: str = "greedy", n: int = 1, seed: int = 0
    ) -> list[list[int]]:
        """Continue the case from position t. Returns ``n`` activity-id suffixes; a suffix stops
        at (and excludes) EOS, or is truncated at ``max_len``. ``mode`` ∈ {greedy, sample}."""

    def predict_cases(self, cases: list[EncodedCase]) -> list[CasePreds]:
        """Batched version of ``predict_case``; default loops."""
        return [self.predict_case(c) for c in cases]

    def rollout_many(self, items: list[tuple[EncodedCase, int]], max_len: int, mode: str = "greedy", n: int = 1, seed: int = 0) -> list[list[list[int]]]:
        """Batched version of ``rollout`` over many (case, position) pairs. Default: loop."""
        return [self.rollout(enc, t, max_len, mode=mode, n=n, seed=seed + i) for i, (enc, t) in enumerate(items)]

    def param_count(self) -> int:
        return 0
