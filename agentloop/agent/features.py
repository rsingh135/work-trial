"""Features for the learned component (action-validity scorer).

Input: a step *context* (instruction, last observation, last error, short history) and a
*candidate action* (code). Output: sparse hashed features — no vocabulary to fit, so the
featuriser is stateless and identical at train and deploy time.

Blocks
  code_words   : word 1-2grams of the candidate code
  code_chars   : char 3-5grams of the candidate code (catches typos / malformed calls)
  api_calls    : tokens like "spotify.login", plus app-only "spotify"
  err_words    : word 1-2grams of the previous step's error message (if any)
  obs_words    : word unigrams of the last observation (truncated)
  cross        : api_call × (had_error_prev) interactions — "retrying the same call after an error"
  numeric      : n_lines, n_chars, n_api_calls, calls_complete_task, prev_was_error, step_index,
                 repeats_previous_code, n_prev_errors
"""

from __future__ import annotations

import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

_API = re.compile(r"apis\.([a-z_]+)\.([a-z_]+)\(")
_H = dict(alternate_sign=False, norm="l2")


class Featurizer:
    version = "1"

    def __init__(self, n_features: int = 2**18):
        self.v_code_w = HashingVectorizer(n_features=n_features, ngram_range=(1, 2), token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|[=()\[\].,:]", **_H)
        self.v_code_c = HashingVectorizer(n_features=n_features, analyzer="char_wb", ngram_range=(3, 5), **_H)
        self.v_api = HashingVectorizer(n_features=n_features, ngram_range=(1, 1), token_pattern=r"\S+", lowercase=False, **_H)
        self.v_err = HashingVectorizer(n_features=n_features, ngram_range=(1, 2), **_H)
        self.v_obs = HashingVectorizer(n_features=n_features, ngram_range=(1, 1), **_H)
        self.v_cross = HashingVectorizer(n_features=n_features, ngram_range=(1, 1), token_pattern=r"\S+", lowercase=False, **_H)

    @staticmethod
    def api_tokens(code: str) -> list[str]:
        calls = _API.findall(code)
        return [f"{a}.{b}" for a, b in calls] + [f"app:{a}" for a, _ in calls]

    def transform(self, contexts: list[dict], codes: list[str]) -> sparse.csr_matrix:
        api_docs = [" ".join(self.api_tokens(c)) for c in codes]
        errs = [ctx.get("last_error") or "" for ctx in contexts]
        obs = [(ctx.get("last_observation") or "")[:1500] for ctx in contexts]
        cross = []
        num = []
        for ctx, code, ap in zip(contexts, codes, api_docs):
            had_err = 1.0 if ctx.get("last_error") else 0.0
            prev_code = ctx.get("last_code") or ""
            cross.append(" ".join(f"{t}|err{int(had_err)}" for t in ap.split()))
            num.append([
                code.count("\n") + 1, min(len(code), 4000) / 1000.0, len(_API.findall(code)),
                1.0 if "complete_task" in code else 0.0, had_err, float(ctx.get("step_index", 0)) / 10.0,
                1.0 if prev_code.strip() == code.strip() else 0.0, float(ctx.get("n_prev_errors", 0)) / 5.0,
                1.0 if "show_api_doc" in code or "show_api_descriptions" in code or "show_app_descriptions" in code else 0.0,
                1.0 if "show_account_passwords" in code else 0.0, 1.0 if ".login(" in code else 0.0,
                1.0 if "print(" in code else 0.0,
            ])
        X = sparse.hstack([
            self.v_code_w.transform(codes), self.v_code_c.transform(codes), self.v_api.transform(api_docs),
            self.v_err.transform(errs), self.v_obs.transform(obs), self.v_cross.transform(cross),
            sparse.csr_matrix(np.asarray(num, dtype=np.float32)),
        ]).tocsr()
        return X
