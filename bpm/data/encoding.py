"""Train-fit encoder: vocabularies, scalers, and per-position targets.

Observation at position t (after seeing events 0..t):
  act[t]            activity id (train vocab; UNK for unseen)
  comps[t, :]       decomposed activity component ids (shared tables; UNK for unseen)
  cat[t, :]         event categorical ids (per field; UNK for unseen / rare)
  time[t, :]        [z(log1p Δt_s), z(log1p elapsed_s), sin/cos hour, sin/cos weekday, z(log1p pos)]
  case_cat[:]       case categorical ids (known at case start)
  case_num[:]       z(log1p |x|) with a missing indicator per field

Targets at position t:
  next_act[t]       act[t+1], or EOS if t is the last event of a *complete* case; -1 (masked)
                    if t is the last event of an incomplete case
  next_dt[t]        log1p(seconds to next event); NaN at last position
  remaining[t]      seconds from event t to the last event; NaN for incomplete cases
  suffix            handled at eval time from raw activities (only complete cases)

All statistics are fitted on the training split only (``Encoder.fit``).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bpm.data.cases import Case, decompose_activity

PAD, UNK, EOS = 0, 1, 2
SPECIALS = ["<PAD>", "<UNK>", "<EOS>"]


def _fit_vocab(values, min_count: int, specials=("<PAD>", "<UNK>")) -> dict[str, int]:
    cnt = Counter(values)
    vocab = {s: i for i, s in enumerate(specials)}
    for v, c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        if c >= min_count and v not in vocab:
            vocab[v] = len(vocab)
    return vocab


@dataclass
class Scaler:
    mean: float = 0.0
    std: float = 1.0

    def fit(self, x: np.ndarray) -> "Scaler":
        x = x[np.isfinite(x)]
        self.mean = float(x.mean()) if len(x) else 0.0
        self.std = float(x.std()) if len(x) and x.std() > 1e-8 else 1.0
        return self

    def __call__(self, x):
        return (x - self.mean) / self.std


@dataclass
class EncodedCase:
    case_id: str
    act: np.ndarray  # [T] int64
    comps: np.ndarray  # [T, C] int64
    cat: np.ndarray  # [T, F] int64
    time: np.ndarray  # [T, D] float32
    case_cat: np.ndarray  # [Fc] int64
    case_num: np.ndarray  # [2*Fn] float32
    next_act: np.ndarray  # [T] int64 (-1 masked)
    next_dt: np.ndarray  # [T] float32 log1p seconds (nan masked)
    remaining: np.ndarray  # [T] float32 seconds (nan masked)
    timestamps: np.ndarray  # [T] float64 raw seconds (for evaluation only)
    activities: list[str]
    complete: bool

    def __len__(self) -> int:
        return len(self.act)


@dataclass
class Encoder:
    family: str
    event_cat_fields: list[str]
    case_cat_fields: list[str]
    case_num_fields: list[str]
    min_count_cat: int = 5
    min_count_act: int = 1
    act_vocab: dict[str, int] = field(default_factory=dict)
    comp_vocab: dict[str, int] = field(default_factory=dict)
    n_comps: int = 0
    cat_vocabs: dict[str, dict[str, int]] = field(default_factory=dict)
    case_cat_vocabs: dict[str, dict[str, int]] = field(default_factory=dict)
    dt_scaler: Scaler = field(default_factory=Scaler)
    elapsed_scaler: Scaler = field(default_factory=Scaler)
    pos_scaler: Scaler = field(default_factory=Scaler)
    case_num_scalers: dict[str, Scaler] = field(default_factory=dict)
    fitted: bool = False

    TIME_DIM = 7

    # ---------------------------------------------------------------- fitting
    def fit(self, cases: list[Case], extra_labels: list[str] | None = None) -> "Encoder":
        acts = [a for c in cases for a in c.activities]
        self.act_vocab = _fit_vocab(acts, self.min_count_act, specials=SPECIALS)
        for lbl in extra_labels or []:  # pre-registered labels (union vocab for transfer)
            self.act_vocab.setdefault(lbl, len(self.act_vocab))
        comps = [comp for a in set(acts) for comp in decompose_activity(a, self.family)]
        self.comp_vocab = _fit_vocab(comps, 1)
        self.n_comps = max(len(decompose_activity(a, self.family)) for a in set(acts))
        for f in self.event_cat_fields:
            self.cat_vocabs[f] = _fit_vocab([v for c in cases for v in c.event_cat[f]], self.min_count_cat)
        for f in self.case_cat_fields:
            self.case_cat_vocabs[f] = _fit_vocab([c.case_cat[f] for c in cases], self.min_count_cat)
        dts = np.concatenate([np.diff(c.timestamps) for c in cases if len(c) > 1] or [np.zeros(1)])
        self.dt_scaler.fit(np.log1p(np.maximum(dts, 0)))
        el = np.concatenate([c.timestamps - c.timestamps[0] for c in cases])
        self.elapsed_scaler.fit(np.log1p(np.maximum(el, 0)))
        self.pos_scaler.fit(np.log1p(np.concatenate([np.arange(len(c)) for c in cases]).astype(float)))
        for f in self.case_num_fields:
            vals = np.array([c.case_num.get(f, np.nan) for c in cases], dtype=float)
            self.case_num_scalers[f] = Scaler().fit(np.log1p(np.abs(vals[np.isfinite(vals)])))
        self.fitted = True
        return self

    # ------------------------------------------------------------- transform
    @property
    def n_activities(self) -> int:
        return len(self.act_vocab)

    @property
    def id2act(self) -> list[str]:
        inv = [""] * len(self.act_vocab)
        for k, v in self.act_vocab.items():
            inv[v] = k
        return inv

    def act_id(self, label: str) -> int:
        return self.act_vocab.get(label, UNK)

    def transform(self, case: Case) -> EncodedCase:
        assert self.fitted
        T = len(case)
        act = np.array([self.act_id(a) for a in case.activities], dtype=np.int64)
        comps = np.full((T, self.n_comps), UNK, dtype=np.int64)
        for t, a in enumerate(case.activities):
            for j, comp in enumerate(decompose_activity(a, self.family)[: self.n_comps]):
                comps[t, j] = self.comp_vocab.get(comp, UNK)
        cat = np.zeros((T, len(self.event_cat_fields)), dtype=np.int64)
        for j, f in enumerate(self.event_cat_fields):
            vocab = self.cat_vocabs[f]
            cat[:, j] = [vocab.get(v, UNK) for v in case.event_cat[f]]
        ts = case.timestamps
        dt = np.concatenate([[0.0], np.maximum(np.diff(ts), 0)])
        elapsed = np.maximum(ts - ts[0], 0)
        pos = np.arange(T, dtype=float)
        hour = case.local_hour.astype(float)
        wd = case.local_weekday.astype(float)
        time = np.stack(
            [
                self.dt_scaler(np.log1p(dt)),
                self.elapsed_scaler(np.log1p(elapsed)),
                np.sin(2 * math.pi * hour / 24),
                np.cos(2 * math.pi * hour / 24),
                np.sin(2 * math.pi * wd / 7),
                np.cos(2 * math.pi * wd / 7),
                self.pos_scaler(np.log1p(pos)),
            ],
            axis=1,
        ).astype(np.float32)
        case_cat = np.array(
            [self.case_cat_vocabs[f].get(case.case_cat.get(f, "<MISSING>"), UNK) for f in self.case_cat_fields],
            dtype=np.int64,
        )
        cn = []
        for f in self.case_num_fields:
            v = case.case_num.get(f, np.nan)
            miss = not np.isfinite(v)
            cn.append(0.0 if miss else float(self.case_num_scalers[f](np.log1p(abs(v)))))
            cn.append(1.0 if miss else 0.0)
        case_num = np.array(cn, dtype=np.float32)

        next_act = np.full(T, -1, dtype=np.int64)
        next_act[:-1] = act[1:]
        if case.complete:
            next_act[-1] = EOS
        next_dt = np.full(T, np.nan, dtype=np.float32)
        next_dt[:-1] = np.log1p(np.maximum(np.diff(ts), 0))
        remaining = np.full(T, np.nan, dtype=np.float32)
        if case.complete:
            remaining[:] = ts[-1] - ts
        return EncodedCase(
            case_id=case.case_id,
            act=act,
            comps=comps,
            cat=cat,
            time=time,
            case_cat=case_cat,
            case_num=case_num,
            next_act=next_act,
            next_dt=next_dt,
            remaining=remaining,
            timestamps=ts,
            activities=list(case.activities),
            complete=case.complete,
        )

    def transform_all(self, cases: list[Case]) -> list[EncodedCase]:
        return [self.transform(c) for c in cases]

    # ------------------------------------------------------------- persistence
    def to_json(self, path: Path) -> None:
        d = {
            "family": self.family,
            "event_cat_fields": self.event_cat_fields,
            "case_cat_fields": self.case_cat_fields,
            "case_num_fields": self.case_num_fields,
            "min_count_cat": self.min_count_cat,
            "min_count_act": self.min_count_act,
            "act_vocab": self.act_vocab,
            "comp_vocab": self.comp_vocab,
            "n_comps": self.n_comps,
            "cat_vocabs": self.cat_vocabs,
            "case_cat_vocabs": self.case_cat_vocabs,
            "scalers": {
                "dt": self.dt_scaler.__dict__,
                "elapsed": self.elapsed_scaler.__dict__,
                "pos": self.pos_scaler.__dict__,
                "case_num": {k: v.__dict__ for k, v in self.case_num_scalers.items()},
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, indent=1))

    @staticmethod
    def from_json(path: Path) -> "Encoder":
        d = json.loads(Path(path).read_text())
        e = Encoder(d["family"], d["event_cat_fields"], d["case_cat_fields"], d["case_num_fields"],
                    d["min_count_cat"], d["min_count_act"])
        e.act_vocab, e.comp_vocab, e.n_comps = d["act_vocab"], d["comp_vocab"], d["n_comps"]
        e.cat_vocabs, e.case_cat_vocabs = d["cat_vocabs"], d["case_cat_vocabs"]
        s = d["scalers"]
        e.dt_scaler, e.elapsed_scaler, e.pos_scaler = Scaler(**s["dt"]), Scaler(**s["elapsed"]), Scaler(**s["pos"])
        e.case_num_scalers = {k: Scaler(**v) for k, v in s["case_num"].items()}
        e.fitted = True
        return e


def encoder_for(schema, cases_train: list[Case], min_count_cat: int = 5, extra_labels=None) -> Encoder:
    return Encoder(
        family=schema.family,
        event_cat_fields=list(schema.event_cat) + list(schema.event_highcard),
        case_cat_fields=list(schema.case_cat),
        case_num_fields=list(schema.case_num),
        min_count_cat=min_count_cat,
    ).fit(cases_train, extra_labels=extra_labels)
