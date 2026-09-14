"""Case-level view of a log, plus case-level splits.

A ``Case`` is the *observation stream* the models consume: ordered events, each with an
activity label, timestamp, low-card categorical attributes, and case-start attributes.
Everything downstream (encoders, targets) is derived from this object.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bpm.ingest.registry import LogSchema
from bpm.ingest.xes import parse_xes

MISSING_TOKENS = {"", "UNKNOWN", "UNDEFINED", "nan", "None", "-", "<NA>"}

# BPI2020 activity grammar: "<object> <ACTION> by <ACTOR>"; BPI2013: "<status>|<substatus>"
_BPI2020_RE = re.compile(r"^(?P<obj>.+?) (?P<act>[A-Z_]+) by (?P<actor>[A-Z_ ]+)$")


def decompose_activity(label: str, family: str) -> tuple[str, ...]:
    """Split an activity label into schema-agnostic components.

    The components feed *shared* embedding tables so a label never seen in training can still
    be partially represented (e.g. ``Request For Payment APPROVED by SUPERVISOR`` shares
    ``APPROVED`` and ``SUPERVISOR`` with ``Declaration APPROVED by SUPERVISOR``).
    """
    if family == "bpi2013":
        parts = label.split("|", 1)
        return (f"status={parts[0]}", f"sub={parts[1] if len(parts) > 1 else ''}")
    if family == "agent":  # "app.api|ok" / "app.api|err" (agentloop.trace_model)
        base, _, status = label.partition("|")
        app, _, api = base.partition(".")
        return (f"app={app}", f"api={api}", f"status={status}")
    m = _BPI2020_RE.match(label)
    if m:
        return (f"obj={m['obj']}", f"act={m['act']}", f"actor={m['actor']}")
    return (f"obj={label}", "act=<none>", "actor=<none>")


@dataclass
class Case:
    case_id: str
    activities: list[str]
    timestamps: np.ndarray  # float64 seconds since epoch (UTC)
    local_hour: np.ndarray  # int, local wall-clock hour
    local_weekday: np.ndarray  # int 0..6
    event_cat: dict[str, list[str]]  # field -> per-event values
    case_cat: dict[str, str]
    case_num: dict[str, float]
    complete: bool
    family: str

    def __len__(self) -> int:
        return len(self.activities)

    @property
    def start(self) -> float:
        return float(self.timestamps[0])


def _clean(v) -> str:
    v = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    return "<MISSING>" if v in MISSING_TOKENS else v


def load_cases(schema: LogSchema, max_cases: int | None = None, min_len: int = 1) -> list[Case]:
    L = parse_xes(schema.path, name=schema.name, max_traces=max_cases)
    df = L.events
    act_cols = [f"ev_{k}" for k in schema.activity_keys]
    df = df.assign(activity=df[act_cols].fillna("").astype(str).agg("|".join, axis=1))
    # order within a case by timestamp; ties broken by file order (stable)
    df = df.sort_values(["case_id", "timestamp", "event_idx"], kind="stable")
    loc = df.timestamp.dt.tz_convert(schema.local_tz)
    df = df.assign(_hour=loc.dt.hour.astype(int), _wd=loc.dt.dayofweek.astype(int),
                   _ts=(df.timestamp - pd.Timestamp(0, tz="UTC")).dt.total_seconds())  # unit-safe (pandas 3 may store µs)

    cases: list[Case] = []
    for cid, g in df.groupby("case_id", sort=False):
        if len(g) < min_len:
            continue
        acts = g.activity.tolist()
        complete = (not schema.censored) and (acts[-1] in schema.terminal)
        ev_cat = {}
        for f in schema.event_cat:
            col = f"ev_{f}"
            ev_cat[f] = [_clean(v) for v in g[col].tolist()] if col in g else ["<MISSING>"] * len(g)
        for f in schema.event_highcard:
            col = f"ev_{f}"
            ev_cat[f] = [_clean(v) for v in g[col].tolist()] if col in g else ["<MISSING>"] * len(g)
        row0 = g.iloc[0]
        c_cat = {f: _clean(row0.get(f"case_{f}")) for f in schema.case_cat}
        c_num = {}
        for f in schema.case_num:
            v = row0.get(f"case_{f}")
            try:
                c_num[f] = float(v) if v is not None and str(v) not in MISSING_TOKENS else float("nan")
            except (TypeError, ValueError):
                c_num[f] = float("nan")
        cases.append(
            Case(
                case_id=str(cid),
                activities=acts,
                timestamps=g._ts.to_numpy(dtype=np.float64),
                local_hour=g._hour.to_numpy(dtype=np.int64),
                local_weekday=g._wd.to_numpy(dtype=np.int64),
                event_cat=ev_cat,
                case_cat=c_cat,
                case_num=c_num,
                complete=complete,
                family=schema.family,
            )
        )
    return cases


# ----------------------------------------------------------------------------------------
# Splits — always at the case level; persisted as JSON so every run is reproducible.
# ----------------------------------------------------------------------------------------


@dataclass
class Split:
    name: str
    train: list[str]
    val: list[str]
    test: list[str]
    meta: dict = field(default_factory=dict)

    def assert_disjoint(self) -> None:
        a, b, c = set(self.train), set(self.val), set(self.test)
        assert not (a & b) and not (a & c) and not (b & c), "case leakage across splits"

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=1))

    @staticmethod
    def from_json(path: Path) -> "Split":
        d = json.loads(Path(path).read_text())
        return Split(**d)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for part in (self.train, self.val, self.test):
            h.update("\n".join(part).encode())
            h.update(b"|")
        return h.hexdigest()[:12]


def split_random(cases: list[Case], seed: int = 0, fracs=(0.7, 0.15, 0.15)) -> Split:
    ids = np.array([c.case_id for c in cases])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_tr = int(len(ids) * fracs[0])
    n_va = int(len(ids) * fracs[1])
    tr, va, te = perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]
    s = Split("random", ids[tr].tolist(), ids[va].tolist(), ids[te].tolist(), {"seed": seed, "fracs": fracs})
    s.assert_disjoint()
    return s


def split_chronological(cases: list[Case], fracs=(0.7, 0.15, 0.15), strict: bool = False) -> Split:
    """Order cases by *start* time; earliest → train, latest → test.

    Note: events of late-train cases can overlap in wall-clock with early-test cases (a case
    that starts in the train window may finish in the test window). We accept this mild
    contamination because the alternative — dropping overlapping cases — biases the test set
    toward short cases. The overlap fraction is recorded in ``meta``.
    """
    order = sorted(cases, key=lambda c: (c.start, c.case_id))
    ids = [c.case_id for c in order]
    n_tr = int(len(ids) * fracs[0])
    n_va = int(len(ids) * fracs[1])
    tr, va, te = ids[:n_tr], ids[n_tr : n_tr + n_va], ids[n_tr + n_va :]
    test_start = order[n_tr + n_va].start if n_tr + n_va < len(order) else float("inf")
    overlap = sum(1 for c in order[:n_tr] if c.timestamps[-1] >= test_start) / max(n_tr, 1)
    s = Split(
        "chronological_strict" if strict else "chronological",
        tr,
        va,
        te,
        {
            "fracs": fracs,
            "strict": strict,
            "cutoff_ts": float(test_start) if test_start < float("inf") else None,
            "train_end_start_ts": pd.Timestamp(order[n_tr - 1].start, unit="s", tz="UTC").isoformat(),
            "test_first_start_ts": pd.Timestamp(test_start, unit="s", tz="UTC").isoformat() if test_start < float("inf") else None,
            "frac_train_cases_overlapping_test_window": overlap,
        },
    )
    s.assert_disjoint()
    return s


def split_cv(cases: list[Case], fold: int, n_folds: int = 5, seed: int = 0, val_frac: float = 0.2) -> Split:
    """k-fold cross-validation over cases (published BPI2013 protocol, Rama-Maneiro et al.):
    fold ``fold`` is the test set; the remaining folds are split 80/20 into train/validation."""
    ids = np.array(sorted(c.case_id for c in cases))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    folds = np.array_split(perm, n_folds)
    te = ids[folds[fold]]
    rest = np.concatenate([f for i, f in enumerate(folds) if i != fold])
    rng2 = np.random.default_rng(seed + 1000 + fold)
    rest = rest[rng2.permutation(len(rest))]
    n_va = int(len(rest) * val_frac)
    va, tr = ids[rest[:n_va]], ids[rest[n_va:]]
    s = Split(f"cv{n_folds}_fold{fold}", tr.tolist(), va.tolist(), te.tolist(), {"seed": seed, "fold": fold, "n_folds": n_folds, "val_frac": val_frac})
    s.assert_disjoint()
    return s


def truncate_at_cutoff(cases: list[Case], cutoff: float) -> list[Case]:
    """Strict chronological hygiene: drop every event at/after the cutoff from the given cases (train/val),
    marking truncated cases incomplete; cases with no event before the cutoff are dropped."""
    out = []
    for c in cases:
        keep = c.timestamps < cutoff
        n = int(keep.sum())
        if n == 0:
            continue
        if n == len(c):
            out.append(c)
        else:
            out.append(Case(c.case_id, c.activities[:n], c.timestamps[:n], c.local_hour[:n], c.local_weekday[:n],
                            {k: v[:n] for k, v in c.event_cat.items()}, c.case_cat, c.case_num, False, c.family))
    return out
