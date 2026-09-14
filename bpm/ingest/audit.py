"""Data audit for every registered log.

Produces ``results/audit/<log>.json`` and a human summary ``results/audit/SUMMARY.md``.
Everything the design note says about missingness, censoring, cardinality, time
irregularity and leakage candidates is computed here, not eyeballed.

Run: ``uv run python -m bpm.ingest.audit``
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from bpm.ingest.registry import LOGS, LogSchema
from bpm.ingest.xes import parse_xes

OUT = Path("results/audit")

MISSING_TOKENS = {"", "UNKNOWN", "UNDEFINED", "nan", "None", "-"}


def activity_label(df: pd.DataFrame, schema: LogSchema) -> pd.Series:
    cols = [f"ev_{k}" for k in schema.activity_keys]
    return df[cols].fillna("").astype(str).agg("|".join, axis=1)


def _q(x: pd.Series, qs=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)) -> dict[str, float]:
    x = x.dropna()
    if len(x) == 0:
        return {}
    return {f"p{int(q*100):02d}": float(x.quantile(q)) for q in qs}


def audit_log(schema: LogSchema, max_traces: int | None = None) -> dict:
    L = parse_xes(schema.path, name=schema.name, max_traces=max_traces)
    df = L.events.copy()
    df["activity"] = activity_label(df, schema)
    df = df.sort_values(["case_id", "timestamp", "event_idx"], kind="stable")

    rep: dict = {"log": schema.name, "file": schema.file, "family": schema.family}
    rep["n_cases"] = int(df.case_id.nunique())
    rep["n_events"] = int(len(df))
    rep["log_attrs"] = L.log_attrs
    rep["classifiers"] = L.classifiers
    rep["event_globals"] = sorted(L.globals_event)
    rep["trace_globals"] = sorted(L.globals_trace)

    # --- activity vocabulary
    act_counts = df.activity.value_counts()
    rep["activity_vocab_size"] = int(len(act_counts))
    rep["activity_counts"] = {k: int(v) for k, v in act_counts.items()}
    rep["activity_keys"] = list(schema.activity_keys)

    # --- case length / duration
    g = df.groupby("case_id", sort=False)
    lengths = g.size()
    rep["case_length"] = _q(lengths)
    rep["case_length_mean"] = float(lengths.mean())
    first, last = g.timestamp.min(), g.timestamp.max()
    dur_h = (last - first).dt.total_seconds() / 3600
    rep["case_duration_hours"] = _q(dur_h)
    rep["case_duration_hours_mean"] = float(dur_h.mean())
    rep["log_time_span"] = {"start": str(first.min()), "end": str(last.max())}

    # --- terminal activity distribution + censoring proxies
    last_acts = g.activity.last().value_counts()
    rep["last_activity_counts"] = {k: int(v) for k, v in last_acts.items()}
    first_acts = g.activity.first().value_counts()
    rep["first_activity_counts"] = {k: int(v) for k, v in first_acts.head(10).items()}
    if schema.terminal:
        ends_terminal = g.activity.last().isin(schema.terminal)
        rep["frac_cases_ending_in_terminal"] = float(ends_terminal.mean())
    log_end = last.max()
    days_to_log_end = (log_end - last).dt.total_seconds() / 86400
    rep["cases_ending_within_7d_of_log_end"] = int((days_to_log_end <= 7).sum())
    rep["cases_ending_within_30d_of_log_end"] = int((days_to_log_end <= 30).sum())
    rep["declared_censored"] = schema.censored

    # --- time irregularity
    dt = g.timestamp.diff().dt.total_seconds()
    dt_valid = dt.dropna()
    rep["delta_t_seconds"] = _q(dt_valid)
    rep["frac_zero_delta_t"] = float((dt_valid == 0).mean()) if len(dt_valid) else None
    rep["frac_negative_delta_t_raw_order"] = float(
        (L.events.sort_values(["case_id", "event_idx"]).groupby("case_id").timestamp.diff()
         .dt.total_seconds() < 0).mean()
    )
    ts_missing = df.timestamp.isna().mean()
    rep["frac_missing_timestamp"] = float(ts_missing)
    loc = df.timestamp.dt.tz_convert(schema.local_tz)
    rep["frac_midnight_timestamps_local"] = float(
        ((loc.dt.hour == 0) & (loc.dt.minute == 0) & (loc.dt.second == 0)).mean()
    )
    rep["events_per_weekday_local"] = {int(k): int(v) for k, v in loc.dt.dayofweek.value_counts().sort_index().items()}
    rep["events_per_hour_local"] = {int(k): int(v) for k, v in loc.dt.hour.value_counts().sort_index().items()}
    # per-activity midnight share (which activities are date-only?)
    mid = ((loc.dt.hour == 0) & (loc.dt.minute == 0) & (loc.dt.second == 0))
    rep["midnight_share_by_activity"] = {k: float(v) for k, v in mid.groupby(df.activity).mean().sort_values(ascending=False).head(8).items() if v > 0}

    # --- per-column missingness + cardinality
    cols = [c for c in df.columns if c.startswith(("ev_", "case_"))]
    col_rep = {}
    for c in cols:
        s = df[c]
        n_missing = int(s.isna().sum() + s.isin(MISSING_TOKENS).sum())
        nun = int(s.dropna().nunique())
        entry = {"missing_frac": n_missing / len(df), "cardinality": nun}
        if c.startswith("case_"):
            # is this constant within case? (should be) and how many cases carry it
            per_case = g[c].nunique(dropna=True)
            entry["varies_within_case_frac"] = float((per_case > 1).mean())
            entry["cases_with_value"] = int((g[c].first().notna() & ~g[c].first().isin(MISSING_TOKENS)).sum())
        else:
            per_case = g[c].nunique(dropna=True)
            entry["constant_within_case_frac"] = float((per_case <= 1).mean())
        if nun <= 30:
            entry["top_values"] = {str(k): int(v) for k, v in s.value_counts().head(30).items()}
        else:
            entry["top_values"] = {str(k): int(v) for k, v in s.value_counts().head(5).items()}
            # long-tail share: fraction of events whose value appears < 5 times
            vc = s.value_counts()
            entry["frac_events_with_rare_value_lt5"] = float(s.map(vc).lt(5).mean())
        col_rep[c] = entry
    rep["columns"] = col_rep

    # --- leakage candidates: case attrs whose value correlates with last activity
    leak = {}
    last_act = g.activity.last()
    for c in cols:
        if not c.startswith("case_"):
            continue
        v = g[c].first()
        if v.nunique() < 2 or v.nunique() > 200:
            continue
        ct = pd.crosstab(v, last_act)
        # normalised mutual information proxy: how predictable is the terminal from attr
        p_xy = ct / ct.values.sum()
        p_x = p_xy.sum(axis=1).values[:, None]
        p_y = p_xy.sum(axis=0).values[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.nansum(p_xy.values * np.log(p_xy.values / (p_x * p_y)))
        h_y = -np.nansum(p_y * np.log(p_y))
        leak[c] = {"nmi_with_last_activity": float(mi / h_y) if h_y > 0 else 0.0}
    rep["case_attr_vs_terminal_nmi"] = dict(sorted(leak.items(), key=lambda kv: -kv[1]["nmi_with_last_activity"]))

    # --- variants
    variants = g.activity.agg(tuple)
    vc = Counter(variants)
    rep["n_variants"] = len(vc)
    rep["top_variant_share"] = sum(v for _, v in vc.most_common(5)) / len(variants)
    rep["top_variants"] = [{"variant": " > ".join(k), "count": v} for k, v in vc.most_common(5)]

    # --- registry decisions echoed for traceability
    rep["registry"] = {
        "event_cat": list(schema.event_cat),
        "event_highcard": list(schema.event_highcard),
        "case_cat": list(schema.case_cat),
        "case_num": list(schema.case_num),
        "excluded": schema.excluded,
        "terminal": list(schema.terminal),
        "notes": schema.notes,
    }
    return rep


def write_summary(reports: list[dict]) -> str:
    lines = ["# Data audit summary", "", "Generated by `bpm/ingest/audit.py`. Full detail per log in the JSON files.", ""]
    lines.append("| log | cases | events | acts | variants | len p50/p95 | dur h p50/p95 | zero-Δt | midnight | ends-terminal | ≤30d of log end |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in reports:
        lines.append(
            f"| {r['log']} | {r['n_cases']} | {r['n_events']} | {r['activity_vocab_size']} | {r['n_variants']} | "
            f"{r['case_length'].get('p50', 0):.0f}/{r['case_length'].get('p95', 0):.0f} | "
            f"{r['case_duration_hours'].get('p50', 0):.0f}/{r['case_duration_hours'].get('p95', 0):.0f} | "
            f"{(r['frac_zero_delta_t'] or 0):.2f} | {r['frac_midnight_timestamps_local']:.2f} | "
            f"{r.get('frac_cases_ending_in_terminal', float('nan')):.2f} | {r['cases_ending_within_30d_of_log_end']} |"
        )
    lines.append("")
    for r in reports:
        lines.append(f"## {r['log']}")
        lines.append(f"- span: {r['log_time_span']['start'][:10]} → {r['log_time_span']['end'][:10]}")
        lines.append(f"- activities ({r['activity_vocab_size']}): " + ", ".join(f"`{k}`={v}" for k, v in list(r["activity_counts"].items())[:25]))
        lines.append("- last activity: " + ", ".join(f"`{k}`={v}" for k, v in list(r["last_activity_counts"].items())[:8]))
        lines.append(f"- top-5 variants cover {r['top_variant_share']:.0%} of cases")
        if r["midnight_share_by_activity"]:
            lines.append("- date-only (midnight local) activities: " + ", ".join(f"`{k}`={v:.0%}" for k, v in r["midnight_share_by_activity"].items()))
        hi = [(c, e["cardinality"]) for c, e in r["columns"].items() if e["cardinality"] > 30]
        lines.append("- high-cardinality columns: " + ", ".join(f"`{c}`({n})" for c, n in sorted(hi, key=lambda x: -x[1])[:10]))
        miss = [(c, e["missing_frac"]) for c, e in r["columns"].items() if e["missing_frac"] > 0.05]
        lines.append("- columns with >5% missing/UNKNOWN: " + ", ".join(f"`{c}`({m:.0%})" for c, m in sorted(miss, key=lambda x: -x[1])[:12]))
        leak = list(r["case_attr_vs_terminal_nmi"].items())[:5]
        lines.append("- case attrs most predictive of terminal activity (NMI): " + ", ".join(f"`{c}`={v['nmi_with_last_activity']:.2f}" for c, v in leak))
        lines.append("")
    return "\n".join(lines)


def main(max_traces: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, schema in LOGS.items():
        rep = audit_log(schema, max_traces=max_traces)
        (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1, default=str))
        reports.append(rep)
        print(f"audited {name}: {rep['n_cases']} cases, {rep['n_events']} events, {rep['activity_vocab_size']} activities")
    (OUT / "SUMMARY.md").write_text(write_summary(reports))


if __name__ == "__main__":
    main()
