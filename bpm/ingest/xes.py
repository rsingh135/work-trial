"""Streaming XES → flat event table.

Design notes
------------
* Uses ``lxml.etree.iterparse`` over the (gzipped) XES so memory stays flat regardless of
  log size and we avoid pm4py's heavyweight import. Each ``<trace>`` is materialised, its
  events flattened, then the element is cleared.
* Every attribute is kept as a *string* column prefixed with ``case_`` (trace scope) or
  ``ev_`` (event scope). Typed interpretation (timestamps, floats) is done downstream in
  the per-log schema so that "similarly named fields" are not silently coerced to the same
  semantics across logs. Only ``time:timestamp`` is parsed here because it is the one
  attribute the XES standard fixes.
* Nested attributes (XES ``nested-attributes`` feature) are flattened as ``parent.child``.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd
from lxml import etree

XES_NS = "{http://www.xes-standard.org/}"


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if tag.startswith("{") else tag


def _flatten_attrs(elem: etree._Element, prefix: str = "") -> dict[str, str]:
    """Flatten XES attribute children of ``elem`` into ``{key: value}`` (strings)."""
    out: dict[str, str] = {}
    for child in elem:
        tag = _strip_ns(child.tag)
        if tag in ("event", "trace"):
            continue
        key = child.get("key")
        if key is None:
            continue
        full = f"{prefix}{key}"
        out[full] = child.get("value", "")
        # nested attributes
        if len(child):
            out.update(_flatten_attrs(child, prefix=full + "."))
    return out


@dataclass
class ParsedLog:
    name: str
    path: Path
    events: pd.DataFrame
    log_attrs: dict[str, str] = field(default_factory=dict)
    globals_event: dict[str, str] = field(default_factory=dict)
    globals_trace: dict[str, str] = field(default_factory=dict)
    classifiers: dict[str, list[str]] = field(default_factory=dict)


def _open(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def iter_traces(path: Path) -> Iterator[tuple[dict[str, str], list[dict[str, str]]]]:
    """Yield ``(trace_attrs, [event_attrs, ...])`` for each trace, streaming."""
    with _open(path) as fh:
        for _, elem in etree.iterparse(fh, events=("end",), tag=(f"{XES_NS}trace", "trace")):
            trace_attrs = _flatten_attrs(elem)
            events = []
            for ev in elem:
                if _strip_ns(ev.tag) != "event":
                    continue
                events.append(_flatten_attrs(ev))
            yield trace_attrs, events
            elem.clear()
            # free preceding siblings kept by lxml
            while elem.getprevious() is not None:
                del elem.getparent()[0]


def read_header(path: Path) -> tuple[dict, dict, dict, dict]:
    """Parse log-level attributes, event/trace globals and classifiers (stops at first trace)."""
    log_attrs, g_event, g_trace, classifiers = {}, {}, {}, {}
    with _open(path) as fh:
        for _, elem in etree.iterparse(fh, events=("end",)):
            tag = _strip_ns(elem.tag)
            if tag == "trace":
                break
            if tag == "global":
                target = g_event if elem.get("scope") == "event" else g_trace
                target.update(_flatten_attrs(elem))
            elif tag == "classifier":
                classifiers[elem.get("name")] = elem.get("keys", "").split()
            elif tag in ("string", "date", "int", "float", "boolean") and elem.getparent() is not None:
                parent_tag = _strip_ns(elem.getparent().tag)
                if parent_tag == "log":
                    key = elem.get("key", "")
                    if not key.startswith("meta_"):
                        log_attrs[key] = elem.get("value", "")
    return log_attrs, g_event, g_trace, classifiers


def parse_xes(path: Path, name: str | None = None, max_traces: int | None = None) -> ParsedLog:
    """Parse an XES file into a flat event table.

    Columns: ``log``, ``case_id`` (trace concept:name), ``event_idx`` (0-based order within
    trace as stored in file), ``timestamp`` (tz-aware UTC), ``ev_<key>`` for event attrs,
    ``case_<key>`` for trace attrs.
    """
    path = Path(path)
    name = name or path.name.split(".")[0]
    log_attrs, g_event, g_trace, classifiers = read_header(path)

    rows: list[dict] = []
    for t_i, (trace_attrs, events) in enumerate(iter_traces(path)):
        if max_traces is not None and t_i >= max_traces:
            break
        case_id = trace_attrs.get("concept:name", f"__trace_{t_i}")
        case_cols = {f"case_{k}": v for k, v in trace_attrs.items()}
        for e_i, ev in enumerate(events):
            row = {"log": name, "case_id": case_id, "event_idx": e_i}
            row.update(case_cols)
            row.update({f"ev_{k}": v for k, v in ev.items()})
            rows.append(row)

    df = pd.DataFrame(rows)
    if "ev_time:timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ev_time:timestamp"], utc=True, format="ISO8601")
    else:
        df["timestamp"] = pd.NaT
    df = df.astype({c: "string" for c in df.columns if c.startswith(("ev_", "case_"))})
    return ParsedLog(
        name=name,
        path=path,
        events=df,
        log_attrs=log_attrs,
        globals_event=g_event,
        globals_trace=g_trace,
        classifiers=classifiers,
    )
