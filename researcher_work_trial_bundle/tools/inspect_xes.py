#!/usr/bin/env python3
"""Print a compact, dependency-free inventory for one or more XES(.gz) logs."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def open_log(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def inspect(path: Path) -> dict:
    trace_count = 0
    event_count = 0
    lengths: list[int] = []
    trace_fields: dict[str, Counter] = defaultdict(Counter)
    event_fields: dict[str, Counter] = defaultdict(Counter)
    activities: Counter = Counter()
    activity_lifecycle_pairs: Counter = Counter()
    timestamps: list[datetime] = []

    with open_log(path) as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if local_name(element.tag) != "trace":
                continue
            trace_count += 1
            trace_events = 0
            for child in element:
                child_type = local_name(child.tag)
                if child_type == "event":
                    trace_events += 1
                    event_values: dict[str, str] = {}
                    for attr in child:
                        key = attr.attrib.get("key")
                        value = attr.attrib.get("value")
                        if not key:
                            continue
                        event_fields[key][local_name(attr.tag)] += 1
                        if value is not None:
                            event_values[key] = value
                        if key == "concept:name" and value is not None:
                            activities[value] += 1
                        elif key == "time:timestamp" and value:
                            parsed = parse_timestamp(value)
                            if parsed is not None:
                                timestamps.append(parsed)
                    if "concept:name" in event_values and "lifecycle:transition" in event_values:
                        pair = f'{event_values["concept:name"]} | {event_values["lifecycle:transition"]}'
                        activity_lifecycle_pairs[pair] += 1
                else:
                    key = child.attrib.get("key")
                    if key:
                        trace_fields[key][child_type] += 1
            lengths.append(trace_events)
            event_count += trace_events
            element.clear()

    return {
        "file": path.name,
        "traces": trace_count,
        "events": event_count,
        "activities": len(activities),
        "activity_lifecycle_pairs": len(activity_lifecycle_pairs),
        "min_trace_length": min(lengths, default=0),
        "mean_trace_length": round(statistics.fmean(lengths), 3) if lengths else 0,
        "max_trace_length": max(lengths, default=0),
        "time_start": min(timestamps).isoformat() if timestamps else None,
        "time_end": max(timestamps).isoformat() if timestamps else None,
        "trace_fields": {key: dict(types) for key, types in sorted(trace_fields.items())},
        "event_fields": {key: dict(types) for key, types in sorted(event_fields.items())},
        "activity_counts": dict(activities.most_common()),
        "activity_lifecycle_pair_counts": dict(activity_lifecycle_pairs.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    results = [inspect(path) for path in args.paths]
    print(json.dumps(results, indent=2 if args.pretty else None, ensure_ascii=False))


if __name__ == "__main__":
    main()
