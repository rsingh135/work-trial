"""Episodes → prefix/target training examples for the action-validity scorer.

    uv run python -m agentloop.build_dataset traces/collect_train --out datasets/validity

One example per executed step: context (instruction, last observation, last error, last code,
step index, #previous errors) → label_valid (1 if the executed action produced no error).
label_success carries the episode outcome for secondary analysis. Steps with a harness-level
error (llm_error / parse_error) are kept with label_valid=0 — they are failed actions too.

Examples are split by *group* (scenario id, falling back to task id) into train/heldout so
the offline metric is measured on scenarios never seen in training. Provenance links every
example to (trace file, episode, step, candidate).
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np
import json
import random
from pathlib import Path

from agentloop.schema import Episode, Provenance, TrainingExample

BUILDER_VERSION = "2"


def episode_examples(ep: Episode, trace_file: str) -> list[TrainingExample]:
    out = []
    n_prev_errors = 0
    last_obs, last_err, last_code = ep.task_instruction, None, None
    for s in ep.steps:
        cand = s.candidates[s.chosen_index]
        code = cand.action.get("code", "")
        ctx = {"instruction": ep.task_instruction, "last_observation": last_obs or "", "last_error": last_err or "", "last_code": last_code or "",
               "step_index": str(s.step_index), "n_prev_errors": str(n_prev_errors)}
        succ = None if ep.final_eval is None or ep.final_eval.success is None else int(ep.final_eval.success)
        prog = None if s.reward is None else int(s.reward > 0)
        ev = s.evaluator_output or {}
        regr = None if s.reward is None else int(s.reward < 0 or bool(s.success_state and ev.get("failures", 0) > 0))
        eid = hashlib.sha256(f"{ep.episode_id}:{s.step_index}:{s.chosen_index}".encode()).hexdigest()[:16]
        out.append(TrainingExample(
            example_id=eid, group_id=ep.scenario_id or ep.task_id, split=ep.split, context=ctx, action={"type": cand.action.get("type", "none"), "code": code},
            label_valid=int(s.error is None and bool(code)), label_success=succ, label_progress=prog, label_regress=regr, source="executed",
            step_index=s.step_index,
            provenance=Provenance(episode_id=ep.episode_id, step_index=s.step_index, candidate_index=s.chosen_index, schema_version=ep.schema_version,
                                  builder_version=BUILDER_VERSION, trace_file=trace_file),
        ))
        # counterfactual candidates (executed in forks): same context, their own labels
        for k, c in enumerate(s.candidates):
            cf = c.counterfactual
            if k == s.chosen_index or cf is None:
                continue
            ccode = c.action.get("code", "")
            delta = cf.passes_after - cf.passes_before
            out.append(TrainingExample(
                example_id=hashlib.sha256(f"{ep.episode_id}:{s.step_index}:{k}".encode()).hexdigest()[:16], group_id=ep.scenario_id or ep.task_id, split=ep.split,
                context=ctx, action={"type": c.action.get("type", "none"), "code": ccode},
                label_valid=int(cf.error is None and bool(ccode)), label_success=None, label_progress=int(delta > 0), label_regress=int(delta < 0 or (cf.done and cf.failures_after > 0)),
                source="counterfactual", step_index=s.step_index,
                provenance=Provenance(episode_id=ep.episode_id, step_index=s.step_index, candidate_index=k, schema_version=ep.schema_version,
                                      builder_version=BUILDER_VERSION, trace_file=trace_file),
            ))
        last_obs, last_err, last_code = s.tool_result, (s.error.message if s.error else None), code
        if s.error:
            n_prev_errors += 1
    return out


def build(trace_dirs: list[Path], out: Path, heldout_frac: float = 0.25, seed: int = 0) -> dict:
    examples: list[TrainingExample] = []
    n_ep = 0
    for d in trace_dirs:
        f = d / "episodes.jsonl"
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            ep = Episode.from_jsonl_line(line)
            n_ep += 1
            examples += episode_examples(ep, str(f))
    groups = sorted({e.group_id for e in examples})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_held = max(1, int(len(groups) * heldout_frac)) if len(groups) > 1 else 0
    held = set(groups[:n_held])
    out.mkdir(parents=True, exist_ok=True)
    stats = {"n_episodes": n_ep, "n_examples": len(examples), "n_groups": len(groups), "heldout_groups": sorted(held),
             "pos_rate": sum(e.label_valid for e in examples) / max(len(examples), 1), "builder_version": BUILDER_VERSION,
             "n_counterfactual": sum(1 for e in examples if e.source == "counterfactual"),
             "progress_rate": float(np.mean([e.label_progress for e in examples if e.label_progress is not None])) if any(e.label_progress is not None for e in examples) else None,
             "regress_rate": float(np.mean([e.label_regress for e in examples if e.label_regress is not None])) if any(e.label_regress is not None for e in examples) else None}
    with open(out / "train.jsonl", "w") as ftr, open(out / "heldout.jsonl", "w") as fho:
        for e in examples:
            (fho if e.group_id in held else ftr).write(e.model_dump_json() + "\n")
    stats["n_train"] = sum(1 for e in examples if e.group_id not in held)
    stats["n_heldout"] = len(examples) - stats["n_train"]
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    stats = build([Path(d) for d in a.trace_dirs], Path(a.out), a.heldout_frac, a.seed)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
