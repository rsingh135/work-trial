"""Baseline vs. trained-agent comparison on held-out tasks (same tasks, same LLM, same seeds).

    uv run python -m agentloop.evaluate --env appworld --split dev --n-tasks 30 --runs 2 \
        --policy-model models/validity_v1/model.pkl --out results/part2_eval

Runs the baseline policy and the reranker policy over the same task list for the same number
of runs, writes both trace sets (redacted, validated) and a summary JSON with: task-goal
completion (TGC), scenario-goal completion (SGC: every variant of a scenario succeeded in a
run), invalid-action rate, steps and wall-clock per episode, tokens/cost, per-run variation,
trace validation rate, and the reranker's override rate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from agentloop.agent.llm_agent import LLMAgent
from agentloop.collect import make_client, make_env, make_policy, select_tasks, write_episode
from agentloop.redaction import Redactor
from agentloop.schema import Episode


def _fatal_llm_error(ep) -> bool:
    if not ep.steps or not ep.steps[0].error or ep.steps[0].error.type != "llm_error":
        return False
    m = ep.steps[0].error.message.lower()
    return any(k in m for k in ("credit balance", "authentication", "invalid x-api-key", "permission"))


def summarize(episodes: list[Episode]) -> dict:
    by_run: dict[str, list[Episode]] = {}
    for e in episodes:
        by_run.setdefault(e.run_id, []).append(e)
    per_run = []
    for rid, eps in sorted(by_run.items()):
        succ = [bool(e.final_eval and e.final_eval.success) for e in eps]
        scen: dict[str, list[bool]] = {}
        for e, s in zip(eps, succ):
            scen.setdefault(e.scenario_id or e.task_id, []).append(s)
        n_steps = sum(e.totals.steps for e in eps)
        per_run.append({
            "run_id": rid, "n_episodes": len(eps), "tgc": sum(succ) / len(eps),
            "sgc": sum(all(v) for v in scen.values()) / len(scen),
            "invalid_action_rate": sum(e.totals.invalid_actions for e in eps) / max(n_steps, 1),
            "mean_steps": n_steps / len(eps), "mean_wall_s": sum(e.totals.wall_s for e in eps) / len(eps),
            "mean_cost_usd": sum(e.totals.cost_usd for e in eps) / len(eps),
            "termination": {t: sum(1 for e in eps if e.termination_reason and e.termination_reason.value == t) for t in ("completed", "max_steps", "error", "budget")},
            "override_rate": (sum(1 for e in eps for s in e.steps if s.policy_meta.get("overrode_first")) / max(n_steps, 1)),
        })
    agg = {}
    for k in ("tgc", "sgc", "invalid_action_rate", "mean_steps", "mean_wall_s", "mean_cost_usd", "override_rate"):
        vals = [r[k] for r in per_run]
        agg[k] = {"mean": statistics.mean(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0, "runs": vals}
    return {"n_episodes": len(episodes), "n_runs": len(per_run), "per_run": per_run, **agg}


def run_policy(env, tasks, split, policy_kind, policy_model, n_candidates, client_kind, model_id, runs, seed, max_steps, temperature, out, noise, run_tag, score_mode="product", validity_model=None, gate=False, prompt_version="v1"):
    out.mkdir(parents=True, exist_ok=True)
    (out / "schemas").mkdir(exist_ok=True)
    policy = make_policy(policy_kind, policy_model, n_candidates, score_mode, validity_model, gate)
    redactor = Redactor(salt=run_tag)
    schemas: dict = {}
    eps, n_ok, n_bad = [], 0, 0
    with open(out / "episodes.jsonl", "w") as fh_ok, open(out / "invalid.jsonl", "w") as fh_bad:
        for r in range(runs):
            client = make_client(client_kind, model_id, seed=seed + r, noise=noise)
            agent = LLMAgent(client, policy, max_steps=max_steps, temperature=temperature, prompt_version=prompt_version)
            for t in tasks:
                ep = agent.run_episode(env, t, split, run_id=f"{run_tag}-r{r}", seed=seed + r, schema_store=schemas)
                if _fatal_llm_error(ep):
                    raise SystemExit(f"FATAL LLM error (billing/auth) — stopping: {ep.steps[0].error.message[:120]}")
                ok = write_episode(ep, redactor, fh_ok, fh_bad, schemas, out)
                n_ok += ok; n_bad += (not ok)
                eps.append(ep)
                print(f"[{run_tag}] run={r} task={t} success={ep.final_eval.success} steps={ep.totals.steps} invalid={ep.totals.invalid_actions} cost=${ep.totals.cost_usd:.3f}", flush=True)
    for h, s in schemas.items():
        (out / "schemas" / f"{h}.json").write_text(json.dumps(s, indent=1, default=str))
    summ = summarize(eps)
    summ["trace_validation"] = {"valid": n_ok, "invalid": n_bad, "rate": n_ok / max(n_ok + n_bad, 1)}
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="appworld")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--client", default="anthropic", choices=["anthropic", "scripted"])
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--policy-model", default=None)
    ap.add_argument("--validity-model", default=None, help="tracemodel only: also multiply by the token-level validity scorer (hybrid)")
    ap.add_argument("--prompt-version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--gate", action="store_true", help="wrap the policy with the schema + progress gates (rule-based, from the published action schema)")
    ap.add_argument("--policy", default="reranker", choices=["reranker", "tracemodel", "baseline"], help="which component to reinsert (baseline + --gate = rule-based control)")
    ap.add_argument("--n-candidates", type=int, default=3)
    ap.add_argument("--score-mode", default="product", choices=["valid", "success", "product", "plausible"])
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    env = make_env(a.env)
    tasks = a.tasks or select_tasks(env, a.split, a.n_tasks, a.seed)
    t0 = time.time()
    common = dict(env=env, tasks=tasks, split=a.split, client_kind=a.client, model_id=a.model, runs=a.runs, seed=a.seed, max_steps=a.max_steps, temperature=a.temperature, noise=a.noise, prompt_version=a.prompt_version)
    res = {"args": vars(a), "tasks": tasks, "n_tasks": len(tasks)}
    if not a.skip_baseline:
        res["baseline"] = run_policy(policy_kind="baseline", policy_model=None, n_candidates=1, out=out / "baseline", run_tag=f"eval-{a.env}-baseline", **common)
    res["reranker"] = run_policy(policy_kind=a.policy, policy_model=a.policy_model, n_candidates=a.n_candidates, out=out / "reranker", run_tag=f"eval-{a.env}-reranker", score_mode=a.score_mode, validity_model=a.validity_model, gate=a.gate, **common)
    if "baseline" in res:
        res["delta"] = {k: res["reranker"][k]["mean"] - res["baseline"][k]["mean"] for k in ("tgc", "sgc", "invalid_action_rate", "mean_steps", "mean_wall_s", "mean_cost_usd")}
    res["total_seconds"] = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    if hasattr(env, "shutdown"):
        env.shutdown()
    print(json.dumps({k: v for k, v in res.items() if k in ("delta",)} | {p: {k: res[p][k]["mean"] for k in ("tgc", "sgc", "invalid_action_rate", "mean_steps")} for p in ("baseline", "reranker") if p in res}, indent=1))


if __name__ == "__main__":
    main()
