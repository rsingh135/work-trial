"""Collect episodes → validated, redacted JSONL traces.

    uv run python -m agentloop.collect --env appworld --split train --n-tasks 30 --runs 1 --out traces/collect_train
    uv run python -m agentloop.collect --env mock --split train --client scripted --out traces/mock_train

Outputs: <out>/episodes.jsonl (one Episode per line), <out>/schemas/<hash>.json (action
schemas), <out>/manifest.json (config, versions, validation stats). Episodes that fail schema
validation are written to <out>/invalid.jsonl (never silently dropped).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from agentloop.agent.llm_agent import AnthropicClient, CannedClient, LLMAgent, ScriptedClient
from agentloop.agent.policy import FirstCandidatePolicy, RerankerPolicy
from agentloop.redaction import RULES_VERSION, Redactor
from agentloop.schema import Episode, Redaction


def make_env(name: str, port: int | None = None, **kw):
    if name == "appworld":
        from agentloop.envs.appworld import AppWorldEnv
        return AppWorldEnv(**({"port": port} if port else {}), **kw)
    if name == "mock":
        from agentloop.envs.mock import MockEnv
        return MockEnv()
    raise ValueError(name)


def make_fork_pool(name: str, n_workers: int, base_port: int = 9200):
    if n_workers <= 0:
        return None
    from agentloop.envs.forking import ForkPool
    return ForkPool(lambda i: make_env(name, port=base_port + i), n_workers=n_workers)


def make_client(kind: str, model_id: str, seed: int = 0, noise: float = 0.3):
    if kind == "anthropic":
        return AnthropicClient(model_id)
    if kind == "scripted":
        return ScriptedClient(noise=noise, seed=seed)
    if kind == "canned":
        return CannedClient()
    raise ValueError(kind)


def make_policy(kind: str, model_path: str | None, n_candidates: int, score_mode: str = "product", validity_model: str | None = None, gate: bool = False):
    from agentloop.agent.policy import GatedPolicy
    inner = _make_inner(kind, model_path, n_candidates, score_mode, validity_model)
    return GatedPolicy(inner) if gate else inner


def _make_inner(kind: str, model_path: str | None, n_candidates: int, score_mode: str = "product", validity_model: str | None = None):
    if kind == "baseline":
        return FirstCandidatePolicy(n_candidates=n_candidates)
    if kind == "oracle":
        from agentloop.agent.policy import OracleLookaheadPolicy
        return OracleLookaheadPolicy(n_candidates=n_candidates)
    if kind == "reranker":
        return RerankerPolicy(model_path, n_candidates=n_candidates, score_mode=score_mode)
    if kind == "tracemodel":
        from agentloop.trace_model import TraceModelPolicy
        return TraceModelPolicy(model_path, n_candidates=n_candidates, score_mode=score_mode, validity_model=validity_model)
    raise ValueError(kind)


def _fatal_llm_error(ep) -> bool:
    """A billing or authentication failure on the very first step means every further episode would be empty."""
    if not ep.steps or not ep.steps[0].error or ep.steps[0].error.type != "llm_error":
        return False
    m = ep.steps[0].error.message.lower()
    return any(k in m for k in ("credit balance", "authentication", "invalid x-api-key", "permission"))


def select_tasks(env, split: str, n_tasks: int | None, seed: int, by_scenario: bool = True) -> list[str]:
    """Pick whole scenarios (all task variants) so scenario-goal completion is measurable."""
    ids = env.task_ids(split)
    if n_tasks is None or n_tasks >= len(ids):
        return ids
    rng = random.Random(seed)
    if by_scenario:
        scen = {}
        for t in ids:
            scen.setdefault(env.scenario_id(t), []).append(t)
        keys = sorted(scen)
        rng.shuffle(keys)
        out = []
        for k in keys:
            if len(out) + len(scen[k]) > n_tasks:
                continue
            out += scen[k]
            if len(out) >= n_tasks:
                break
        return out
    return sorted(rng.sample(ids, n_tasks))


def write_episode(ep: Episode, redactor: Redactor, fh_ok, fh_bad, schemas: dict, out: Path) -> bool:
    before = redactor.count
    data = redactor.obj(ep.model_dump(mode="json"))
    data["redaction"] = Redaction(rules_version=RULES_VERSION, n_redactions=redactor.count - before).model_dump()
    try:
        clean = Episode.model_validate(data)  # validate the *redacted* record: this is what a reader will load
        fh_ok.write(clean.model_dump_json() + "\n")
        fh_ok.flush()
        return True
    except Exception as e:
        fh_bad.write(json.dumps({"error": str(e), "episode": data}, default=str) + "\n")
        fh_bad.flush()
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="appworld")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--tasks", nargs="*", default=None, help="explicit task ids (overrides --n-tasks)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--client", default="anthropic", choices=["anthropic", "scripted", "canned"])
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--policy", default="baseline", choices=["baseline", "reranker", "tracemodel", "oracle"])
    ap.add_argument("--policy-model", default=None)
    ap.add_argument("--validity-model", default=None, help="tracemodel only: also multiply by the token-level validity scorer (hybrid)")
    ap.add_argument("--prompt-version", default="v1", choices=["v1", "v2", "v3"])
    ap.add_argument("--max-tokens", type=int, default=1024, help="LLM output cap per turn (v1/v2 experiments used 1024)")
    ap.add_argument("--fork-workers", type=int, default=0, help="execute every candidate in a forked env copy (counterfactual labels / oracle); 0 = off")
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes (each with its own env server / fork pool)")
    ap.add_argument("--no-step-eval", action="store_true", help="skip per-step evaluator calls (dense progress reward)")
    ap.add_argument("--gate", action="store_true", help="wrap the policy with the schema + progress gates (rule-based, from the published action schema)")
    ap.add_argument("--n-candidates", type=int, default=3)
    ap.add_argument("--score-mode", default="product", choices=["valid", "success", "product", "plausible", "progress"])
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.3, help="scripted client only")
    ap.add_argument("--cost-budget", type=float, default=None, help="USD cap for the whole collection")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / "schemas").mkdir(parents=True, exist_ok=True)
    env = make_env(args.env)
    pool = make_fork_pool(args.env, args.fork_workers)
    tasks = args.tasks or select_tasks(env, args.split, args.n_tasks, args.seed)
    n_cand = args.n_candidates if (args.policy != "baseline" or args.fork_workers > 0) else 1
    policy = make_policy(args.policy, args.policy_model, n_cand, args.score_mode, args.validity_model, args.gate)
    run_id = args.run_id or f"{args.env}-{args.split}-{args.policy}-{time.strftime('%Y%m%d%H%M%S')}"
    redactor = Redactor(salt=run_id)
    schemas: dict = {}
    n_ok = n_bad = 0
    total_cost = 0.0
    manifest = {"args": vars(args), "run_id": run_id, "tasks": tasks, "episodes": []}
    if args.workers > 1:
        from agentloop.parallel import run_parallel
        spec = dict(env=args.env, fork_workers=args.fork_workers, policy=args.policy, policy_model=args.policy_model, n_candidates=n_cand,
                    score_mode=args.score_mode, validity_model=args.validity_model, gate=args.gate, client=args.client, model=args.model, seed=args.seed,
                    noise=args.noise, max_steps=args.max_steps, temperature=args.temperature, prompt_version=args.prompt_version,
                    step_eval=not args.no_step_eval, split=args.split, run_tag=run_id, max_tokens=args.max_tokens)
        jobs = [(r, t) for r in range(args.runs) for t in tasks]
        with open(out / "episodes.jsonl", "a") as fh_ok, open(out / "invalid.jsonl", "a") as fh_bad:
            state = {"ok": 0, "bad": 0, "cost": 0.0}
            def on_ep(r, t, ep):
                ok = write_episode(ep, redactor, fh_ok, fh_bad, schemas, out)
                state["ok"] += ok; state["bad"] += (not ok); state["cost"] += ep.totals.cost_usd
                manifest["episodes"].append({"task_id": t, "run": r, "episode_id": ep.episode_id, "success": ep.final_eval.success, "steps": ep.totals.steps,
                                             "invalid_actions": ep.totals.invalid_actions, "cost_usd": ep.totals.cost_usd, "termination": ep.termination_reason.value, "valid_trace": ok})
                print(f"[{run_id}] run={r} task={t} success={ep.final_eval.success} steps={ep.totals.steps} invalid={ep.totals.invalid_actions} "
                      f"term={ep.termination_reason.value} cost=${ep.totals.cost_usd:.3f} (total ${state['cost']:.2f})", flush=True)
            schemas.update(run_parallel(spec, jobs, args.workers, on_ep))
            n_ok, n_bad, total_cost = state["ok"], state["bad"], state["cost"]
    else:
      with open(out / "episodes.jsonl", "a") as fh_ok, open(out / "invalid.jsonl", "a") as fh_bad:
          for r in range(args.runs):
              seed = args.seed + r
              client = make_client(args.client, args.model, seed=seed, noise=args.noise)
              agent = LLMAgent(client, policy, max_steps=args.max_steps, temperature=args.temperature, prompt_version=args.prompt_version,
                               fork_pool=pool, step_eval=not args.no_step_eval)
              for t in tasks:
                  if args.cost_budget is not None and total_cost >= args.cost_budget:
                      print(f"cost budget {args.cost_budget} reached; stopping")
                      break
                  ep = agent.run_episode(env, t, args.split, run_id=f"{run_id}-r{r}", seed=seed, schema_store=schemas)
                  if _fatal_llm_error(ep):
                      print(f"FATAL LLM error (billing/auth) — stopping: {ep.steps[0].error.message[:120]}"); break
                  ok = write_episode(ep, redactor, fh_ok, fh_bad, schemas, out)
                  n_ok += ok; n_bad += (not ok)
                  total_cost += ep.totals.cost_usd
                  manifest["episodes"].append({"task_id": t, "run": r, "episode_id": ep.episode_id, "success": ep.final_eval.success, "steps": ep.totals.steps,
                                               "invalid_actions": ep.totals.invalid_actions, "cost_usd": ep.totals.cost_usd, "termination": ep.termination_reason.value, "valid_trace": ok})
                  print(f"[{run_id}] run={r} task={t} success={ep.final_eval.success} steps={ep.totals.steps} invalid={ep.totals.invalid_actions} "
                        f"term={ep.termination_reason.value} cost=${ep.totals.cost_usd:.3f} (total ${total_cost:.2f})", flush=True)
    for h, s in schemas.items():
        p = out / "schemas" / f"{h}.json"
        if not p.exists():
            p.write_text(json.dumps(s, indent=1, default=str))
    manifest.update({"n_valid": n_ok, "n_invalid": n_bad, "validation_rate": n_ok / max(n_ok + n_bad, 1), "total_cost_usd": total_cost,
                     "env_version": env.version(), "redaction_rules_version": RULES_VERSION, "n_redactions": redactor.count})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    if pool is not None:
        pool.shutdown()
    if hasattr(env, "shutdown"):
        env.shutdown()
    print(f"wrote {n_ok} valid / {n_bad} invalid episodes to {out}  total cost ${total_cost:.2f}")


if __name__ == "__main__":
    main()
