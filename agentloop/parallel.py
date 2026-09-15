"""Parallel episode execution: N worker processes, each holding its own environment (own AppWorld
server port), fork pool, policy and LLM clients for the whole run; jobs = (run index, task id) are
streamed to the pool one at a time so progress is visible and results are validated/redacted/written
by the parent exactly as in the serial path. AppWorld allows one active world per process, so
parallelism is process-level with distinct ports (bundle's parallelizing-worlds note)."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

BASE_PORT = 9300
PORT_STRIDE = 20
_G: dict[str, Any] = {}


def _init(spec: dict[str, Any], wid_queue) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from agentloop.collect import make_client, make_env, make_fork_pool, make_policy
    import atexit
    wid = wid_queue.get()
    port = spec.get("port_base", BASE_PORT) + PORT_STRIDE * wid
    _G["env"] = make_env(spec["env"], port=port)
    _G["pool"] = make_fork_pool(spec["env"], spec["fork_workers"], base_port=port + 1) if spec.get("fork_mode", "snapshot") == "replay" else None
    _G["fork_on"] = spec["fork_workers"] > 0

    def _cleanup():  # workers exit without the parent's shutdown path; orphaned servers would keep the ports busy
        try:
            if _G.get("pool") is not None:
                _G["pool"].shutdown()
            if hasattr(_G.get("env"), "shutdown"):
                _G["env"].shutdown()
        except Exception:
            pass
    atexit.register(_cleanup)
    _G["policy"] = make_policy(spec["policy"], spec["policy_model"], spec["n_candidates"], spec["score_mode"], spec["validity_model"], spec["gate"])
    _G["clients"] = {}
    _G["spec"] = spec
    _G["schemas"] = {}
    _G["make_client"] = make_client


def _run_job(r: int, task: str) -> tuple[int, str, str, dict]:
    from agentloop.agent.llm_agent import LLMAgent
    spec = _G["spec"]
    if r not in _G["clients"]:
        _G["clients"][r] = _G["make_client"](spec["client"], spec["model"], seed=spec["seed"] + r, noise=spec["noise"])
    agent = LLMAgent(_G["clients"][r], _G["policy"], max_steps=spec["max_steps"], temperature=spec["temperature"],
                     prompt_version=spec["prompt_version"], fork_pool=_G["pool"], step_eval=spec["step_eval"], max_tokens=spec.get("max_tokens", 1024),
                     fork_mode=spec.get("fork_mode", "snapshot") if _G["fork_on"] else "replay", memory=spec.get("memory", "raw"))
    ep = agent.run_episode(_G["env"], task, spec["split"], run_id=f"{spec['run_tag']}-r{r}", seed=spec["seed"] + r, schema_store=_G["schemas"])
    return r, task, ep.model_dump_json(), dict(_G["schemas"])


def run_parallel(spec: dict[str, Any], jobs: list[tuple[int, str]], n_workers: int, on_episode) -> dict:
    """Run jobs across n_workers processes; call on_episode(run, task, Episode) as each finishes. Returns action schemas."""
    from agentloop.schema import Episode
    ctx = mp.get_context("spawn")
    q = ctx.Manager().Queue()
    for wid in range(n_workers):
        q.put(wid)
    schemas: dict = {}
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_init, initargs=(spec, q)) as ex:
        futs = [ex.submit(_run_job, r, t) for r, t in jobs]
        for fut in as_completed(futs):
            r, task, ep_json, sch = fut.result()
            schemas.update(sch)
            on_episode(r, task, Episode.model_validate_json(ep_json))
    return schemas
