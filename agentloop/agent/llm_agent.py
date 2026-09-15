"""LLM code-agent loop that emits schema-conformant episodes.

Benchmark-agnostic: the env supplies the task text and executes actions; the prompt only
assumes the action language is "one python code block per turn". ``LLMClient`` is a tiny
protocol so a scripted client can drive the loop in tests / the no-API-key demo.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from agentloop.envs.base import Env
from agentloop.schema import (Candidate, Counterfactual, Episode, ErrorInfo, FinalEval, Step, TerminationReason, Totals, Usage,
                              Versions, content_hash, now)

# USD per million tokens: (input, output, cache_write, cache_read)
PRICES = {
    "claude-haiku-4-5": (1.0, 5.0, 1.25, 0.10),
    "claude-haiku-4-5-20251001": (1.0, 5.0, 1.25, 0.10),
    "claude-sonnet-5": (3.0, 15.0, 3.75, 0.30),
}

SYSTEM_PROMPT_V1 = """You are an autonomous assistant completing tasks for your supervisor inside a sandboxed "app world".
You act ONLY by writing Python code. Each turn, reply with exactly one ```python code block (nothing else). The code runs in a persistent Python shell; variables survive between turns. The environment prints the output of your code (or the error traceback) back to you.

Interface:
- `apis.api_docs.show_app_descriptions()` lists the apps. `apis.api_docs.show_api_descriptions(app_name=...)` lists an app's APIs. `apis.api_docs.show_api_doc(app_name=..., api_name=...)` shows one API's full doc (parameters, response schema). Read docs before calling an API you have not used yet.
- Most apps need login: `apis.supervisor.show_account_passwords()` gives passwords; `token = apis.<app>.login(username=<supervisor email or phone as the doc says>, password=...)['access_token']`; pass `access_token=token` to that app's other calls.
- `apis.supervisor.show_profile()` / `show_addresses()` / `show_payment_cards()` give supervisor details when needed.
- Use `print(...)` to see results. Keep each turn small (one or two API calls) so errors are easy to diagnose. Paginate (`page_index`) when lists may be long.
- When the task is done, call `apis.supervisor.complete_task()`; if the task asks a question, call `apis.supervisor.complete_task(answer=<answer>)` with the plain value only (a string/number), not a sentence. Do not call complete_task until you have actually done the work.
- Never invent API names or parameters. Never ask the user questions; there is no user."""

# v2: same prompt with three bug fixes found in the v1 traces (invented API names, forbidden `import`,
# premature complete_task). Selected per run via LLMAgent(prompt_version=...); the prompt hash is in every trace.
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT_V1.replace(
    "- Never invent API names or parameters. Never ask the user questions; there is no user.",
    "- Never invent API names or parameters: call ONLY names returned by `show_api_descriptions`; if a call fails with "
    "\"No APIs with name\", list the app's APIs and pick from that list.\n"
    "- Never `import` anything; `apis`, `json`, `datetime` etc. are already available. Write plain statements, no multi-line string literals.\n"
    "- Do NOT call `complete_task` in your first turn, and never before you have printed the concrete result you are going to report "
    "(or performed the requested change and seen a success response).\n"
    "- Never ask the user questions; there is no user.")
PROMPTS = {"v1": SYSTEM_PROMPT_V1, "v2": SYSTEM_PROMPT_V2}
SYSTEM_PROMPT = SYSTEM_PROMPT_V1

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def parse_code(text: str) -> tuple[str | None, str | None]:
    m = _CODE_BLOCK.findall(text)
    if m:
        return m[-1].strip(), None
    t = text.strip()
    if t and ("apis." in t or t.startswith("print(")):
        return t, "no_fence"
    return None, "no_code_block"


@dataclass
class LLMResponse:
    text: str
    usage: Usage


class LLMClient(Protocol):
    model_id: str

    def complete(self, system: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> LLMResponse: ...


class AnthropicClient:
    def __init__(self, model_id: str = "claude-haiku-4-5", max_retries: int = 3, timeout_s: float = 120.0):
        import anthropic  # local import so the package is optional in offline mode
        self.model_id = model_id
        # bounded per-attempt timeout: the SDK default (10 min × retries) let one stalled request freeze an
        # evaluation for ~45 min; a failed call is recorded as an llm_error step, not a crash
        self._client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout_s)

    # Models that still accept sampling parameters (anthropic SDK 1.x removed `temperature` as a named
    # argument; Opus 5 / Sonnet 5 / Fable reject it with a 400, Haiku 4.5 accepts it via extra_body).
    SAMPLING_OK = ("claude-haiku-4-5",)

    def complete(self, system, messages, temperature, max_tokens) -> LLMResponse:
        t0 = time.time()
        extra = {"temperature": temperature} if self.model_id.startswith(self.SAMPLING_OK) else {}
        import os
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")  # required when the API key is org-scoped rather than workspace-scoped
        kw = {"workspace_id": ws} if ws else {}
        r = self._client.messages.create(
            model=self.model_id, max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages, extra_body=extra, **kw,
        )
        u = r.usage
        p = PRICES.get(self.model_id, (0, 0, 0, 0))
        cr, cw = getattr(u, "cache_read_input_tokens", 0) or 0, getattr(u, "cache_creation_input_tokens", 0) or 0
        cost = (u.input_tokens * p[0] + u.output_tokens * p[1] + cw * p[2] + cr * p[3]) / 1e6
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return LLMResponse(text, Usage(input_tokens=u.input_tokens, output_tokens=u.output_tokens, cache_read_tokens=cr, cache_write_tokens=cw, cost_usd=cost, latency_s=time.time() - t0))


class ScriptedClient:
    """Deterministic stand-in for tests and the offline demo (drives MockEnv).
    Behaviour: list keys → fetch the key from the instruction → complete. With probability
    ``noise`` a step instead emits a wrong/invalid action so traces contain failures."""
    model_id = "scripted-v1"

    def __init__(self, noise: float = 0.3, seed: int = 0):
        import random
        self.rng = random.Random(seed)
        self.noise = noise

    def complete(self, system, messages, temperature, max_tokens) -> LLMResponse:
        instr = messages[0]["content"]
        key = re.search(r"key (\w+)", instr).group(1)
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        last = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
        bad = self.rng.random() < self.noise * (1 if temperature > 0 else 0.5)
        if bad:
            code = self.rng.choice([f"print(apis.store.get('{key}x'))", "print(apis.storage.get('a'))", "apis.supervisor.complete_task(answer='wrong')"])
        elif n_assistant == 0:
            code = "print(apis.store.keys())"
        elif "Execution failed" in last or n_assistant == 1:
            code = f"print(apis.store.get('{key}'))"
        else:
            m = re.fullmatch(r"(v\d+\d)", last.strip())
            code = f"apis.supervisor.complete_task(answer='{m.group(1)}')" if m else f"print(apis.store.get('{key}'))"
        return LLMResponse(f"```python\n{code}\n```", Usage(input_tokens=50, output_tokens=10, latency_s=0.0))


class CannedClient:
    """Replays a fixed list of code actions (default: an AppWorld docs → passwords → complete
    script). Used to dry-run a real environment adapter without an API key; exercises redaction
    on genuine password output."""
    model_id = "canned-v1"
    DEFAULT = ["print(apis.api_docs.show_app_descriptions())", "print(apis.supervisor.show_account_passwords())",
               "print(apis.spotify.login(username='nobody', password='wrong'))", "apis.supervisor.complete_task(answer='unknown')"]

    def __init__(self, script: list[str] | None = None):
        self.script = script or self.DEFAULT

    def complete(self, system, messages, temperature, max_tokens) -> LLMResponse:
        i = sum(1 for m in messages if m["role"] == "assistant")
        code = self.script[min(i, len(self.script) - 1)]
        return LLMResponse(f"```python\n{code}\n```", Usage(input_tokens=0, output_tokens=0))


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class LLMAgent:
    def __init__(self, client: LLMClient, policy, max_steps: int = 25, temperature: float = 0.7, max_tokens: int = 1024,
                 history_turns: int = 12, max_obs_chars: int = 2500, cost_budget_usd: float | None = None, prompt_version: str = "v1",
                 fork_pool=None, step_eval: bool = True):
        """fork_pool: optional ForkPool; when given, every sampled candidate is executed in a forked copy of the
        environment before the policy chooses (counterfactual labels; needed by the oracle lookahead policy).
        step_eval: call env.evaluate() after every step and record the evaluator's pass count (dense progress)."""
        self.client, self.policy = client, policy
        self.fork_pool, self.step_eval = fork_pool, step_eval
        self.system_prompt = PROMPTS[prompt_version]
        self.max_steps, self.temperature, self.max_tokens = max_steps, temperature, max_tokens
        self.history_turns, self.max_obs_chars = history_turns, max_obs_chars
        self.cost_budget = cost_budget_usd
        self.prompt_hash = hashlib.sha256(self.system_prompt.encode()).hexdigest()[:12]

    def _messages(self, task_text: str, turns: list[tuple[str, str]]) -> list[dict[str, str]]:
        msgs = [{"role": "user", "content": task_text}]
        kept = turns[-self.history_turns:]
        dropped = len(turns) - len(kept)
        if dropped:
            msgs[0]["content"] += f"\n\n[{dropped} earlier turns omitted; variables defined there still exist.]"
        for code, out in kept:
            msgs.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            msgs.append({"role": "user", "content": out[: self.max_obs_chars]})
        return msgs

    def run_episode(self, env: Env, task_id: str, split: str, run_id: str, seed: int | None = None, schema_store: dict | None = None) -> Episode:
        t_start = time.time()
        obs = env.reset(task_id, run_tag=run_id, seed=seed)
        schema = env.action_schema()
        schema_ref = content_hash(schema)
        if schema_store is not None:
            schema_store.setdefault(schema_ref, schema)
        ep = Episode(
            env_id=env.name, benchmark=env.benchmark, task_id=task_id, scenario_id=env.scenario_id(task_id),
            episode_id=uuid.uuid4().hex[:16], run_id=run_id, split=split,
            versions=Versions(env_version=env.version(), benchmark_version=env.version(), model_id=self.client.model_id, prompt_hash=self.prompt_hash,
                              harness_git_sha=git_sha(), policy_id=self.policy.policy_id, policy_version=str(getattr(self.policy, "version", "none")), seed=seed),
            started_at=now(), task_instruction=obs.task_meta.get("instruction", obs.text), task_meta=obs.task_meta,
        )
        turns: list[tuple[str, str]] = []
        history: list[dict] = []  # structured trajectory so far, for sequence-model policies
        executed_actions: list[dict] = []  # for replay-forking
        current_obs = obs.text
        last_error, last_code, n_prev_errors, invalid = None, None, 0, 0
        termination = TerminationReason.max_steps
        cost = 0.0
        passes = None
        if self.step_eval:
            try:
                ev0 = env.evaluate()
                passes = len(ev0.passes)
            except Exception:
                passes = None
        for i in range(self.max_steps):
            msgs = self._messages(obs.text, turns)
            candidates: list[Candidate] = []
            codes: list[str] = []
            step_usage = Usage()
            llm_error = None
            for k in range(self.policy.n_candidates):
                try:
                    r = self.client.complete(self.system_prompt, msgs, self.temperature, self.max_tokens)
                except Exception as e:  # network / API errors are part of the trace, not a crash
                    llm_error = f"{type(e).__name__}: {e}"[:300]
                    break
                code, perr = parse_code(r.text)
                candidates.append(Candidate(action_raw=r.text, action={"type": "execute_code", "code": code or "", "api_calls": []}, usage=r.usage, parse_error=perr if code is None else None))
                codes.append(code or "")
                for f in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "cost_usd", "latency_s"):
                    setattr(step_usage, f, getattr(step_usage, f) + getattr(r.usage, f))
            cost += step_usage.cost_usd
            context = {"instruction": ep.task_instruction, "last_observation": current_obs, "last_error": last_error, "last_code": last_code,
                       "step_index": i, "n_prev_errors": n_prev_errors, "history": list(history), "benchmark": env.benchmark,
                       "action_schema": schema}
            if llm_error or not candidates:
                ep.steps.append(Step(step_index=i, timestamp=now(), observation=current_obs, action_schema_ref=schema_ref,
                                     candidates=[Candidate(action_raw="", action={"type": "none"})], chosen_index=0, tool_result="",
                                     error=ErrorInfo(type="llm_error", message=llm_error or "no candidates"), usage=step_usage))
                termination = TerminationReason.error
                break
            # counterfactual execution of every candidate in a forked copy (label-time / lookahead)
            if self.fork_pool is not None:
                cfs = []
                for code in codes:
                    if not code:
                        cfs.append(None); continue
                    try:
                        r_cf, ev_cf = self.fork_pool.try_candidate(task_id, executed_actions, {"type": "execute_code", "code": code}, seed=seed)
                        cfs.append(Counterfactual(error=ErrorInfo(**r_cf.error) if r_cf.error else None, passes_before=passes or 0,
                                                  passes_after=len(ev_cf.passes), failures_after=len(ev_cf.failures), done=r_cf.done,
                                                  result_head=r_cf.observation[:200]))
                    except Exception as e:  # a fork failure must not kill the episode
                        cfs.append(None)
                for c, cf in zip(candidates, cfs):
                    c.counterfactual = cf
                context["counterfactuals"] = [None if cf is None else cf.model_dump() for cf in cfs]
            chosen, scores, meta = self.policy.choose(context, codes)
            for c, s in zip(candidates, scores):
                c.score = s
            code = codes[chosen]
            if not code:
                res_obs, err, done = "No code block found in your reply. Reply with exactly one ```python block.", {"type": "parse_error", "message": "no code block"}, False
            else:
                res = env.step({"type": "execute_code", "code": code})
                res_obs, err, done = res.observation, res.error, res.done
                candidates[chosen].action["api_calls"] = res.info.get("api_calls", [])
            if err:
                invalid += 1
                n_prev_errors += 1
            reward, ev_out = None, None
            if self.step_eval and code:
                try:
                    ev_s = env.evaluate()
                    new_passes = len(ev_s.passes)
                    reward = float(new_passes - passes) if passes is not None else None
                    ev_out = {"passes": new_passes, "failures": len(ev_s.failures), "success": ev_s.success}
                    passes = new_passes
                except Exception:
                    pass
            ep.steps.append(Step(step_index=i, timestamp=now(), observation=current_obs, action_schema_ref=schema_ref, candidates=candidates, chosen_index=chosen,
                                 tool_result=res_obs, error=ErrorInfo(**err) if err else None, success_state=done, reward=reward, evaluator_output=ev_out,
                                 usage=step_usage, policy_meta=meta))
            if code:
                executed_actions.append({"type": "execute_code", "code": code})
            turns.append((code or "(no code)", res_obs))
            history.append({"code": code or "", "error_type": err["type"] if err else None, "timestamp": time.time()})
            current_obs, last_error, last_code = res_obs, (err["message"] if err else None), code
            if done:
                termination = TerminationReason.completed
                break
            if self.cost_budget is not None and cost > self.cost_budget:
                termination = TerminationReason.budget
                break
        ev = env.evaluate()
        env.close()
        ep.final_eval = FinalEval(success=ev.success, num_tests=ev.num_tests, passes=ev.passes, failures=ev.failures, raw=ev.raw)
        ep.termination_reason = termination
        ep.ended_at = now()
        ep.totals = Totals(steps=len(ep.steps), invalid_actions=invalid, input_tokens=sum(s.usage.input_tokens for s in ep.steps),
                           output_tokens=sum(s.usage.output_tokens for s in ep.steps), cost_usd=cost, wall_s=time.time() - t_start)
        return ep
