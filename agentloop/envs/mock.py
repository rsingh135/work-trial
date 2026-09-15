"""Offline mock environment: a tiny deterministic "tool world" used by the tests and by the
no-API-key demo. It exercises every part of the Env protocol (errors, completion, evaluation)
so the collector → dataset → train → reinsert path can be verified without AppWorld or an LLM.

World: a key-value store with a hidden target. Actions are code strings executed against a
minimal ``apis`` object: ``apis.store.get(key)``, ``apis.store.keys()``,
``apis.supervisor.complete_task(answer=...)``. Anything else raises → execution error.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from agentloop.envs.base import EvalResult, Observation, StepResult

_TASKS = {f"mock{i:02d}_{v}": (f"scenario{i:02d}", f"What is the value stored under key k{i}?", f"k{i}", f"v{i}{v}")
          for i in range(12) for v in (1, 2, 3)}


class MockEnv:
    name = "mock"
    benchmark = "mockworld"

    def __init__(self, fail_rate_on_unknown: float = 1.0):
        self._task = None
        self._answer = None
        self._done = False
        self._steps = 0

    def version(self) -> str:
        return "mockworld=0.1"

    def task_ids(self, split: str) -> list[str]:
        ids = sorted(_TASKS)
        if split == "train":
            return [t for t in ids if int(t[4:6]) < 8]
        if split == "dev":
            return [t for t in ids if 8 <= int(t[4:6]) < 12]
        return []

    def scenario_id(self, task_id: str) -> str | None:
        return _TASKS[task_id][0]

    def reset(self, task_id: str, *, run_tag: str = "", seed: int | None = None) -> Observation:
        self._task = task_id
        self._answer, self._done, self._steps = None, False, 0
        _, instr, key, val = _TASKS[task_id]
        self._store = {key: val, "other": "noise"}
        return Observation(text=f"Task: {instr}", task_meta={"instruction": instr})

    def step(self, action: dict[str, Any]) -> StepResult:
        self._steps += 1
        code = action.get("code", "") if action.get("type") == "execute_code" else None
        if code is None:
            return StepResult("Invalid action.", {"type": "invalid_action", "message": "malformed"}, False)
        m = re.fullmatch(r"\s*print\(apis\.store\.keys\(\)\)\s*", code)
        if m:
            return StepResult(str(sorted(self._store)), None, False)
        m = re.fullmatch(r"\s*print\(apis\.store\.get\(['\"](\w+)['\"]\)\)\s*", code)
        if m:
            k = m.group(1)
            if k in self._store:
                return StepResult(self._store[k], None, False)
            return StepResult(f"Execution failed. Traceback:\nKeyError: {k!r}", {"type": "api_error", "message": f"KeyError: {k!r}"}, False)
        m = re.fullmatch(r"\s*apis\.supervisor\.complete_task\(answer=['\"]([^'\"]*)['\"]\)\s*", code)
        if m:
            self._answer, self._done = m.group(1), True
            return StepResult("Execution successful.", None, True)
        return StepResult("Execution failed. Traceback:\nNameError: unknown api", {"type": "runtime_error", "message": "NameError: unknown api"}, False)

    def try_candidates(self, actions):
        out = []
        for a in actions:
            if not a or not a.get("code"):
                out.append(None); continue
            snap = (dict(self._store), self._answer, self._done, self._steps)
            r = self.step(a); ev = self.evaluate()
            out.append((r, ev))
            self._store, self._answer, self._done, self._steps = dict(snap[0]), snap[1], snap[2], snap[3]
        return out

    def action_schema(self) -> dict[str, Any]:
        return {"action_language": "execute_code", "apps": {"store": ["get", "keys"], "supervisor": ["complete_task"]}}

    def evaluate(self) -> EvalResult:
        ok = self._answer == _TASKS[self._task][3]
        return EvalResult(success=ok, num_tests=1, passes=["answer"] if ok else [], failures=[] if ok else ["answer"], raw={"answer": self._answer})

    def close(self) -> None:
        self._task = None
