"""Forking by replay — test-time and label-time branching for environments that are deterministic
under replay (AppWorld is: same task + same code sequence → same outputs, verified).

A ``ForkPool`` owns K worker environments. ``fork(task_id, prefix_codes)`` resets a worker to the
task and re-executes the prefix, returning an env positioned exactly where the main world is; the
caller then executes one candidate on it and reads the evaluator. This is the harness analogue of
snapshot-forking VMs: it lets us (a) label *every* sampled candidate counterfactually, and (b) run
a lookahead policy that picks the candidate whose fork actually makes progress.

Generic: works for any Env whose ``step`` is deterministic given the action sequence.
"""

from __future__ import annotations

from typing import Any, Callable

from agentloop.envs.base import Env, EvalResult, StepResult


class ForkPool:
    def __init__(self, make_env: Callable[[int], Env], n_workers: int = 1):
        self.workers: list[Env] = [make_env(i) for i in range(n_workers)]
        self._next = 0

    def fork(self, task_id: str, prefix_actions: list[dict[str, Any]], run_tag: str = "fork", seed: int | None = None) -> Env:
        w = self.workers[self._next % len(self.workers)]
        self._next += 1
        w.close()
        w.reset(task_id, run_tag=run_tag, seed=seed)
        for a in prefix_actions:
            w.step(a)
        return w

    def try_candidate(self, task_id: str, prefix_actions: list[dict[str, Any]], action: dict[str, Any], seed: int | None = None) -> tuple[StepResult, EvalResult]:
        """Execute ``action`` after ``prefix_actions`` in a fork; return the step result and the evaluator state."""
        w = self.fork(task_id, prefix_actions, seed=seed)
        r = w.step(action)
        ev = w.evaluate()
        return r, ev

    def shutdown(self):
        for w in self.workers:
            w.close()
            if hasattr(w, "shutdown"):
                w.shutdown()
