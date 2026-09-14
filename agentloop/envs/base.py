"""Benchmark-agnostic environment protocol.

An adapter turns a benchmark into: task ids per split, reset → observation, step(action) →
result, a description of the action space, and a final evaluation. Nothing else in the loop
(collector, dataset builder, trainer, evaluator, policies) imports a benchmark package.

Actions are plain dicts with a ``type`` key so different benchmarks can use different action
languages (AppWorld: ``{"type": "execute_code", "code": str}``; a JSON-tool-call benchmark would
use ``{"type": "tool_call", "name": ..., "arguments": {...}}``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Observation:
    text: str
    task_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    observation: str  # tool result / env output shown to the agent next
    error: dict[str, str] | None  # {"type", "message"} if the action failed, else None
    done: bool  # env says the episode is over (task declared complete, etc.)
    reward: float | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    success: bool | None
    num_tests: int | None = None
    passes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    raw: dict[str, Any] | None = None


@runtime_checkable
class Env(Protocol):
    name: str
    benchmark: str

    def version(self) -> str: ...
    def task_ids(self, split: str) -> list[str]: ...
    def scenario_id(self, task_id: str) -> str | None: ...
    def reset(self, task_id: str, *, run_tag: str = "", seed: int | None = None) -> Observation: ...
    def step(self, action: dict[str, Any]) -> StepResult: ...
    def action_schema(self) -> dict[str, Any]:
        """Full description of the action space / tools visible to the agent (hashed into traces)."""
        ...
    def evaluate(self) -> EvalResult: ...
    def close(self) -> None: ...
