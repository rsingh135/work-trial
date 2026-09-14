"""Versioned trace schema (pydantic v2). One ``Episode`` per (task, run); one ``Step`` per
agent action, including failed ones. This is the contract between collector, dataset builder,
trainer and evaluator — every benchmark adapter must produce it.

Design choices
--------------
* ``action_schema_ref`` — tool/API schemas are large and identical across steps; each step
  carries a content hash, the full schema is stored once per collection run in
  ``schemas/<hash>.json`` next to the episodes.
* ``candidates`` — when the policy samples several actions, all are recorded with the
  policy's score; ``chosen_index`` says which one was executed. Only the executed action
  has ground-truth ``error``/``tool_result``; the others are kept for provenance and
  offline analysis (not silently dropped).
* ``provenance`` on training examples points back to (episode_id, step_index,
  candidate_index, schema_version, builder_version) so any example can be reconstructed.
* Redaction is applied at serialisation (``Episode.redacted()``); ``redaction`` records the
  rule version and count so a reader knows what was masked.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"


class TerminationReason(str, Enum):
    completed = "completed"  # agent declared completion
    max_steps = "max_steps"
    error = "error"  # harness / LLM / env error
    timeout = "timeout"
    budget = "budget"  # token/cost budget exhausted


class ErrorInfo(BaseModel):
    type: str  # e.g. "execution_error", "syntax_error", "llm_error", "env_error", "parse_error"
    message: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


class Candidate(BaseModel):
    action_raw: str  # full LLM output
    action: dict[str, Any]  # structured: {"type": "execute_code", "code": "...", "api_calls": [...]}
    score: float | None = None  # policy score (None when not scored)
    usage: Usage = Field(default_factory=Usage)
    parse_error: str | None = None


class Step(BaseModel):
    step_index: int
    timestamp: datetime
    observation: str  # what the agent saw before acting (env output of the previous step, or task prompt)
    action_schema_ref: str  # sha256 of the action/tool schema visible to the agent
    candidates: list[Candidate]
    chosen_index: int
    tool_result: str  # env output for the executed action (raw, may be truncated with marker)
    error: ErrorInfo | None = None
    retry_index: int = 0  # >0 when this step re-attempts after a harness-level failure (e.g. LLM error)
    reward: float | None = None
    success_state: bool | None = None  # env-reported "task completed" flag after this step
    evaluator_output: dict[str, Any] | None = None  # per-step evaluator info if the env provides it
    usage: Usage = Field(default_factory=Usage)  # totals over candidates for this step
    policy_meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def action(self) -> Candidate:
        return self.candidates[self.chosen_index]

    @model_validator(mode="after")
    def _check_chosen(self):
        if not (0 <= self.chosen_index < len(self.candidates)):
            raise ValueError("chosen_index out of range")
        return self


class Versions(BaseModel):
    env_version: str
    benchmark_version: str
    model_id: str
    prompt_hash: str
    harness_git_sha: str
    policy_id: str
    policy_version: str = "none"
    seed: int | None = None


class FinalEval(BaseModel):
    success: bool | None = None
    num_tests: int | None = None
    passes: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    raw: dict[str, Any] | None = None


class Totals(BaseModel):
    steps: int
    invalid_actions: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    wall_s: float


class Redaction(BaseModel):
    rules_version: str
    n_redactions: int


class Episode(BaseModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    env_id: str  # adapter name, e.g. "appworld"
    benchmark: str  # e.g. "appworld"
    task_id: str
    scenario_id: str | None = None  # benchmark-level grouping (AppWorld: task family)
    episode_id: str
    run_id: str
    split: str  # train / dev / test / ...
    versions: Versions
    started_at: datetime
    ended_at: datetime | None = None
    task_instruction: str
    task_meta: dict[str, Any] = Field(default_factory=dict)
    steps: list[Step] = Field(default_factory=list)
    termination_reason: TerminationReason | None = None
    final_eval: FinalEval | None = None
    totals: Totals | None = None
    redaction: Redaction | None = None

    @field_validator("steps")
    @classmethod
    def _contiguous(cls, v):
        for i, s in enumerate(v):
            if s.step_index != i:
                raise ValueError(f"step_index {s.step_index} at position {i}")
        return v

    def to_jsonl_line(self) -> str:
        return self.model_dump_json()

    @staticmethod
    def from_jsonl_line(line: str) -> "Episode":
        return Episode.model_validate_json(line)


class Provenance(BaseModel):
    episode_id: str
    step_index: int
    candidate_index: int
    schema_version: str
    builder_version: str
    trace_file: str


class TrainingExample(BaseModel):
    """Prefix → target example for the learned component (action-validity scorer)."""

    example_id: str
    group_id: str  # for grouped splits (scenario_id or task_id)
    split: str
    context: dict[str, str]  # {"instruction", "last_observation", "last_error", "history_summary"}
    action: dict[str, Any]
    label_valid: int  # 1 = executed without error
    label_success: int | None  # episode-level success (None if unknown)
    step_index: int
    provenance: Provenance


def content_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def now() -> datetime:
    return datetime.now(timezone.utc)
