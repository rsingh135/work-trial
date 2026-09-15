"""Part 2 tests: trace schema contracts, redaction, and the critical collect → build → train →
reinsert path on the offline mock environment (no API key, no AppWorld needed)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentloop.agent.llm_agent import LLMAgent, ScriptedClient, parse_code
from agentloop.agent.policy import FirstCandidatePolicy, RerankerPolicy
from agentloop.build_dataset import build, episode_examples
from agentloop.envs.mock import MockEnv
from agentloop.redaction import Redactor
from agentloop.schema import Candidate, Episode, ErrorInfo, FinalEval, Step, TerminationReason, Totals, Versions, now
from agentloop.train import train


def _versions():
    return Versions(env_version="mock=0", benchmark_version="mock=0", model_id="scripted", prompt_hash="abc", harness_git_sha="000", policy_id="first_candidate")


def _step(i, code="print(1)", error=None, chosen=0, n_cand=1):
    cands = [Candidate(action_raw=f"```python\n{code}\n```", action={"type": "execute_code", "code": code}) for _ in range(n_cand)]
    return Step(step_index=i, timestamp=now(), observation="obs", action_schema_ref="h", candidates=cands, chosen_index=chosen, tool_result="1", error=error)


def test_schema_roundtrip_and_required_fields():
    ep = Episode(env_id="mock", benchmark="mockworld", task_id="t1", episode_id="e1", run_id="r1", split="train", versions=_versions(), started_at=now(),
                 task_instruction="do x", steps=[_step(0), _step(1, error=ErrorInfo(type="api_error", message="boom"))],
                 termination_reason=TerminationReason.max_steps, final_eval=FinalEval(success=False), totals=Totals(steps=2, invalid_actions=1, input_tokens=0, output_tokens=0, cost_usd=0, wall_s=0))
    line = ep.to_jsonl_line()
    back = Episode.from_jsonl_line(line)
    assert back == ep and back.schema_version == "1.1.0"
    assert back.steps[1].error.type == "api_error"  # failed actions are retained


def test_schema_rejects_bad_steps():
    with pytest.raises(ValidationError):
        Episode(env_id="m", benchmark="m", task_id="t", episode_id="e", run_id="r", split="train", versions=_versions(), started_at=now(), task_instruction="x",
                steps=[_step(0), _step(2)])  # non-contiguous
    with pytest.raises(ValidationError):
        _step(0, chosen=3, n_cand=2)  # chosen_index out of range
    with pytest.raises(ValidationError):
        Episode(env_id="m", benchmark="m", task_id="t", episode_id="e", run_id="r", split="train", versions=_versions(), started_at=now(), task_instruction="x", schema_version="0.9")


def test_redaction_rules():
    r = Redactor(salt="s")
    s = 'print(apis.spotify.login(username="a@b.com", password="Sup3r!"))\n[{"account_name": "gmail", "password": "r^2p&]H"}] token = "eyJabcdefghijk.abcdefghijklmn.abcdefghijklmn"'
    out = r.text(s)
    assert "Sup3r!" not in out and "r^2p&]H" not in out and "eyJabcdefghijk" not in out
    assert out.count("<REDACTED:") == 3 and r.count == 3
    assert "a@b.com" in out  # non-secret values untouched
    d = r.obj({"steps": [{"tool_result": "password: 'x'", "meta": {"api_key": "K123"}}]})
    assert d["steps"][0]["meta"]["api_key"].startswith("<REDACTED:")


def test_parse_code():
    assert parse_code("text\n```python\nprint(1)\n```")[0] == "print(1)"
    assert parse_code("no code here") == (None, "no_code_block")
    assert parse_code("<function_calls><invoke name=\"x\"><parameter>apis.spotify.login()</parameter>")[0] is None
    assert parse_code("```python\nprint(apis.x.y(")[1] == "truncated_code_block"


def test_end_to_end_collect_build_train_reinsert(tmp_path: Path):
    env = MockEnv()
    schemas = {}
    # 1) collect with the baseline policy (noisy scripted LLM → some invalid actions)
    agent = LLMAgent(ScriptedClient(noise=0.4, seed=1), FirstCandidatePolicy(), max_steps=8)
    eps = [agent.run_episode(env, t, "train", run_id="r0", seed=1, schema_store=schemas) for t in env.task_ids("train")]
    assert all(e.totals.steps >= 1 for e in eps) and any(e.totals.invalid_actions > 0 for e in eps)
    d = tmp_path / "traces"
    d.mkdir()
    (d / "episodes.jsonl").write_text("".join(e.to_jsonl_line() + "\n" for e in eps))
    # 2) build examples with provenance
    exs = episode_examples(eps[0], "f")
    assert len(exs) == eps[0].totals.steps and exs[0].provenance.episode_id == eps[0].episode_id
    stats = build([d], tmp_path / "ds", heldout_frac=0.25, seed=0)
    assert stats["n_examples"] == sum(e.totals.steps for e in eps) and stats["n_heldout"] > 0
    held = set(stats["heldout_groups"])
    for line in (tmp_path / "ds" / "train.jsonl").read_text().splitlines():
        assert json.loads(line)["group_id"] not in held  # group hygiene
    # 3) train
    rep = train(tmp_path / "ds", tmp_path / "model")
    assert rep["heldout_metrics"]["valid"]["auroc"] > 0.8
    # 4) reinsert: the policy must prefer a valid candidate over an invalid one, and change behaviour
    pol = RerankerPolicy(tmp_path / "model" / "model.pkl", n_candidates=3, score_mode="valid")
    ctx = {"instruction": "What is the value stored under key k3?", "last_observation": "['k3', 'other']", "last_error": None, "last_code": "print(apis.store.keys())", "step_index": 1, "n_prev_errors": 0}
    idx, scores, meta = pol.choose(ctx, ["print(apis.storage.get('a'))", "print(apis.store.get('k3'))"])
    assert idx == 1 and scores[1] > scores[0] and meta["overrode_first"]
    agent2 = LLMAgent(ScriptedClient(noise=0.4, seed=1), pol, max_steps=8)
    ep2 = agent2.run_episode(env, env.task_ids("dev")[0], "dev", run_id="r1", seed=1)
    assert ep2.versions.policy_id == "validity_reranker" and len(ep2.steps[0].candidates) == 3
    assert all(c.score is not None for c in ep2.steps[0].candidates)


def test_trace_world_model_train_and_policy(tmp_path: Path):
    """Agent traces as an event log: episode→case conversion, Part 1 model trained on them, and the
    resulting policy scoring candidates (incl. the hybrid with the token-level validity scorer)."""
    from agentloop.trace_model import OUTCOME_FAIL, OUTCOME_OK, TraceModelPolicy, action_activity, episode_to_case, train as train_tm
    env = MockEnv()
    agent = LLMAgent(ScriptedClient(noise=0.4, seed=2), FirstCandidatePolicy(), max_steps=8)
    eps = [agent.run_episode(env, t, "train", run_id="r0", seed=2) for t in env.task_ids("train")]
    c = episode_to_case(eps[0])
    assert len(c) == eps[0].totals.steps + 2 and c.activities[-1] in (OUTCOME_OK, OUTCOME_FAIL) and c.complete
    assert action_activity("print(apis.store.get('k1'))", None) == "store.get|ok" and action_activity("x", "api_error") == "python.no_api|err"
    d = tmp_path / "tr"; d.mkdir()
    (d / "episodes.jsonl").write_text("".join(e.to_jsonl_line() + "\n" for e in eps))
    rep = train_tm([d], tmp_path / "tm", epochs=15)
    assert rep["vocab"] > 3 and "heldout" in rep
    pol = TraceModelPolicy(tmp_path / "tm", n_candidates=2, score_mode="plausible")
    ctx = {"instruction": "What is the value stored under key k3?", "history": [{"code": "print(apis.store.keys())", "error_type": None, "timestamp": 1.0}], "benchmark": "mockworld"}
    idx, scores, meta = pol.choose(ctx, ["print(apis.nonexistent.call())", "print(apis.store.get('k3'))"])
    assert idx == 1 and "plausible" in meta


def test_schema_and_progress_gates():
    from agentloop.agent.policy import GatedPolicy, premature_completion, schema_violations
    schema = {"apps": {"store": ["get", "keys"], "supervisor": ["complete_task"], "api_docs": ["show_api_doc"]}}
    assert schema_violations("print(apis.store.get('k'))", schema) == []
    assert schema_violations("print(apis.store.list_all())", schema) == ["store.list_all"]
    assert schema_violations("x = apis.storage.get('k')", schema) == ["storage.get"]
    assert premature_completion("apis.supervisor.complete_task()", []) is True
    assert premature_completion("apis.supervisor.complete_task()", [{"code": "print(apis.api_docs.show_api_doc())", "error_type": None}]) is True
    assert premature_completion("apis.supervisor.complete_task(answer=v)", [{"code": "v = apis.store.get('k')", "error_type": None}]) is False
    pol = GatedPolicy(FirstCandidatePolicy())
    ctx = {"action_schema": schema, "history": []}
    idx, scores, meta = pol.choose(ctx, ["apis.supervisor.complete_task()", "print(apis.store.list_all())", "print(apis.store.keys())"])
    assert idx == 2 and meta["gate_rejected"] == [0, 1] and pol.n_candidates == 3
    idx, _, meta = pol.choose(ctx, ["print(apis.store.list_all())"])  # nothing survives → fallback to inner choice
    assert idx == 0 and meta["gate_fallback"]


def test_forking_counterfactuals_and_oracle(tmp_path: Path):
    """Forked execution labels every candidate; the dataset builder emits counterfactual examples with
    progress/regress labels; the oracle policy picks the candidate whose fork made progress."""
    from agentloop.agent.policy import OracleLookaheadPolicy
    from agentloop.envs.forking import ForkPool
    env = MockEnv()
    pool = ForkPool(lambda i: MockEnv(), n_workers=1)
    agent = LLMAgent(ScriptedClient(noise=0.4, seed=3), FirstCandidatePolicy(n_candidates=3), max_steps=6, fork_pool=pool)
    ep = agent.run_episode(env, env.task_ids("train")[0], "train", run_id="r0", seed=3)
    s0 = ep.steps[0]
    assert len(s0.candidates) == 3 and all(c.counterfactual is not None for c in s0.candidates)
    assert s0.reward is not None and s0.evaluator_output is not None
    exs = episode_examples(ep, "f")
    assert sum(e.source == "counterfactual" for e in exs) == 2 * len(ep.steps)
    assert all(e.label_progress is not None for e in exs)
    # oracle: a candidate whose fork reaches the passing state must be preferred over an erroring one
    pol = OracleLookaheadPolicy()
    cfs = [{"error": {"type": "api_error", "message": "x"}, "passes_before": 0, "passes_after": 0, "failures_after": 1, "done": False},
           {"error": None, "passes_before": 0, "passes_after": 1, "failures_after": 0, "done": True},
           {"error": None, "passes_before": 0, "passes_after": 0, "failures_after": 1, "done": True}]
    idx, scores, meta = pol.choose({"counterfactuals": cfs}, ["a", "b", "c"])
    assert idx == 1 and scores[2] < scores[0]  # premature 'done' ranks below a plain error
