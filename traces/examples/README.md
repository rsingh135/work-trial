# Example traces

* `mock_episodes.jsonl` — two episodes from the offline mock environment (scripted LLM), including failed actions.
* `appworld_canned_episode.jsonl` — one AppWorld episode driven by canned actions (no LLM): shows the real action-schema hash, an `api_error` step, and redaction of `show_account_passwords()` output (`<REDACTED:…>`).
* `appworld_action_schema.json` — the action/tool schema referenced by `action_schema_ref` (457 API docs across AppWorld apps).

All files validate against `agentloop/schema.py` (`Episode.from_jsonl_line`). Real-LLM AppWorld examples are added under this folder by `scripts/run_part2_appworld.sh` (see PROGRESS.md for the run log).
* `appworld_haiku_episodes.jsonl` — three real episodes from the Claude Haiku 4.5 runs (redacted at write time): one successful training episode, one failed training episode with invalid actions, and one held-out dev episode under the token-scorer reranker (three candidates per step with scores and the chosen index).
