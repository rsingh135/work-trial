#!/usr/bin/env bash
# Wait for the v2 evaluation script to exit, rerun its crashed oracle stage cleanly, then run v3.
cd "$(dirname "$0")/.."
while pgrep -f "run_part2_v2_eval.sh" >/dev/null; do sleep 20; done
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=2
rm -rf results/part2_appworld_v2_oracle_gate
uv run python -m agentloop.evaluate --env appworld --split dev --n-tasks 30 --runs 2 --model claude-haiku-4-5 --prompt-version v2 --workers 4 --port-base 9500 --skip-baseline \
    --policy oracle --gate --fork-workers 1 --fork-mode snapshot --n-candidates 3 --out results/part2_appworld_v2_oracle_gate
echo "V2 ORACLE DONE"
scripts/run_part2_v3.sh
