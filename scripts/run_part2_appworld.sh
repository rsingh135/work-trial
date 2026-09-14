#!/usr/bin/env bash
# Part 2 on AppWorld with a real LLM. Requires ANTHROPIC_API_KEY and `make appworld-setup` done.
#   N_TRAIN_TASKS, N_DEV_TASKS, RUNS, MODEL, BUDGET can be overridden via env vars.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}"
N_TRAIN_TASKS=${N_TRAIN_TASKS:-45}; N_DEV_TASKS=${N_DEV_TASKS:-30}; RUNS=${RUNS:-2}; MODEL=${MODEL:-claude-haiku-4-5}; BUDGET=${BUDGET:-10}
TAG=${TAG:-v1}
uv run python -m agentloop.collect --env appworld --split train --n-tasks $N_TRAIN_TASKS --runs 1 --model $MODEL \
    --cost-budget $BUDGET --out traces/appworld_train_$TAG
uv run python -m agentloop.build_dataset traces/appworld_train_$TAG --out datasets/appworld_$TAG
uv run python -m agentloop.train datasets/appworld_$TAG --out models/appworld_$TAG
uv run python -m agentloop.evaluate --env appworld --split dev --n-tasks $N_DEV_TASKS --runs $RUNS --model $MODEL \
    --policy-model models/appworld_$TAG/model.pkl --score-mode product --out results/part2_appworld_$TAG
echo "summary: results/part2_appworld_$TAG/summary.json"
# trace world model (Part 1 model on the same traces) alone and hybrid, same dev tasks
uv run python -m agentloop.trace_model train traces/appworld_train_$TAG --out models/tracemodel_$TAG
uv run python -m agentloop.evaluate --env appworld --split dev --n-tasks $N_DEV_TASKS --runs $RUNS --model $MODEL --skip-baseline \
    --policy tracemodel --policy-model models/tracemodel_$TAG --score-mode product --out results/part2_appworld_${TAG}_tracemodel
uv run python -m agentloop.evaluate --env appworld --split dev --n-tasks $N_DEV_TASKS --runs $RUNS --model $MODEL --skip-baseline \
    --policy tracemodel --policy-model models/tracemodel_$TAG --validity-model models/appworld_$TAG/model.pkl --score-mode product --out results/part2_appworld_${TAG}_hybrid
