#!/usr/bin/env bash
# v3 = v2 loop + harness fixes (prompt v3: programmatic password lookup, short turns; 4096-token cap; strict parsing).
set -u
cd "$(dirname "$0")/.."
: "${ANTHROPIC_API_KEY:?}"; TAG=v3; N_DEV=${N_DEV_TASKS:-30}; RUNS=${RUNS:-2}; MODEL=${MODEL:-claude-haiku-4-5}; W=${WORKERS:-4}
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=2
rm -rf traces/appworld_train_$TAG
uv run python -m agentloop.collect --env appworld --split train --runs 1 --model $MODEL --prompt-version v3 --max-tokens 4096 \
    --n-candidates 3 --fork-workers 1 --workers $W --max-steps 25 --cost-budget 30 --out traces/appworld_train_$TAG
uv run python -m agentloop.build_dataset traces/appworld_train_$TAG --out datasets/appworld_$TAG
uv run python -m agentloop.train datasets/appworld_$TAG --out models/appworld_$TAG
uv run python -m agentloop.trace_model train traces/appworld_train_$TAG --out models/tracemodel_$TAG
COMMON="--env appworld --split dev --n-tasks $N_DEV --runs $RUNS --model $MODEL --prompt-version v3 --max-tokens 4096 --workers $W --skip-baseline"
uv run python -m agentloop.evaluate $COMMON --policy baseline --n-candidates 1 --out results/part2_appworld_${TAG}_baseline
uv run python -m agentloop.evaluate $COMMON --policy baseline --gate --n-candidates 3 --out results/part2_appworld_${TAG}_gate
uv run python -m agentloop.evaluate $COMMON --policy reranker --policy-model models/appworld_$TAG/model.pkl --score-mode progress --gate --n-candidates 3 --out results/part2_appworld_${TAG}_token_progress_gate
uv run python -m agentloop.evaluate $COMMON --policy oracle --gate --fork-workers 1 --n-candidates 3 --out results/part2_appworld_${TAG}_oracle_gate
uv run python -m agentloop.evaluate $COMMON --policy tracemodel --policy-model models/tracemodel_$TAG --validity-model models/appworld_$TAG/model.pkl --score-mode product --gate --n-candidates 3 --out results/part2_appworld_${TAG}_hybrid_gate
echo "V3 DONE"
