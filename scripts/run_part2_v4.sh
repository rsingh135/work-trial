#!/usr/bin/env bash
# v4 = Claude Opus 5 as the agent. Arms on the same 30 dev tasks × 2 runs:
#   opus_raw      : prompt v3, raw-history memory (the "pixel history" baseline)
#   opus_wf       : prompt v4, persistent world-frame memory (PERSIST-style state) + seeded init
#   opus_wf_gate_token : world-frame + gates + token scorer (progress mode) trained on Opus world-frame traces
#   opus_wf_oracle: world-frame + gates + oracle lookahead (ceiling)
set -u
cd "$(dirname "$0")/.."
: "${ANTHROPIC_API_KEY:?}"; TAG=v4; N_DEV=${N_DEV_TASKS:-30}; RUNS=${RUNS:-2}; MODEL=${MODEL:-claude-opus-5}; W=${WORKERS:-4}; PB=9800
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=2
rm -rf traces/appworld_train_$TAG
uv run python -m agentloop.collect --env appworld --split train --runs 1 --model $MODEL --prompt-version v4 --memory worldframe --max-tokens 4096 \
    --n-candidates 3 --fork-workers 1 --fork-mode snapshot --workers $W --port-base $PB --max-steps 25 --cost-budget 250 --out traces/appworld_train_$TAG
uv run python -m agentloop.build_dataset traces/appworld_train_$TAG --out datasets/appworld_$TAG
uv run python -m agentloop.train datasets/appworld_$TAG --out models/appworld_$TAG
uv run python -m agentloop.trace_model train traces/appworld_train_$TAG --out models/tracemodel_$TAG
COMMON="--env appworld --split dev --n-tasks $N_DEV --runs $RUNS --model $MODEL --max-tokens 4096 --workers $W --port-base $PB --fork-mode snapshot --skip-baseline"
uv run python -m agentloop.evaluate $COMMON --prompt-version v3 --memory raw        --policy baseline --n-candidates 1 --out results/part2_appworld_${TAG}_opus_raw
uv run python -m agentloop.evaluate $COMMON --prompt-version v4 --memory worldframe --policy baseline --n-candidates 1 --out results/part2_appworld_${TAG}_opus_wf
uv run python -m agentloop.evaluate $COMMON --prompt-version v4 --memory worldframe --policy reranker --policy-model models/appworld_$TAG/model.pkl --score-mode progress --gate --n-candidates 3 --out results/part2_appworld_${TAG}_opus_wf_gate_token
uv run python -m agentloop.evaluate $COMMON --prompt-version v4 --memory worldframe --policy oracle --gate --fork-workers 1 --n-candidates 3 --out results/part2_appworld_${TAG}_opus_wf_oracle
echo "V4 DONE"
