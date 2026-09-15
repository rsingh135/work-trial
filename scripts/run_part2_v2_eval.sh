#!/usr/bin/env bash
# v2: train both learners on the v2 traces (counterfactual + progress labels), then evaluate five policies on the
# same 30 dev tasks × 2 runs, prompt v2, 4 parallel workers. Controls: gate-only (no learning), oracle lookahead (upper bound).
set -u
cd "$(dirname "$0")/.."
: "${ANTHROPIC_API_KEY:?}"; TAG=${TAG:-v2}; N_DEV=${N_DEV_TASKS:-30}; RUNS=${RUNS:-2}; MODEL=${MODEL:-claude-haiku-4-5}; W=${WORKERS:-4}
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=2
uv run python -m agentloop.build_dataset traces/appworld_train_$TAG --out datasets/appworld_$TAG
uv run python -m agentloop.train datasets/appworld_$TAG --out models/appworld_$TAG
uv run python -m agentloop.trace_model train traces/appworld_train_$TAG --out models/tracemodel_$TAG
COMMON="--env appworld --split dev --n-tasks $N_DEV --runs $RUNS --model $MODEL --prompt-version v2 --workers $W --skip-baseline"
# 1. baseline (v2 prompt, 1 sample)   — run via --policy baseline without gate
uv run python -m agentloop.evaluate $COMMON --policy baseline --n-candidates 1 --out results/part2_appworld_${TAG}_baseline
# 2. gate only (rule-based control, 3 samples)
uv run python -m agentloop.evaluate $COMMON --policy baseline --gate --n-candidates 3 --out results/part2_appworld_${TAG}_gate
# 3. token scorer, progress mode + gate
uv run python -m agentloop.evaluate $COMMON --policy reranker --policy-model models/appworld_$TAG/model.pkl --score-mode progress --gate --n-candidates 3 --out results/part2_appworld_${TAG}_token_progress_gate
# 4. oracle lookahead + gate (forks every candidate; upper bound for a learned scorer)
uv run python -m agentloop.evaluate $COMMON --policy oracle --gate --fork-workers 1 --n-candidates 3 --out results/part2_appworld_${TAG}_oracle_gate
# 5. hybrid (trace world model × token scorer) + gate
uv run python -m agentloop.evaluate $COMMON --policy tracemodel --policy-model models/tracemodel_$TAG --validity-model models/appworld_$TAG/model.pkl --score-mode product --gate --n-candidates 3 --out results/part2_appworld_${TAG}_hybrid_gate
echo "V2 EVAL DONE"
