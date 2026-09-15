#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
: "${ANTHROPIC_API_KEY:?}"; TAG=${TAG:-v1}
rm -rf results/part2_appworld_${TAG}_hybrid
PYTHONFAULTHANDLER=1 uv run python -m agentloop.evaluate --env appworld --split dev --n-tasks 30 --runs 2 --model claude-haiku-4-5 --skip-baseline \
    --policy tracemodel --policy-model models/tracemodel_$TAG --validity-model models/appworld_$TAG/model.pkl --score-mode product \
    --out results/part2_appworld_${TAG}_hybrid
echo "HYBRID DONE"
