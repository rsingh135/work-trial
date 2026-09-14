#!/usr/bin/env bash
# Part 2 end-to-end demo WITHOUT any API key or AppWorld: mock env + scripted LLM.
# collect -> build examples -> train scorer -> reinsert (reranker) -> compare vs baseline on held-out dev tasks.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf traces/mock_train datasets/mock models/mock results/part2_mock
uv run python -m agentloop.collect --env mock --split train --client scripted --runs 2 --noise 0.35 --out traces/mock_train
uv run python -m agentloop.build_dataset traces/mock_train --out datasets/mock
uv run python -m agentloop.train datasets/mock --out models/mock
uv run python -m agentloop.evaluate --env mock --split dev --client scripted --runs 3 --noise 0.35 \
    --policy-model models/mock/model.pkl --score-mode product --out results/part2_mock
echo "summary: results/part2_mock/summary.json"
# --- the connection between the parts: Part 1 sequence model trained on the same agent traces, reinserted as policy
uv run python -m agentloop.trace_model train traces/mock_train --out models/mock_tracemodel --epochs 40
uv run python -m agentloop.evaluate --env mock --split dev --client scripted --runs 3 --noise 0.35 \
    --policy tracemodel --policy-model models/mock_tracemodel --score-mode product --skip-baseline --out results/part2_mock_tracemodel
uv run python -m agentloop.evaluate --env mock --split dev --client scripted --runs 3 --noise 0.35 \
    --policy tracemodel --policy-model models/mock_tracemodel --validity-model models/mock/model.pkl --score-mode product --skip-baseline --out results/part2_mock_hybrid
echo "summaries: results/part2_mock{,_tracemodel,_hybrid}/summary.json"
