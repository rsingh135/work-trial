#!/usr/bin/env bash
# Post-hoc analyses on finished runs: failure analysis (GRU + Transformer) and downstream uses of the state.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
for r in "$@"; do
  for m in gru_multihead transformer_multihead; do
    [ -f results/$r/${m}_seed0.pt ] || { echo "skip $r $m"; continue; }
    uv run python -m bpm.analyze_failures results/$r.json --model $m --n-prefixes 500 2>&1 | tail -1
    uv run python -m bpm.downstream results/$r.json --model $m 2>&1 | tail -1
  done
done
echo "ANALYSES DONE"
