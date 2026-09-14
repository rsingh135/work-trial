#!/usr/bin/env bash
# Published-protocol reproduction: 5-fold CV on BPI2013 Closed Problems + Incidents (10 runs).
set -u
cd "$(dirname "$0")/.."
mkdir -p results/logs
for tag in closed incidents; do for k in 0 1 2 3 4; do
  c=published_${tag}_fold$k
  echo "=== $c $(date)" | tee -a results/logs/run_published.log
  uv run python -m bpm.run configs/$c.yaml 2>&1 | tee results/logs/$c.log | grep -E "^\[|wrote|Error|Traceback" | tee -a results/logs/run_published.log
done; done
echo "=== ALL DONE $(date)" | tee -a results/logs/run_published.log
