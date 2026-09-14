#!/usr/bin/env bash
# Full-scale Part 1 runs, sequential (each run already uses all cores). Logs in results/logs/.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/logs
for c in domestic_random domestic_chrono international_random international_chrono incidents_random incidents_chrono closed_random open_random permit_random prepaid_random rfp_transfer_domestic; do
  echo "=== $c $(date)" | tee -a results/logs/run_all.log
  uv run python -m bpm.run configs/$c.yaml 2>&1 | tee results/logs/$c.log | grep -E "^\[|wrote|Error|Traceback" | tee -a results/logs/run_all.log
done
echo "=== ALL DONE $(date)" | tee -a results/logs/run_all.log
