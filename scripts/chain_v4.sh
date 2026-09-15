#!/usr/bin/env bash
cd "$(dirname "$0")/.."
while ! grep -q "V3 DONE" results/logs/part2_chain3.log; do sleep 30; done
scripts/run_part2_v4.sh
