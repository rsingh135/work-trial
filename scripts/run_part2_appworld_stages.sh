#!/usr/bin/env bash
# Remaining v1 stages (trace world model alone, hybrid) with a faulthandler stack dump every 15 min for hang diagnosis.
set -u
cd "$(dirname "$0")/.."
: "${ANTHROPIC_API_KEY:?}"; TAG=${TAG:-v1}; N_DEV_TASKS=${N_DEV_TASKS:-30}; RUNS=${RUNS:-2}; MODEL=${MODEL:-claude-haiku-4-5}
export PYTHONFAULTHANDLER=1
run_eval () {  # $1 = out dir, rest = evaluate args
  out=$1; shift
  uv run python -c "
import faulthandler, sys, io
faulthandler.dump_traceback_later(900, repeat=True, file=open('results/logs/stack_dump.log','a'))
sys.argv = ['evaluate'] + sys.argv[1:]
from agentloop.evaluate import main; main()
" "$@" --out "$out"
}
rm -rf results/part2_appworld_${TAG}_tracemodel results/part2_appworld_${TAG}_hybrid
run_eval results/part2_appworld_${TAG}_tracemodel --env appworld --split dev --n-tasks $N_DEV_TASKS --runs $RUNS --model $MODEL --skip-baseline \
    --policy tracemodel --policy-model models/tracemodel_$TAG --score-mode product
run_eval results/part2_appworld_${TAG}_hybrid --env appworld --split dev --n-tasks $N_DEV_TASKS --runs $RUNS --model $MODEL --skip-baseline \
    --policy tracemodel --policy-model models/tracemodel_$TAG --validity-model models/appworld_$TAG/model.pkl --score-mode product
echo "STAGES DONE"
