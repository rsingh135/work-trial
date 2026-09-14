.PHONY: setup appworld-setup test demo demo-part2 part1-all audit
setup:            ## main env (Python 3.12 via uv)
	uv sync --extra dev
appworld-setup:   ## AppWorld in its own venv (pydantic<2) + data download (~200 MB)
	uv venv .venv-appworld --python 3.11 && uv pip install --python .venv-appworld/bin/python appworld
	mkdir -p appworld_root && cd appworld_root && ../.venv-appworld/bin/appworld install && ../.venv-appworld/bin/appworld download data
test:
	uv run pytest -q
audit:            ## data audit over all 8 logs -> results/audit/
	uv run python -m bpm.ingest.audit
demo:             ## Part 1 one-command demo: sample, baselines + proposed model -> results/demo.json
	uv run python -m bpm.run configs/demo.yaml
demo-part2:       ## Part 2 offline demo (mock env, no API key)
	scripts/demo_part2_offline.sh
part1-all:        ## all full-scale Part 1 runs (hours)
	scripts/run_all_part1.sh
