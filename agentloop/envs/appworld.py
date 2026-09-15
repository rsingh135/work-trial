"""AppWorld adapter — talks to ``appworld serve environment`` over HTTP.

Why out-of-process: the ``appworld`` package pins pydantic<2 while the harness uses pydantic
v2, and an HTTP boundary is also what keeps the loop benchmark-agnostic (the collector never
imports AppWorld). The adapter optionally launches the server itself from the dedicated venv.

Action language: ``{"type": "execute_code", "code": "<python>"}`` — AppWorld's native
interaction is executing Python against ``apis.<app>.<endpoint>(...)``.
Errors: AppWorld returns a string starting with ``Execution failed.`` for any exception;
we classify it into syntax / api / runtime error types for the trace.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from agentloop.envs.base import EvalResult, Observation, StepResult

_API_CALL = re.compile(r"apis\.([a-z_]+)\.([a-z_]+)\(")


def extract_api_calls(code: str) -> list[str]:
    return [f"{a}.{b}" for a, b in _API_CALL.findall(code)]


def classify_error(output: str) -> dict[str, str] | None:
    if not output.startswith("Execution failed"):
        return None
    if "Syntax error" in output:
        t = "syntax_error"
    elif "Response status code is" in output:
        t = "api_error"
    elif "Maximum number of executions" in output:
        t = "budget_error"
    else:
        t = "runtime_error"
    # last non-empty line is the most informative message
    lines = [l for l in output.strip().splitlines() if l.strip()]
    return {"type": t, "message": lines[-1][:500] if lines else output[:500]}


class AppWorldEnv:
    name = "appworld"
    benchmark = "appworld"

    def __init__(self, root: str = "appworld_root", url: str | None = None, port: int = 9123, venv_python: str = ".venv-appworld/bin/python",
                 max_interactions: int = 100_000, autostart: bool = True, max_output_chars: int = 3000):
        # max_interactions is AppWorld's per-world execute budget; the harness enforces its own step limit and the
        # snapshot trials spend several executes per candidate, so the env-side cap must not bind.
        self.root = Path(root)
        self.port = port
        self.url = url or f"http://127.0.0.1:{port}"
        self.venv_python = venv_python
        self.max_interactions = max_interactions
        self.max_output_chars = max_output_chars
        self._proc: subprocess.Popen | None = None
        self._task_id: str | None = None
        self._schema_cache: dict | None = None
        self._dirty = False  # set after a snapshot rollback: AppWorld's load_state pops its time-freezer stack, so the
        # server must be restarted before the next /initialize (otherwise 500 "pop from empty list")
        if autostart and not self._alive():
            self._start_server()

    # ------------------------------------------------------------------ server mgmt
    def _alive(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def _start_server(self):
        cmd = [self.venv_python, "-m", "appworld.cli", "serve", "environment", "--port", str(self.port), "--root", str(self.root), "--no-show-usage"]
        log = open(self.root / "server.log", "ab")
        self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(self.root.parent) if self.root.is_absolute() else None)
        for _ in range(60):
            if self._alive():
                return
            if self._proc.poll() is not None:
                raise RuntimeError(f"appworld server exited early; see {self.root / 'server.log'}")
            time.sleep(1)
        raise RuntimeError("appworld server did not come up in 60s")

    def _restart_server(self):
        """Kill and relaunch this adapter's server (owned or not) on the same port."""
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        else:  # server we did not start (leftover): find it by port and terminate it
            try:
                pids = subprocess.run(["lsof", "-t", f"-iTCP:{self.port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()
                for pid in pids:
                    subprocess.run(["kill", pid])
            except Exception:
                pass
        for _ in range(30):
            if not self._alive():
                break
            time.sleep(0.5)
        self._task_id = None
        self._start_server()

    def _post(self, route: str, payload: dict, timeout: float = 300.0) -> Any:
        req = urllib.request.Request(self.url + route, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["output"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"appworld {route} failed: {e.code} {e.read()[:500]!r}") from e

    def _get(self, route: str) -> Any:
        with urllib.request.urlopen(self.url + route, timeout=60) as r:
            return json.loads(r.read())

    # ------------------------------------------------------------------ Env protocol
    def version(self) -> str:
        try:
            v = (self.root / "data" / "version.txt").read_text().strip()
        except OSError:
            v = "unknown"
        pkg = subprocess.run([self.venv_python, "-c", "import appworld,sys;print(getattr(appworld,'__version__','?'))"], capture_output=True, text=True).stdout.strip()
        return f"appworld={pkg};data={v}"

    def task_ids(self, split: str) -> list[str]:
        p = self.root / "data" / "datasets" / f"{split}.txt"
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]

    def scenario_id(self, task_id: str) -> str | None:
        return task_id.split("_")[0]

    def reset(self, task_id: str, *, run_tag: str = "", seed: int | None = None) -> Observation:
        if self._dirty:
            self._restart_server()
            self._dirty = False
        out = self._post("/initialize", {"task_id": task_id, "experiment_name": f"harness_{run_tag or 'run'}", "max_interactions": self.max_interactions,
                                          "random_seed": seed if seed is not None else 100, "raise_on_failure": True})
        self._task_id = task_id
        sup = out.get("supervisor", {})
        text = (f"Task: {out['instruction']}\n"
                f"You are acting on behalf of the supervisor {sup.get('first_name','')} {sup.get('last_name','')} "
                f"(email {sup.get('email','')}, phone {sup.get('phone_number','')}). Current datetime: {out.get('datetime','')}.")
        return Observation(text=text, task_meta={"instruction": out["instruction"], "supervisor": sup, "datetime": out.get("datetime")})

    def step(self, action: dict[str, Any]) -> StepResult:
        assert self._task_id, "reset() first"
        if action.get("type") != "execute_code" or not isinstance(action.get("code"), str):
            return StepResult(observation="Invalid action: expected {'type': 'execute_code', 'code': str}.", error={"type": "invalid_action", "message": "malformed action"}, done=False)
        out: str = self._post("/execute", {"task_id": self._task_id, "code": action["code"]})
        err = classify_error(out)
        done = bool(self._post("/task_completed", {"task_id": self._task_id}))
        obs = out if len(out) <= self.max_output_chars else out[: self.max_output_chars] + f"\n...[truncated {len(out) - self.max_output_chars} chars]"
        return StepResult(observation=obs, error=err, done=done, info={"raw_len": len(out), "api_calls": extract_api_calls(action["code"])})

    # -------------------------------------------------------------- snapshot forking (O(1) per candidate)
    _NS_SNAP = "__ns_snap = {k: v for k, v in globals().items() if not k.startswith('__')}"
    _NS_RESTORE = ("for __k in [k for k in globals() if not k.startswith('__') and k not in __ns_snap]: del globals()[__k]\n"
                   "globals().update(__ns_snap)")

    def try_candidates(self, actions: list[dict[str, Any]]) -> list[tuple[StepResult, EvalResult] | None]:
        """Execute each candidate from the *current* state and roll back: AppWorld's checkpoint restores the app
        databases, and a namespace snapshot restores the Python shell. Verified equivalent to replay-forking."""
        assert self._task_id
        sid = f"cf{int(time.time() * 1000) % 10_000_000}"
        self._dirty = True
        self._post("/save_state", {"task_id": self._task_id, "state_id": sid})
        out = []
        for a in actions:
            if not a or not a.get("code"):
                out.append(None); continue
            self._post("/execute", {"task_id": self._task_id, "code": self._NS_SNAP})
            r = self.step(a)
            ev = self.evaluate()
            out.append((r, ev))
            self._post("/load_state", {"task_id": self._task_id, "state_id": sid})
            self._post("/execute", {"task_id": self._task_id, "code": self._NS_RESTORE})
        return out

    def action_schema(self) -> dict[str, Any]:
        if self._schema_cache is None:
            docs = self._get("/api_docs")
            self._schema_cache = {"action_language": "execute_code", "apps": {app: sorted(apis.keys()) for app, apis in docs.items()}, "api_docs": docs}
        return self._schema_cache

    def evaluate(self) -> EvalResult:
        assert self._task_id
        d = self._post("/evaluate", {"task_id": self._task_id, "suppress_errors": True, "report": False})
        return EvalResult(success=bool(d.get("success")), num_tests=d.get("num_tests"),
                          passes=[str(p) for p in d.get("passes", [])], failures=[str(f) for f in d.get("failures", [])], raw=d)

    def close(self) -> None:
        if self._task_id:
            if not self._dirty:  # after a rollback /close raises inside AppWorld; the restart at reset() replaces it
                try:
                    self._post("/close", {"task_id": self._task_id})
                except Exception:
                    pass
            self._task_id = None

    def shutdown(self) -> None:
        self.close()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
