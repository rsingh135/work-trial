"""Persistent structured world state for the agent (PERSIST-style memory, arXiv:2603.03482 applied to
tool-use agents).

The raw-history agent conditions on a window of past tool outputs (the "pixel history"): truncated,
re-parsed every turn, and the source of invented API names, repeated logins and forgotten results.
The world-frame instead keeps a compact, explicitly updated state that is *rendered* into the prompt:

  apps        apps the episode has touched or the task implies
  apis        per app: API names known (from show_api_descriptions or the seeded schema) and signatures
              of the ones whose docs were read (show_api_doc)
  auth        per app: the Python variable currently holding a live access token
  entities    per app.api: how many records were returned, their id field, the first ids, the keys
  facts       short free-text notes the agent chose to remember (via `note("...")` lines in its code)
  errors      last error per app.api (so the same mistake is not repeated)

Initialisation (the paper's explicit w0): apps implied by the task instruction are pre-populated with
their API name list from the environment's published action schema plus the login signature.
Rendering is a fixed-cost block regardless of episode length; raw history is kept to the last few turns.
"""

from __future__ import annotations

import json
import re
from typing import Any

_API = re.compile(r"apis\.([a-z_]+)\.([a-z_]+)\(")
_TOKEN_ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*apis\.([a-z_]+)\.login\(.*?\)\s*\[\s*['\"]access_token['\"]\s*\]", re.M)
_TOKEN_DICT = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*apis\.([a-z_]+)\.login\(", re.M)
_NOTE = re.compile(r"^\s*note\(\s*(['\"])(.*?)\1\s*\)", re.M)
_DOC_CALL = re.compile(r"apis\.api_docs\.show_api_doc\(\s*app_name\s*=\s*['\"]([a-z_]+)['\"]\s*,\s*api_name\s*=\s*['\"]([a-z_]+)['\"]")
_DESC_CALL = re.compile(r"apis\.api_docs\.show_api_descriptions\(\s*app_name\s*=\s*['\"]([a-z_]+)['\"]")

APP_KEYWORDS = {
    "spotify": ["spotify", "song", "playlist", "album", "artist", "music"],
    "venmo": ["venmo", "pay ", "payment", "money", "owe", "rent"],
    "gmail": ["gmail", "email", "e-mail", "mail", "inbox"],
    "amazon": ["amazon", "order", "product", "cart", "purchase", "return"],
    "splitwise": ["splitwise", "split", "expense", "bill"],
    "todoist": ["todoist", "todo", "task list", "to-do", "project"],
    "phone": ["phone", "contact", "text message", "sms", "alarm", "message"],
    "file_system": ["file", "folder", "directory"],
    "simple_note": ["note", "notes"],
}


class WorldFrame:
    def __init__(self, max_entities_per_key: int = 5, max_facts: int = 12):
        self.apps: set[str] = set()
        self.apis: dict[str, dict[str, str | None]] = {}  # app -> {api: signature or None}
        self.auth: dict[str, str] = {}  # app -> variable name holding the token
        self.entities: dict[str, dict[str, Any]] = {}  # "app.api" -> summary
        self.facts: list[str] = []
        self.errors: dict[str, str] = {}
        self.max_entities_per_key, self.max_facts = max_entities_per_key, max_facts
        self.seeded: list[str] = []

    # ------------------------------------------------------------------ initialisation (w0)
    def seed(self, instruction: str, schema: dict[str, Any] | None) -> None:
        if not schema or "apps" not in schema:
            return
        text = instruction.lower()
        implied = [app for app, kws in APP_KEYWORDS.items() if app in schema["apps"] and any(k in text for k in kws)]
        docs = schema.get("api_docs", {})
        for app in implied:
            self.apps.add(app)
            self.apis.setdefault(app, {})
            for api in schema["apps"][app]:
                self.apis[app].setdefault(api, None)
            login = (docs.get(app, {}) or {}).get("login")
            if login:
                self.apis[app]["login"] = _signature(app, "login", login)
        self.seeded = implied

    # ------------------------------------------------------------------ update per executed step
    def update(self, code: str, result: str, error: dict | None) -> None:
        for app, api in _API.findall(code or ""):
            if app == "api_docs":
                continue
            self.apps.add(app)
            self.apis.setdefault(app, {}).setdefault(api, None)
        for note in _NOTE.findall(code or ""):
            self._fact(note[1])
        if error:
            for app, api in _API.findall(code or ""):
                if app != "api_docs":
                    self.errors[f"{app}.{api}"] = error.get("message", "")[:120]
            return
        for var, app in _TOKEN_ASSIGN.findall(code or "") or _TOKEN_DICT.findall(code or ""):
            self.auth[app] = var
        # docs read → signatures / api lists
        m = _DESC_CALL.search(code or "")
        if m:
            app = m.group(1)
            for name in _api_names_from_output(result):
                self.apis.setdefault(app, {}).setdefault(name, None)
        m = _DOC_CALL.search(code or "")
        if m:
            app, api = m.group(1), m.group(2)
            sig = _signature_from_doc_output(app, api, result)
            if sig:
                self.apis.setdefault(app, {})[api] = sig
        # entities from JSON outputs of non-docs calls
        calls = [(a, b) for a, b in _API.findall(code or "") if a not in ("api_docs", "supervisor")]
        if calls:
            summary = _entity_summary(result, self.max_entities_per_key)
            if summary:
                self.entities[f"{calls[-1][0]}.{calls[-1][1]}"] = summary
                self.errors.pop(f"{calls[-1][0]}.{calls[-1][1]}", None)

    def _fact(self, text: str) -> None:
        text = text.strip()[:160]
        if text and text not in self.facts:
            self.facts.append(text)
            self.facts = self.facts[-self.max_facts:]

    # ------------------------------------------------------------------ render
    def render(self) -> str:
        lines = ["## World state (persistent; updated after every step)"]
        if self.seeded:
            lines.append(f"apps implied by the task: {', '.join(self.seeded)}")
        for app in sorted(self.apps):
            apis = self.apis.get(app, {})
            known = [f"{k}{'' if v is None else ' ' + v}" for k, v in sorted(apis.items())]
            tok = self.auth.get(app)
            lines.append(f"- {app}: token var = {tok if tok else 'NONE (login first)'}; known APIs ({len(known)}): " + (", ".join(known)[:900] if known else "call show_api_descriptions"))
        if self.entities:
            lines.append("entities seen:")
            for k, v in list(self.entities.items())[-10:]:
                lines.append(f"- {k}: {v}")
        if self.errors:
            lines.append("last errors (do not repeat): " + "; ".join(f"{k}: {v}" for k, v in list(self.errors.items())[-5:]))
        if self.facts:
            lines.append("facts noted: " + " | ".join(self.facts))
        lines.append("(to remember something, include a line note(\"...\") in your code)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"apps": sorted(self.apps), "apis": {a: sorted(v) for a, v in self.apis.items()}, "auth": self.auth, "entities": self.entities,
                "facts": self.facts, "errors": self.errors, "seeded": self.seeded}


# ---------------------------------------------------------------------- parsing helpers
def _signature(app: str, api: str, doc: dict) -> str:
    params = doc.get("parameters", []) if isinstance(doc, dict) else []
    ps = [f"{p.get('name')}{'' if p.get('required') else '?'}" for p in params if isinstance(p, dict)]
    return f"({', '.join(ps)})"


def _api_names_from_output(result: str) -> list[str]:
    data = _first_json(result)
    if data is None:
        return re.findall(r'"name":\s*"([a-z_]+)"', result)[:80]
    if isinstance(data, list):
        return [d.get("name") for d in data if isinstance(d, dict) and d.get("name")][:80]
    return []


def _signature_from_doc_output(app: str, api: str, result: str) -> str | None:
    d = _first_json(result)
    if d is None:
        return None
    if isinstance(d, dict) and "parameters" in d:
        return _signature(app, api, d)
    return None


def _first_json(text: str):
    """Parse the whole output as JSON, else the first top-level JSON array/object substring."""
    try:
        return json.loads(text)
    except Exception:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        i = text.find(opener)
        while i != -1:
            depth = 0
            for j in range(i, len(text)):
                if text[j] == opener:
                    depth += 1
                elif text[j] == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i:j + 1])
                        except Exception:
                            break
            i = text.find(opener, i + 1)
    return None


def _entity_summary(result: str, k: int) -> dict[str, Any] | None:
    d = _first_json(result)
    if d is None:
        return None
    if isinstance(d, list) and d and all(isinstance(x, dict) for x in d):
        keys = sorted(set().union(*[x.keys() for x in d[:20]]))
        idk = next((key for key in keys if key.endswith("_id") or key == "id"), None)
        return {"count": len(d), "id_field": idk, "first_ids": [x.get(idk) for x in d[:k]] if idk else None, "keys": keys[:15]}
    if isinstance(d, dict):
        keys = sorted(d.keys())
        return {"count": 1, "keys": keys[:15], "values": {key: (str(d[key])[:40]) for key in keys[:6]}}
    return None
