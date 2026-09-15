"""Redaction rules applied to every string in a trace before it is written to disk.

Rules (version 2; v1 missed three shapes found in a pre-release scan of the traces, see PROGRESS.md):
  R1  JSON-ish key/value pairs whose key matches SECRET_KEYS → value replaced by
      ``<REDACTED:<8-hex sha256 of value>>`` (hash keeps equality checks possible without
      revealing the value; the hash is salted per collection run so cross-run rainbow
      matching is not possible).
  R2  Python assignments and keyword arguments whose name *contains* a secret key
      (``password='...'``, ``venmo_password = "..."``, ``access_token=...``) → same. A string
      cut off by the token limit (no closing quote) is redacted to the end of the line.
  R3  JWT-style blobs (``eyJ…`` with two or more dot-separated segments) → same.
Not redacted (documented): supervisor first/last name, e-mail and phone number in AppWorld are
synthetic persona data that the task instruction itself references; masking them would break
trace readability without protecting anyone. Real deployments should add them to SECRET_KEYS.
The agent process always works on unredacted values; redaction is purely at serialisation.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

RULES_VERSION = "2"
SECRET_KEYS = ("password", "passwd", "secret", "api_key", "apikey", "access_token", "auth_token", "token", "verification_code", "otp")

_KEY = "|".join(SECRET_KEYS)
# "password": "value"   or  'password': 'value'   (JSON / dict repr)
_R1 = re.compile(rf'(\b["\']?(?:{_KEY})["\']?\s*:\s*["\'])((?!<REDACTED:)[^"\'\n]*)(["\']|$)', re.IGNORECASE | re.MULTILINE)
# password='value' / password = "value"   (keyword args in code)
_R2 = re.compile(rf'(\b\w*(?:{_KEY})\w*\s*=\s*["\'])((?!<REDACTED:)[^"\'\n]*)(["\']|$)', re.IGNORECASE | re.MULTILINE)
# bearer / jwt style blobs
_R3 = re.compile(r"\b(eyJ[A-Za-z0-9_\-]{10,}(?:\.[A-Za-z0-9_\-]{10,})+)\b")


class Redactor:
    def __init__(self, salt: str = ""):
        self.salt = salt
        self.count = 0

    def _mask(self, value: str) -> str:
        h = hashlib.sha256((self.salt + value).encode()).hexdigest()[:8]
        self.count += 1
        return f"<REDACTED:{h}>"

    def text(self, s: str) -> str:
        if not s:
            return s
        s = _R1.sub(lambda m: m.group(1) + self._mask(m.group(2)) + m.group(3), s)
        s = _R2.sub(lambda m: m.group(1) + self._mask(m.group(2)) + m.group(3), s)
        s = _R3.sub(lambda m: self._mask(m.group(1)), s)
        return s

    def obj(self, x: Any) -> Any:
        """Recursively redact every string inside dicts/lists/pydantic dumps."""
        if isinstance(x, str):
            return self.text(x)
        if isinstance(x, dict):
            return {k: (self._mask(v) if isinstance(v, str) and str(k).lower() in SECRET_KEYS and v and not v.startswith("<REDACTED:") else self.obj(v)) for k, v in x.items()}
        if isinstance(x, list):
            return [self.obj(v) for v in x]
        return x
