"""AGP module: schema (split from antigravity_proxy.py)."""

import json
import os
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from agp.constants import _GEMINI_SCHEMA_ALLOWED, _GEMINI_SCHEMA_FOLD_TO_DESC

def _schema_constraint_suffix(schema):
    hints = []
    for key in _GEMINI_SCHEMA_FOLD_TO_DESC:
        val = schema.get(key)
        if val is not None and val is not False:
            if isinstance(val, (dict, list)):
                val = json.dumps(val, separators=(",", ":"), sort_keys=True)
            hints.append(key + ": " + str(val))
    return " [Constraint: " + ", ".join(hints) + "]" if hints else ""

def _sanitize_gemini_schema(schema):
    """Rewrite an OpenAI-style JSON Schema into the subset Gemini's function
    calling accepts.

    - type arrays: ["string","null"] -> type + nullable (Gemini proto field
      is not repeating, cannot start list)
    - enum with null: strip null, set nullable
    - additionalProperties:false is ACCEPTED by the sandbox (litellm keeps
      it too) -> preserve instead of dropping
    - rejected keywords (exclusiveMinimum/multipleOf/$ref/patternProperties/
      $schema) are folded into description as [Constraint: ...] so the model
      still sees them (sub2api approach).
    """
    if not isinstance(schema, dict):
        return schema
    out = {}
    constraint_suffix = _schema_constraint_suffix(schema)
    for key, value in schema.items():
        if key == "type" and isinstance(value, list):
            non_null = [t for t in value if t != "null"]

            if non_null:
                out["type"] = non_null[0]
            else:
                out["type"] = "string"
            if "null" in value:
                out["nullable"] = True
        elif key == "enum" and isinstance(value, list):
            if any(v is None for v in value):
                out["nullable"] = True
                value = [v for v in value if v is not None]
            if value:
                out["enum"] = value
        elif key == "items" and isinstance(value, dict):
            out["items"] = _sanitize_gemini_schema(value)
        elif key == "properties" and isinstance(value, dict):
            out["properties"] = {k: _sanitize_gemini_schema(v) for k, v in value.items()}
        elif key == "additionalProperties" and value is False:
            out[key] = False
        elif key == "definitions" and isinstance(value, dict):
            pass  # container stripped; refs folded to description, not resolved
        elif key in _GEMINI_SCHEMA_ALLOWED:
            out[key] = value
        # Everything else is dropped (handled by constraint suffix).
    if constraint_suffix:
        desc = out.get("description", "")
        if desc and constraint_suffix not in desc:
            out["description"] = desc + constraint_suffix
        elif not desc:
            out["description"] = constraint_suffix.strip()
    return out
