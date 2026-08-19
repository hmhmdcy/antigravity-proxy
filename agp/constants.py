"""AGP module: shared constants (split from antigravity_proxy.py)."""

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

CLIENT_ID = os.environ.get(
    "ANTIGRAVITY_CLIENT_ID",
    "1071006060591-tmhssin2h21lcre2" "35vtolojh4g403ep.apps.googleusercontent.com",
)
CLIENT_SECRET = os.environ.get(
    "ANTIGRAVITY_CLIENT_SECRET",
    "GOCSPX-K58FWR486LdL" "J1mLB8sXC4z6qDAf",
)
TOKEN_FILE = os.path.expanduser("~/.gemini/antigravity-cli/antigravity-oauth-token")
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
CLOUDCODE_BASE = "https://daily-cloudcode-pa.sandbox.googleapis.com"
LOAD_CODEASSIST_URL = f"{CLOUDCODE_BASE}/v1internal:loadCodeAssist"
GENERATE_CONTENT_URL = f"{CLOUDCODE_BASE}/v1internal:generateContent"
FETCH_MODELS_URL = f"{CLOUDCODE_BASE}/v1internal:fetchAvailableModels"
STREAM_GENERATE_CONTENT_URL = f"{CLOUDCODE_BASE}/v1internal:streamGenerateContent?alt=sse"
ANTIGRAVITY_USER_AGENT = "antigravity/2.8.1 windows/amd64"
CLIENT_METADATA = json.dumps(
    {"ideType": "ANTIGRAVITY", "platform": "MACOS", "pluginType": "GEMINI"},
    separators=(",", ":"),
)
DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful AI assistant."
UPSTREAM_TIMEOUT = 300  # 5 minutes
TOKEN_REFRESH_SKEW = 120  # refresh this many seconds before actual expiry
_LOCATION_RETRIES = 3
_LOCATION_RETRY_DELAY = 1.0
MODEL_MAP = {
    "gemini-3.1-pro": ("gemini-3.1-pro-low", "low"),
    "gemini-3.1-pro-high": ("gemini-3.1-pro-low", "high"),
    "gemini-3-flash": ("gemini-3-flash", None),
    "gemini-3.5-flash": ("gemini-3.5-flash-low", None),
    "gemini-2.5-pro": ("gemini-2.5-pro", None),
    "gemini-2.5-flash": ("gemini-2.5-flash", None),
    "claude-sonnet-4.6": ("claude-sonnet-4-6", None),
    "claude-opus-4.6": ("claude-opus-4-6-thinking", None),
    "gemini-3.6-flash": ("gemini-3.6-flash-high", None),
    "gemini-3.7-flash": ("gemini-3.7-flash-high", None),
}
DEFAULT_MODEL = "gemini-3.6-flash"
_token_lock = threading.Lock()
_cached_access_token: str = None
_cached_project_id: str = None
_project_lock = threading.Lock()
_GEMINI_SCHEMA_ALLOWED = {
    "type", "nullable", "enum", "description", "format", "items",
    "properties", "required", "minItems", "maxItems", "minLength",
    "maxLength", "pattern", "minimum", "maximum", "minProperties",
    "maxProperties", "default", "additionalProperties",
}
_GEMINI_SCHEMA_FOLD_TO_DESC = (
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "patternProperties", "$ref", "$schema",
)
_FINISH_TO_RESPONSES_STATUS = {
    "STOP": "completed",
    "MAX_TOKENS": "incomplete",
    "SAFETY": "incomplete",
    "RECITATION": "incomplete",
    "BLOCKLIST": "incomplete",
    "PROHIBITED_CONTENT": "incomplete",
    "SPII": "incomplete",
    "IMAGE_SAFETY": "incomplete",
    "LANGUAGE": "incomplete",
    "OTHER": "completed",
}
_GEMINI_PASSTHROUGH_RE = re.compile(
    r"^/v1beta/models/(.+):generateContent$"
)
_GEMINI_STREAM_PASSTHROUGH_RE = re.compile(
    r"^/v1beta/models/(.+):streamGenerateContent$"
)
