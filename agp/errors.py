"""AGP module: errors (split from antigravity_proxy.py)."""

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

def _openai_error(message: str, err_type: str = "api_error", code=None, status: int = 500):
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": None,
            "code": code,
        }
    }

def _stream_error_payload(message: str, err_type: str = "api_error", code=None, status: int = 500):
    """Build an OpenAI-style SSE error event payload.

    The non-streaming error body is
        {"error": {"message": ..., "type": ..., "param": null, "code": ...}}
    and per the SSE convention the event payload carries the same shape
    (the event name may also be "error"; clients key on the payload).
    """
    return _openai_error(message, err_type, code, status)
