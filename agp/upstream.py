"""AGP module: upstream (split from antigravity_proxy.py)."""

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

from agp.auth import get_access_token
from agp.constants import ANTIGRAVITY_USER_AGENT, CLIENT_METADATA, GENERATE_CONTENT_URL, STREAM_GENERATE_CONTENT_URL, UPSTREAM_TIMEOUT, _LOCATION_RETRIES, _LOCATION_RETRY_DELAY
from agp.log import _log

def _make_upstream_request(url: str, envelope: dict, streaming: bool = False):
    """Build and return a urllib Request object for the upstream call."""
    body = json.dumps(envelope).encode()
    req = Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {get_access_token()}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", ANTIGRAVITY_USER_AGENT)
    req.add_header("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
    req.add_header("Client-Metadata", CLIENT_METADATA)
    if streaming:
        req.add_header("Accept", "text/event-stream")
    return req

def _is_location_error(e) -> bool:
    body = getattr(e, "body", "") or ""
    return "location is not supported" in body.lower()

def call_generate_content(envelope: dict) -> dict:
    """Non-streaming call. Returns the unwrapped inner Gemini response dict.

    Retries transient "User location is not supported" rejections.
    """
    last_exc = None
    for attempt in range(_LOCATION_RETRIES + 1):
        req = _make_upstream_request(GENERATE_CONTENT_URL, envelope, streaming=False)
        try:
            with urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:
                raw = resp.read().decode()
        except HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode()
            except Exception:
                pass
            last_exc = UpstreamError(f"Upstream HTTP {e.code}: {detail}", status=e.code, body=detail)
            if _is_location_error(last_exc) and attempt < _LOCATION_RETRIES:
                _log(f"location-limited upstream error, retrying ({attempt + 1}/{_LOCATION_RETRIES})...")
                time.sleep(_LOCATION_RETRY_DELAY * (attempt + 1))
                continue
            raise last_exc
        except URLError as e:
            raise UpstreamError(f"Upstream network error: {e}", status=502)

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise UpstreamError(f"Upstream returned non-JSON: {raw[:500]}", status=502)

        # Unwrap the envelope: {"response": {<gemini>}} or sometimes bare.
        if isinstance(payload, dict) and "response" in payload:
            return payload["response"]
        return payload
    raise last_exc  # pragma: no cover - loop always returns or raises

def _stream_once(envelope: dict):
    """Single streaming attempt; yields parsed SSE event objects."""
    req = _make_upstream_request(STREAM_GENERATE_CONTENT_URL, envelope, streaming=True)
    try:
        resp = urlopen(req, timeout=UPSTREAM_TIMEOUT)
    except HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()
        except Exception:
            pass
        raise UpstreamError(f"Upstream stream HTTP {e.code}: {detail}", status=e.code, body=detail)
    except URLError as e:
        raise UpstreamError(f"Upstream stream network error: {e}", status=502)

    try:
        buffer = ""
        # Read in chunks; urllib's response supports iteration by bytes.
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            if isinstance(chunk, bytes):
                text_chunk = chunk.decode("utf-8", errors="replace")
            else:
                text_chunk = chunk
            buffer += text_chunk

            # Try to parse complete SSE events or JSON array elements.
            while True:
                parsed_event = _try_parse_one_event(buffer)
                if parsed_event is None:
                    break
                event_obj, consumed = parsed_event
                buffer = buffer[consumed:]
                if event_obj is not None:
                    yield event_obj
        # Flush any remaining buffer.
        if buffer.strip():
            for obj in _parse_remaining(buffer):
                if obj is not None:
                    yield obj
    finally:
        resp.close()

def stream_generate_content(envelope: dict):
    """Streaming call. Yields parsed JSON event objects from the SSE stream.

    Retries transient "User location is not supported" rejections; only the
    connection/HTTP stage is retried (before any event has been yielded).

    The Antigravity stream endpoint returns either:
      - SSE format: lines "data: {json}\n\n"
      - Or a bare JSON array of incremental response objects (some endpoints).
    We handle both.
    """
    attempt = 0
    while True:
        try:
            yield from _stream_once(envelope)
            return
        except UpstreamError as e:
            if not _is_location_error(e) or attempt >= _LOCATION_RETRIES:
                raise
            attempt += 1
            _log(f"location-limited stream error, retrying ({attempt}/{_LOCATION_RETRIES})...")
            time.sleep(_LOCATION_RETRY_DELAY * attempt)

def _try_parse_one_event(buffer: str):
    """Attempt to parse one SSE event or JSON array element from the buffer.

    Returns (event_obj_or_None, bytes_consumed) or None if incomplete.
    The event_obj may be None when it's a comment/heartbeat line.
    """
    # SSE "data:" lines.
    if "data:" in buffer[:10] or buffer.startswith("\n") or buffer.startswith(":"):
        # Find end of this SSE block (double newline).
        for sep in ("\n\n", "\r\n\r\n"):
            idx = buffer.find(sep)
            if idx != -1:
                block = buffer[:idx]
                consumed = idx + len(sep)
                obj = _parse_sse_block(block)
                return (obj, consumed)
        # Maybe single newline-terminated line (some servers use \n not \n\n).
        nl = buffer.find("\n")
        if nl != -1 and buffer.strip().startswith("data:"):
            # Only consume if we have a complete data line.
            line = buffer[:nl]
            consumed = nl + 1
            obj = _parse_sse_block(line)
            return (obj, consumed)
        return None  # incomplete

    # Bare JSON array stream: [ {...}, {...} ]
    if buffer.lstrip().startswith("["):
        obj, consumed = _try_parse_json_element_in_array(buffer)
        if obj is not None or consumed > 0:
            return (obj, consumed)
        return None

    # Bare JSON object stream: {...}{...}
    if buffer.lstrip().startswith("{"):
        obj, consumed = _try_parse_one_json_object(buffer)
        if obj is not None:
            return (obj, consumed)
        return None

    # Unknown prefix; skip a line to resync.
    nl = buffer.find("\n")
    if nl != -1:
        return (None, nl + 1)
    return None

def _parse_sse_block(block: str):
    """Parse an SSE block (one or more lines) into a JSON object or None."""
    data_lines = []
    for line in block.split("\n"):
        line = line.rstrip("\r")
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif line.startswith(":"):
            continue  # comment / heartbeat
        elif line.startswith("event:") or line.startswith("id:") or line.startswith("retry:"):
            continue
        elif line.strip() == "":
            continue
        else:
            # Unexpected line; ignore.
            pass
    if not data_lines:
        return None
    data_str = "\n".join(data_lines).strip()
    if data_str == "[DONE]":
        return {"__done__": True}
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        return None

def _try_parse_json_element_in_array(buffer: str):
    """For bare JSON array streams like [{...},{...}]. Parse one element."""
    s = buffer.lstrip()
    if not s.startswith("["):
        # Maybe we already consumed the opening bracket.
        pass
    # We need to find a complete top-level object. Use brace counting.
    obj, consumed = _try_parse_one_json_object(buffer)
    return (obj, consumed)

def _try_parse_one_json_object(buffer: str):
    """Parse one complete top-level JSON object from buffer using brace counting.

    Returns (obj, consumed_chars) or (None, 0) if incomplete.
    """
    # Find first '{'.
    start = buffer.find("{")
    if start == -1:
        return (None, 0)
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(buffer)):
        ch = buffer[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = buffer[start:i + 1]
                try:
                    obj = json.loads(candidate)
                    return (obj, i + 1)
                except json.JSONDecodeError:
                    # Keep scanning.
                    continue
    return (None, 0)

def _parse_remaining(buffer: str):
    """Parse any remaining JSON objects in the buffer at stream end."""
    s = buffer.strip()
    if not s:
        return
    # SSE.
    for line in s.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                yield {"__done__": True}
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                pass
    # Bare JSON.
    if s.startswith("[") or s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                for item in obj:
                    yield item
            else:
                yield obj
        except json.JSONDecodeError:
            pass

class UpstreamError(Exception):
    def __init__(self, message, status=502, body=""):
        super().__init__(message)
        self.status = status
        self.body = body
