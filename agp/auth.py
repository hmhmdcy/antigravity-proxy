"""AGP module: auth (split from antigravity_proxy.py)."""

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

from agp.constants import ANTIGRAVITY_USER_AGENT, CLIENT_ID, CLIENT_METADATA, CLIENT_SECRET, LOAD_CODEASSIST_URL, OAUTH_TOKEN_URL, TOKEN_FILE, TOKEN_REFRESH_SKEW, _cached_project_id, _project_lock, _token_lock
from agp.log import _log

def _parse_expiry(expiry_raw):
    """Parse the 'expiry' field from the token file into a unix timestamp.

    The Antigravity CLI stores expiry as an RFC3339 string like
    "2026-06-30T12:55:03.123456789Z". Python stdlib datetime.fromisoformat
    can't handle nanosecond precision, so we truncate to 6 digits.
    """
    if not expiry_raw:
        return 0.0
    if isinstance(expiry_raw, (int, float)):
        return float(expiry_raw)
    s = str(expiry_raw).strip()
    # Truncate sub-second precision to microseconds for fromisoformat.
    try:
        if "." in s:
            head, tail = s.split(".", 1)
            # Keep only the part up to 'Z' or timezone offset, truncate ns.
            tz_part = ""
            for i, ch in enumerate(tail):
                if ch in "Z+-":
                    tz_part = tail[i:]
                    tail = tail[:i]
                    break
            tail = tail[:6]  # microseconds max
            s = f"{head}.{tail}{tz_part}"
        # fromisoformat in 3.12 handles 'Z' suffix.
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0

def _read_token_from_disk():
    """Read and parse the token file. Returns the raw dict or None."""
    try:
        with open(TOKEN_FILE, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        _log(f"WARNING: could not read token file {TOKEN_FILE}: {e}")
        return None

def _write_token_to_disk(data: dict):
    """Persist updated token data back to disk (atomic-ish)."""
    try:
        tmp = TOKEN_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, TOKEN_FILE)
        try:
            os.chmod(TOKEN_FILE, 0o600)
        except OSError:
            pass
    except OSError as e:
        _log(f"WARNING: could not write token file: {e}")

def _auth_opener():
    """Open an opener routing through UPSTREAM_PROXY when configured."""
    proxy_url = os.environ.get("UPSTREAM_PROXY", "").strip()
    if not proxy_url:
        return None
    import urllib.request
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
    )


def _auth_open_url(req, timeout):
    import urllib.request
    opener = _auth_opener()
    if opener is not None:
        return opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def _refresh_access_token(refresh_token: str) -> dict:
    """Refresh the access token via Google's OAuth endpoint.

    Returns the new token dict {access_token, refresh_token?, expiry, token_type}.
    Raises RuntimeError on failure.
    """
    body = json.dumps({
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }).encode()
    req = Request(OAUTH_TOKEN_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with _auth_open_url(req, 30) as resp:
            payload = json.loads(resp.read().decode())
    except HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()
        except Exception:
            pass
        raise RuntimeError(f"OAuth refresh failed (HTTP {e.code}): {detail}")
    except URLError as e:
        raise RuntimeError(f"OAuth refresh network error: {e}")

    new_tok = {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token", refresh_token),
        "token_type": payload.get("token_type", "Bearer"),
        "expiry": payload.get("expires_in", 3600),
    }
    # Convert expires_in (seconds) to an RFC3339 expiry timestamp ~now.
    expires_in = int(payload.get("expires_in", 3600))
    exp_dt = datetime.now(timezone.utc)
    from datetime import timedelta
    exp_dt = exp_dt + timedelta(seconds=expires_in)
    new_tok["expiry"] = exp_dt.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
    return new_tok

def get_access_token() -> str:
    """Return a valid access token, refreshing if necessary. Thread-safe."""
    global _cached_access_token
    with _token_lock:
        data = _read_token_from_disk()
        if not data:
            raise RuntimeError(
                f"No OAuth token found at {TOKEN_FILE}. "
                "Run `agy` to authenticate first."
            )
        token_obj = data.get("token") or {}
        access_token = token_obj.get("access_token")
        expiry_ts = _parse_expiry(token_obj.get("expiry"))
        now = time.time()

        needs_refresh = (
            not access_token
            or expiry_ts == 0.0
            or (expiry_ts - now) < TOKEN_REFRESH_SKEW
        )

        if needs_refresh:
            refresh_token = token_obj.get("refresh_token")
            if not refresh_token:
                raise RuntimeError("No refresh_token available; re-run `agy` to authenticate.")
            _log("Access token expired (or missing) — refreshing...")
            new_tok = _refresh_access_token(refresh_token)
            data["token"] = new_tok
            _write_token_to_disk(data)
            access_token = new_tok["access_token"]

        _cached_access_token = access_token
        return access_token

def _load_code_assist(access_token: str) -> str:
    """Discover the cloudaicompanion project ID. Returns project id string.

    NOTE: The metadata field (ideType/platform/pluginType) is rejected by the
    API with INVALID_ARGUMENT for all known enum string values. Sending an
    empty body {} works and returns the cloudaicompanionProject. The Client-Metadata
    header still carries the ide/plugin info.
    """
    body = json.dumps({}).encode()
    req = Request(LOAD_CODEASSIST_URL, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", ANTIGRAVITY_USER_AGENT)
    req.add_header("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
    req.add_header("Client-Metadata", CLIENT_METADATA)
    try:
        with _auth_open_url(req, 60) as resp:
            payload = json.loads(resp.read().decode())
    except HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()
        except Exception:
            pass
        raise RuntimeError(f"loadCodeAssist failed (HTTP {e.code}): {detail}")
    except URLError as e:
        raise RuntimeError(f"loadCodeAssist network error: {e}")

    project_id = (
        payload.get("cloudaicompanionProject")
        or payload.get("cloudaicompanion_project")
    )
    if not project_id:
        # Some responses nest it differently; do a recursive search.
        project_id = _deep_find(payload, "cloudaicompanionProject")
    if not project_id:
        raise RuntimeError(
            f"loadCodeAssist did not return a project ID. Response: {payload}"
        )
    return project_id

def _deep_find(obj, key):
    """Recursively search for a key in nested dicts/lists. Returns first match."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _deep_find(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _deep_find(item, key)
            if r is not None:
                return r
    return None

def get_project_id() -> str:
    """Return cached project id, discovering it if needed. Thread-safe."""
    global _cached_project_id
    with _project_lock:
        if _cached_project_id:
            return _cached_project_id
        token = get_access_token()
        pid = _load_code_assist(token)
        _cached_project_id = pid
        _log(f"Discovered project ID: {pid}")
        return pid
