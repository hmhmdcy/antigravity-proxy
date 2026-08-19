import os
"""Dynamic model discovery and MODEL_MAP merging.

At startup we ask Antigravity's fetchAvailableModels for the real backend
model ids (e.g. "gemini-3.7-flash-high"), then build the friendly-name ->
(backend, thinking_level) map:

  - Manual MODEL_MAP entries (constants.py) are kept verbatim — they carry
    intent (claude mapping, chat_* internal names, custom tiers).
  - Discovered backend ids are added under their tier-stripped friendly
    name ("gemini-3.7-flash-high" -> "gemini-3.7-flash") with the thinking
    level derived from the tier suffix.
  - Discovered ids that have no tier suffix (e.g. "gemini-3-flash") map
    to themselves with thinking_level None.

This way a new upstream model (say gemini-3.8-flash-*) appears in
/v1/models and is routable without a code change.
"""
import json
import re
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from agp.auth import get_access_token, get_project_id
from agp.constants import ANTIGRAVITY_USER_AGENT, FETCH_MODELS_URL, MODEL_MAP

# Tier suffix -> thinking level used by _build_gemini_request.
_TIER_TO_LEVEL = {
    "-high": "high",
    "-medium": "medium",
    "-low": "low",
    "-extra-low": "low",
    "-tiered": "medium",
}

_lock = threading.Lock()
_merged = None  # cached merged map


def _fetch_upstream_models():
    """Return dict of upstream backend id -> ModelInfo. Raises on failure."""
    tok = get_access_token()
    pid = get_project_id()
    req = Request(FETCH_MODELS_URL, data=json.dumps({"project": pid}).encode(),
                  method="POST")
    req.add_header("Authorization", f"Bearer {tok}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", ANTIGRAVITY_USER_AGENT)
    proxy_url = os.environ.get("UPSTREAM_PROXY", "").strip()
    opener = None
    if proxy_url:
        import urllib.request as _ur
        opener = _ur.build_opener(_ur.ProxyHandler({"http": proxy_url, "https": proxy_url}))
    try:
        if opener is not None:
            with opener.open(req, timeout=30) as r:
                payload = json.loads(r.read().decode())
        else:
            with urlopen(req, timeout=30) as r:
                payload = json.loads(r.read().decode())
    except HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()
        except Exception:
            pass
        raise RuntimeError(f"fetchAvailableModels failed (HTTP {e.code}): {detail}")
    except URLError as e:
        raise RuntimeError(f"fetchAvailableModels network error: {e}")
    return payload.get("models") or {}


def _friendly_name(backend_id):
    """gemini-3.7-flash-high -> gemini-3.7-flash (strip tier suffix)."""
    for suf in sorted(_TIER_TO_LEVEL, key=len, reverse=True):
        if backend_id.endswith(suf):
            return backend_id[: -len(suf)]
    return backend_id


def _level_for(backend_id):
    for suf, level in _TIER_TO_LEVEL.items():
        if backend_id.endswith(suf):
            return level
    return None


def build_merged_model_map():
    """Merge manual MODEL_MAP with discovered upstream models.

    Returns (merged_map, discovered_count, failed). Never raises: discovery
    failure degrades to the static map so the proxy still starts.
    """
    merged = dict(MODEL_MAP)
    try:
        upstream = _fetch_upstream_models()
    except Exception as e:
        return merged, 0, str(e)

    added = 0
    for backend_id in upstream:
        friendly = _friendly_name(backend_id)
        # Manual entry wins; skip if already present (avoids clobbering
        # claude/chat_* custom mappings).
        if friendly in merged:
            continue
        level = _level_for(backend_id)
        merged[friendly] = (backend_id, level)
        added += 1
    return merged, added, None


def get_merged_model_map(force=False):
    """Thread-safe cached merged map; auto-discovers on first call."""
    global _merged
    with _lock:
        if _merged is None or force:
            merged, added, err = build_merged_model_map()
            _merged = merged
            return merged, added, err
        return _merged, 0, None
