"""AGP module: content (split from antigravity_proxy.py)."""

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

def _content_to_parts(content) -> list:
    """Build Gemini parts from OpenAI message content. Text blocks become
    text parts; image_url data URLs become inlineData parts (real vision
    input). Returns a list of part dicts (possibly empty)."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "text")
            if btype == "text":
                t = block.get("text", "")
                if t:
                    parts.append({"text": t})
            elif btype == "image_url":
                iu = block.get("image_url")
                url = iu.get("url", "") if isinstance(iu, dict) else (iu or "")
                if isinstance(url, str) and url.startswith("data:"):
                    try:
                        meta, data = url.split(",", 1)
                        mime = meta[5:].split(";", 1)[0] or "application/octet-stream"
                        if ";base64" in meta:
                            parts.append({"inlineData": {"mimeType": mime, "data": data}})
                        else:
                            import base64
                            parts.append({"inlineData": {"mimeType": mime,
                                                         "data": base64.b64encode(data.encode()).decode()}})
                    except Exception:
                        parts.append({"text": f"[image: {url[:60]}]"})
                elif url:
                    parts.append({"text": f"[image: {url}]"})
            elif btype == "input_text":
                t = block.get("text", "")
                if t:
                    parts.append({"text": t})
            elif btype == "input_image":
                parts.append({"text": "[image provided]"})
            else:
                t = block.get("text")
                if t:
                    parts.append({"text": f"[{btype}: {str(t)[:100]}]"})
        return parts
    return [{"text": str(content)}] if str(content) else []

def _content_to_text(content) -> str:
    """Normalize an OpenAI message 'content' field to plain text.

    Handles str, list of content blocks [{type: text, text: ...}, ...], and None.
    Non-text blocks (images, etc.) are represented as a placeholder marker so
    the model knows something was there even if we can't inline binary data.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for block in content:
            if not isinstance(block, dict):
                chunks.append(str(block))
                continue
            btype = block.get("type", "text")
            if btype == "text":
                chunks.append(block.get("text", ""))
            elif btype == "image_url":
                url = ""
                iu = block.get("image_url")
                if isinstance(iu, dict):
                    url = iu.get("url", "")
                elif isinstance(iu, str):
                    url = iu
                # We pass the data URL through; Gemini's inlineData supports it.
                if url.startswith("data:"):
                    # Best-effort: note its presence; full image passthrough
                    # would require parts[].inlineData. Keep as text marker.
                    chunks.append(f"[image: {url[:60]}...]")
                else:
                    chunks.append(f"[image: {url}]")
            elif btype == "input_text":
                chunks.append(block.get("text", ""))
            elif btype == "input_image":
                chunks.append("[image provided]")
            else:
                chunks.append(f"[{btype}: {json.dumps(block.get('text', ''))[:100]}]")
        return "\n".join(c for c in chunks if c)
    return str(content)

def _normalize_responses_content(content):
    """Responses message content (str or list of parts) -> chat content string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            pt = part.get("type")
            if pt in ("input_text", "output_text", "text"):
                txt = part.get("text")
                if isinstance(txt, str):
                    texts.append(txt)
        return "".join(texts)
    return ""
