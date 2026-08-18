"""AGP module: response (split from antigravity_proxy.py)."""

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

from agp.constants import _FINISH_TO_RESPONSES_STATUS

def _gemini_usage_to_responses_usage(gemini_resp: dict) -> dict:
    um = (gemini_resp or {}).get("usageMetadata") or {}
    return {
        "input_tokens": um.get("promptTokenCount", 0),
        "output_tokens": um.get("candidatesTokenCount", 0),
        "total_tokens": um.get("totalTokenCount", 0),
    }

def _gemini_to_responses_output(gemini_resp: dict, model: str) -> tuple[list, str]:
    """Gemini response -> (responses output items, status)."""
    output = []
    cands = (gemini_resp or {}).get("candidates") or []
    if not cands:
        return output, "incomplete"
    cand = cands[0]
    parts = ((cand.get("content") or {}).get("parts")) or []

    text_parts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
    text = "".join(text_parts)

    fcalls = [p.get("functionCall") for p in parts if isinstance(p, dict) and p.get("functionCall")]

    if text:
        output.append({
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })
    for fc in fcalls:
        if not isinstance(fc, dict):
            continue
        call_id = fc.get("id") or f"call_{uuid.uuid4().hex[:24]}"
        args = fc.get("args") or {}
        output.append({
            "id": f"fc_{uuid.uuid4().hex[:12]}",
            "type": "function_call",
            "status": "completed",
            "call_id": str(call_id),
            "name": str(fc.get("name") or ""),
            "arguments": args if isinstance(args, str) else json.dumps(args),
        })

    fr = (cand.get("finishReason") or "STOP").upper()
    status = _FINISH_TO_RESPONSES_STATUS.get(fr, "completed")
    return output, status

class _ResponsesStreamState:
    """State machine emitting OpenAI Responses SSE events from Gemini stream
    events. Event order follows the Responses API spec:
    response.created -> response.in_progress -> output_item.added
    -> content_part.added -> output_text.delta* -> output_text.done
    -> content_part.done -> output_item.done -> response.completed.
    Tool calls: output_item.added(function_call) -> function_call_arguments.delta*.
    """

    def __init__(self, resp_id: str, created: int, model: str):
        self.resp_id = resp_id
        self.created = created
        self.model = model
        self.sent_created = False
        self.sent_in_progress = False
        self.sent_msg_item = False
        self.sent_msg_part = False
        self.sent_text_done = False
        self.msg_item_id = None
        self.text_buf = []
        self.tool_items = {}  # index -> item id
        self.tool_buf = {}    # index -> [name, call_id, args_str]
        self.finished = False

    def _base_response(self, status="in_progress"):
        return {
            "id": self.resp_id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": None,
            "metadata": {},
            "model": self.model,
            "output": [],
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": None,
            "temperature": None,
            "text": {},
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "truncation": None,
            "usage": None,
            "user": None,
        }

    def initial_events(self) -> list:
        evts = []
        if not self.sent_created:
            self.sent_created = True
            evts.append({"type": "response.created", "response": self._base_response("in_progress")})
        if not self.sent_in_progress:
            self.sent_in_progress = True
            evts.append({"type": "response.in_progress", "response": self._base_response("in_progress")})
        return evts

    def _ensure_msg_item(self, evts: list):
        if self.sent_msg_item:
            return
        self.sent_msg_item = True
        self.msg_item_id = f"msg_{uuid.uuid4().hex[:12]}"
        evts.append({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": self.msg_item_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        })
        if not self.sent_msg_part:
            self.sent_msg_part = True
            evts.append({
                "type": "response.content_part.added",
                "item_id": self.msg_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            })

    def on_text_delta(self, text: str) -> list:
        evts = self.initial_events()
        self._ensure_msg_item(evts)
        self.text_buf.append(text)
        evts.append({
            "type": "response.output_text.delta",
            "item_id": self.msg_item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        })
        return evts

    def on_tool_call(self, index: int, call_id: str, name: str, args: str) -> list:
        evts = self.initial_events()
        if index in self.tool_items:
            # Gemini emits the full functionCall in one event; later duplicate
            # emissions (same name+args) are idempotent no-ops.
            if self.tool_buf[index][0] == name and self.tool_buf[index][2] == args:
                return []
            self.tool_buf[index][0] = name or self.tool_buf[index][0]
            self.tool_buf[index][1] = call_id or self.tool_buf[index][1]
            self.tool_buf[index][2] += args
        else:
            self.tool_items[index] = f"fc_{uuid.uuid4().hex[:12]}"
            self.tool_buf[index] = [name, call_id, args]
            evts.append({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": self.tool_items[index],
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": call_id,
                    "name": name,
                    "arguments": "",
                },
            })
        item_id = self.tool_items[index]
        # Split into small deltas to mimic OpenAI's token-by-token behaviour.
        for i in range(0, len(args), 10):
            evts.append({
                "type": "response.function_call_arguments.delta",
                "item_id": item_id,
                "output_index": 0,
                "delta": args[i:i + 10],
            })
        return evts

    def finish(self, finish_reason: str, usage: dict) -> list:
        evts = self.initial_events()
        # Close message item if we opened one.
        if self.sent_msg_item and not self.sent_text_done:
            self.sent_text_done = True
            text = "".join(self.text_buf)
            evts.append({
                "type": "response.output_text.done",
                "item_id": self.msg_item_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
            })
            evts.append({
                "type": "response.content_part.done",
                "item_id": self.msg_item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            })
            evts.append({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": self.msg_item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                },
            })
        # Close tool call items.
        for index in sorted(self.tool_items):
            item_id = self.tool_items[index]
            name, call_id, args = self.tool_buf[index]
            evts.append({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "arguments": args,
                },
            })
        status = _FINISH_TO_RESPONSES_STATUS.get((finish_reason or "STOP").upper(), "completed")
        final_resp = self._base_response(status)
        final_resp["usage"] = usage or None
        # Rebuild output[] from what we emitted.
        final_resp["output"] = []
        if self.sent_msg_item:
            final_resp["output"].append({
                "id": self.msg_item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "".join(self.text_buf), "annotations": []}],
            })
        for index in sorted(self.tool_items):
            name, call_id, args = self.tool_buf[index]
            final_resp["output"].append({
                "id": self.tool_items[index],
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": args,
            })
        evts.append({"type": "response.completed", "response": final_resp})
        self.finished = True
        return evts

def _extract_parts(parts: list) -> tuple[str, list, str]:
    """Extract (text, tool_calls, finish_reason_hint) from a Gemini parts list.

    tool_calls is a list of OpenAI-format tool_call dicts. The tool_call.id
    encodes the Gemini functionCall.id and thoughtSignature so they can be
    round-tripped back when the assistant message is sent in a later request.
    Format: call_<fc_id>|<thought_signature>
    """
    text_chunks = []
    tool_calls = []
    finish_hint = None
    thought_chunks = []

    for part in parts or []:
        if not isinstance(part, dict):
            continue
        if "text" in part and part["text"]:
            text_chunks.append(part["text"]) and not part.get("thought")
        if "functionCall" in part:
            fc = part["functionCall"] or {}
            name = fc.get("name", "")
            args = fc.get("args", {})
            fc_id = fc.get("id", "")
            # thoughtSignature is a sibling of functionCall in the part, not inside it.
            thought_sig = part.get("thoughtSignature") or part.get("thought_signature") or ""
            if name:
                # Encode the Gemini fc_id and thoughtSignature into the OpenAI tool_call.id
                # so they survive the round-trip through Hermes's message history.
                # Format: call_<fc_id>|<thought_signature>
                encoded_id = f"call_{fc_id}|{thought_sig}" if (fc_id or thought_sig) else ""
                tool_calls.append({
                    "id": encoded_id,  # may be empty; caller assigns if so
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args) if isinstance(args, (dict, list)) else str(args),
                    },
                })
        if "executableCode" in part:
            ec = part["executableCode"] or {}
            code = ec.get("code", "")
            if code:
                text_chunks.append(f"```python\n{code}\n```")
        if "codeExecutionResult" in part:
            cer = part["codeExecutionResult"] or {}
            out = cer.get("output", "")
            if out:
                text_chunks.append(f"```\n{out}\n```")
        if "thought" in part and part["thought"]:
            # Thinking summary part ({"thought": true, "text": ...}).
            # Keep it separate so callers can surface it as
            # reasoning_content.
            thought_text = part.get("text") or ""
            if thought_text:
                thought_chunks.append(thought_text)

    if tool_calls:
        finish_hint = "tool_calls"
    return "".join(text_chunks), tool_calls, finish_hint, "\n".join(thought_chunks)
def _finish_reason_from_gemini(candidate: dict, tool_calls: list) -> str:
    """Map Gemini finishReason / stopReason to OpenAI finish_reason."""
    if tool_calls:
        # If there are tool calls, OpenAI expects "tool_calls".
        fr = (candidate.get("finishReason") or "").upper()
        # Gemini may say STOP even when emitting function calls.
        return "tool_calls"
    fr = (candidate.get("finishReason") or candidate.get("stopReason") or "").upper()
    mapping = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "BLOCKLIST": "content_filter",
        "PROHIBITED_CONTENT": "content_filter",
        "SPII": "content_filter",
        "MALFORMED_FUNCTION_CALL": "stop",
        "IMAGE_SAFETY": "content_filter",
        "LANGUAGE": "content_filter",
        "OTHER": "stop",
    }
    return mapping.get(fr, "stop")

def _extract_usage(response: dict) -> dict:
    """Extract OpenAI-format usage from a Gemini response."""
    usage = response.get("usageMetadata") or {}
    if not usage:
        cands = response.get("candidates") or []
        if cands:
            usage = cands[0].get("usageMetadata") or {}
    pt = usage.get("promptTokenCount", 0) or 0
    ct = usage.get("candidatesTokenCount", 0) or usage.get("completionTokenCount", 0) or 0
    tt = usage.get("totalTokenCount", (pt + ct)) or (pt + ct)
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
    }

def gemini_response_to_openai(response: dict, model: str) -> dict:
    """Transform an unwrapped Gemini generateContent response to OpenAI
    chat.completion format.
    """
    candidates = response.get("candidates") or []
    text = ""
    tool_calls = []
    finish_reason = "stop"
    reasoning = ""

    if candidates:
        cand = candidates[0]
        content = cand.get("content") or {}
        parts = content.get("parts") or []
        text, tool_calls, _, reasoning = _extract_parts(parts)
        finish_reason = _finish_reason_from_gemini(cand, tool_calls)
        # Assign tool call ids.
        for i, tc in enumerate(tool_calls):
            if not tc["id"]:
                tc["id"] = f"call_{uuid.uuid4().hex[:24]}"
    else:
        # No candidates — possibly a promptFeedback block.
        pf = response.get("promptFeedback") or {}
        br = (pf.get("blockReason") or "").upper()
        if br:
            finish_reason = "content_filter"
            text = f"[Content blocked: {br}]"

    message = {"role": "assistant", "content": text if text else None}
    if reasoning:
        # Surface the thinking chain the way OpenAI-compatible clients expect
        # (deepseek-style reasoning_content field).
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": _extract_usage(response),
    }

def _extract_delta_from_parts(parts: list, prev_tool_names: set) -> tuple[str, list, list]:
    """Extract incremental (text, new_tool_calls, all_tool_names) from parts.

    For streaming, each chunk may contain partial text and/or function calls.
    We return text deltas and tool_call entries (with indices).
    """
    text_chunks = []
    tool_call_deltas = []
    thought_chunks = []
    current_names = set(prev_tool_names)

    for part in parts or []:
        if not isinstance(part, dict):
            continue
        if "text" in part and part["text"] and not part.get("thought"):
            text_chunks.append(part["text"])
        if "thought" in part and part["thought"]:
            t = part.get("text") or ""
            if t:
                thought_chunks.append(t)
        if "functionCall" in part:
            fc = part["functionCall"] or {}
            name = fc.get("name", "")
            args = fc.get("args", {})
            fc_id = fc.get("id", "")
            thought_sig = part.get("thoughtSignature") or part.get("thought_signature") or ""
            if name:
                current_names.add(name)
                encoded_id = f"call_{fc_id}|{thought_sig}" if (fc_id or thought_sig) else ""
                tool_call_deltas.append({
                    "index": len(current_names) - 1,
                    "id": encoded_id or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args) if isinstance(args, (dict, list)) else str(args),
                    },
                })
    return "".join(text_chunks), tool_call_deltas, current_names, "\n".join(thought_chunks)
