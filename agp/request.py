"""AGP module: request (split from antigravity_proxy.py)."""

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

from agp.auth import get_project_id
from agp.constants import DEFAULT_SYSTEM_INSTRUCTION, MODEL_MAP
from agp.content import _content_to_parts, _content_to_text, _normalize_responses_content
from agp.schema import _sanitize_gemini_schema

def _openai_tools_to_gemini(tools: list) -> list:
    """Convert OpenAI tools array to Gemini functionDeclarations format.

    OpenAI:  [{"type":"function","function":{"name","description","parameters"}}]
    Gemini:  [{"functionDeclarations":[{"name","description","parameters"}]}]
    """
    if not tools:
        return []
    decls = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not name:
            continue
        decl = {
            "name": name,
            "description": func.get("description", ""),
        }
        params = func.get("parameters")
        if isinstance(params, dict) and params:
            decl["parameters"] = _sanitize_gemini_schema(params)
        decls.append(decl)
    if not decls:
        return []
    return [{"functionDeclarations": decls}]

def _responses_input_to_messages(body: dict) -> list:
    """Responses API ``input`` + ``instructions`` -> chat messages.

    Handles string input, message items (developer/system/user/assistant),
    function_call items and function_call_output items. Consecutive
    function_call items are merged into one assistant message so the Gemini
    tool round-trip stays consistent.
    """
    messages = []
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": _normalize_responses_content(instructions)})

    inp = body.get("input", "")
    tool_names_by_call_id = {}
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t in ("function_call_output", "custom_tool_call_output", "tool_result",
                     "web_search_call", "computer_call_output"):
                cid = item.get("call_id")
                if not cid:
                    continue
                out = item.get("output", "")
                if isinstance(out, list):
                    out = _normalize_responses_content(out)
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": str(cid),
                    "content": "" if out is None else str(out),
                }
                # Gemini requires functionResponse.name to match the original
                # functionCall.name; the Responses function_call_output item only
                # carries call_id, so recover the name from a preceding
                # function_call item in the same input.
                fn_name = tool_names_by_call_id.get(str(cid))
                if fn_name:
                    tool_msg["name"] = fn_name
                messages.append(tool_msg)
            elif t in ("function_call", "custom_tool_call"):
                cid = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                fn_name = str(item.get("name") or "")
                tool_names_by_call_id[str(cid)] = fn_name
                args = item.get("arguments") or {}
                if not isinstance(args, str):
                    args = json.dumps(args)
                new_msg = {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": str(cid),
                        "type": "function",
                        "function": {
                            "name": str(item.get("name") or ""),
                            "arguments": args,
                        },
                    }],
                }
                # Merge consecutive function_call items into the previous assistant message.
                if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
                    messages[-1]["tool_calls"].extend(new_msg["tool_calls"])
                else:
                    messages.append(new_msg)
            else:
                # message item
                role = item.get("role") or "user"
                if role == "developer":
                    role = "system"
                content = item.get("content")
                if content is None:
                    continue
                messages.append({"role": role, "content": _normalize_responses_content(content)})
    return messages

def _responses_tools_to_chat_tools(tools) -> list:
    """Responses tools -> chat tools. Only ``function`` tools are kept;
    server-hosted built-ins (web_search, code_interpreter, computer_use,
    file_search, ...) have no Gemini equivalent and are dropped."""
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        params = tool.get("parameters") or {}
        if not isinstance(params, dict) or "type" not in params:
            params = {"type": "object"}
        fn = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": params,
        }
        if tool.get("strict") is not None:
            fn["strict"] = tool["strict"]
        out.append({"type": "function", "function": fn})
    return out

def _tool_choice_to_gemini(tool_choice) -> str | None:
    """Convert OpenAI tool_choice to Gemini toolConfig.functionCallingConfig.mode.

    Returns None if no transformation applies.
    """
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return "AUTO"
    if tool_choice == "none":
        return "NONE"
    if tool_choice == "required":
        return "ANY"
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        # Specific function forced — Gemini uses ANY + allowed_function_names.
        return "ANY"
    return None

def _build_passthrough_envelope(model: str, gemini_body: dict) -> dict:
    """Wrap a native Gemini generateContent request in the Antigravity
    envelope.  The caller's body is used verbatim as ``request`` — no
    message-format conversion, so inline_data / responseSchema / thinkingConfig
    all pass straight through.

    The friendly model name (e.g. ``gemini-3.5-flash``) is translated to the
    Antigravity backend name (e.g. ``gemini-3.5-flash-low``) via MODEL_MAP so
    callers can use the same name as Google's public API.

    Safety settings are injected only when the caller did not supply its own.
    """
    # Translate friendly model name to Antigravity backend name.
    entry = MODEL_MAP.get(model)
    backend_model = entry[0] if entry else model
    thinking_level = entry[1] if entry else None

    inner = dict(gemini_body)  # shallow copy — don't mutate caller's dict

    # Apply thinking level from MODEL_MAP if the caller didn't set one.
    if thinking_level:
        gc = inner.setdefault("generationConfig", {})
        tc = gc.setdefault("thinkingConfig", {})
        tc.setdefault("thinkingLevel", thinking_level)

    if "safetySettings" not in inner:
        inner["safetySettings"] = [
            {"category": cat, "threshold": "BLOCK_NONE"}
            for cat in (
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            )
        ]

    project_id = get_project_id()
    return {
        "project": project_id,
        "model": backend_model,
        "request": inner,
        "requestType": "agent",
        "userAgent": "antigravity",
        "requestId": f"agent-{uuid.uuid4().hex}",
    }

def _build_gemini_request(body: dict, backend_model: str, thinking_level: str = None):
    """Transform an OpenAI chat completion body into a Gemini generateContent
    request envelope.

    Returns (envelope_dict, openai_model_name).
    """
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        messages = []

    system_texts = []
    contents = []
    # Registry: OpenAI tool_call_id -> real function name, populated from
    # assistant tool_calls so Gemini functionResponse.name can match the
    # original functionCall.name (Gemini requires the match; OpenAI tool
    # messages carry no name field).
    name_by_call_id = {}

    # Collect system instruction pieces and convert messages.
    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            text = _content_to_text(msg.get("content"))
            if text:
                system_texts.append(text)
            continue

        if role == "tool":
            # OpenAI tool result message -> Gemini functionResponse part.
            tool_call_id = msg.get("tool_call_id", "")
            content_text = _content_to_text(msg.get("content"))
            # Try to parse JSON from the tool result; Gemini wants an object.
            response_obj = {}
            if content_text:
                try:
                    parsed = json.loads(content_text)
                    response_obj = parsed if isinstance(parsed, dict) else {"result": parsed}
                except (json.JSONDecodeError, TypeError):
                    response_obj = {"result": content_text}
            # Derive a function name: prefer the message name, else the
            # registry populated from prior assistant tool_calls (OpenAI tool
            # messages omit name), else tool_call_id as last resort.
            fn_name = (msg.get("name") or name_by_call_id.get(tool_call_id)
                       or tool_call_id or "tool_result")
            # Decode the fc_id from the encoded tool_call_id (format: call_<fc_id>|<sig>).
            fc_response_id = ""
            if tool_call_id.startswith("call_"):
                raw = tool_call_id[5:]  # strip "call_"
                if "|" in raw:
                    fc_response_id = raw.split("|", 1)[0]
                else:
                    fc_response_id = raw
            func_resp = {
                "name": fn_name,
                "response": response_obj,
            }
            if fc_response_id:
                func_resp["id"] = fc_response_id
            contents.append({
                "role": "user",
                "parts": [{"functionResponse": func_resp}],
            })
            continue

        if role == "assistant":
            parts = []
            text = _content_to_text(msg.get("content"))
            if text:
                parts.append({"text": text})
            # Prior tool calls from the assistant -> functionCall parts.
            # Decode the encoded tool_call.id to recover the Gemini fc_id and thoughtSignature.
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                fn_name = fn.get("name", "")
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": str(args_str)}
                if fn_name:
                    tc_id = tc.get("id", "") or ""
                    if tc_id:
                        name_by_call_id[tc_id] = fn_name
                    fc_id = ""
                    thought_sig = ""
                    # Decode: format is call_<fc_id>|<thought_signature>
                    if tc_id.startswith("call_"):
                        raw = tc_id[5:]
                        if "|" in raw:
                            fc_id, thought_sig = raw.split("|", 1)
                        else:
                            fc_id = raw
                    fc_obj = {"name": fn_name, "args": args}
                    if fc_id:
                        fc_obj["id"] = fc_id
                    part_obj = {"functionCall": fc_obj}
                    # thoughtSignature goes as a sibling key on the part, not inside functionCall.
                    # If missing, use the sentinel to skip validation (officially supported by Google).
                    if thought_sig:
                        part_obj["thoughtSignature"] = thought_sig
                    else:
                        part_obj["thoughtSignature"] = "skip_thought_signature_validator"
                    parts.append(part_obj)
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        if role == "developer":
            # Treat developer role like system.
            text = _content_to_text(msg.get("content"))
            if text:
                system_texts.append(text)
            continue

        # Default: user role. Build parts from content blocks so image_url
        # data URLs become real inlineData parts instead of text markers.
        parts = _content_to_parts(msg.get("content"))
        # Some clients send tool_calls on user messages; handle gracefully.
        for tc in (msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            fn_name = fn.get("name", "")
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except (json.JSONDecodeError, TypeError):
                args = {"raw": str(args_str)}
            if fn_name:
                parts.append({"functionCall": {"name": fn_name, "args": args}})
        if parts:
            contents.append({"role": "user", "parts": parts})

    # Build system instruction: use user's system messages if provided,
    # otherwise fall back to the default.
    if system_texts:
        system_instruction = {"parts": [{"text": "\n\n".join(s for s in system_texts if s)}]}
    else:
        system_instruction = {"parts": [{"text": DEFAULT_SYSTEM_INSTRUCTION}]}

    # Build the inner Gemini request.
    inner = {
        "contents": contents,
        "systemInstruction": system_instruction,
    }

    # Tools.
    tools = _openai_tools_to_gemini(body.get("tools"))
    if tools:
        inner["tools"] = tools

    # Tool choice -> toolConfig.
    tool_choice = body.get("tool_choice")
    mode = _tool_choice_to_gemini(tool_choice)
    if mode:
        tc_config = {"functionCallingConfig": {"mode": mode}}
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fn_name = (tool_choice.get("function") or {}).get("name")
            if fn_name:
                tc_config["functionCallingConfig"]["allowedFunctionNames"] = [fn_name]
        inner["toolConfig"] = tc_config

    # Generation config.
    gen_config = {}
    if "temperature" in body and body["temperature"] is not None:
        gen_config["temperature"] = float(body["temperature"])
    if "top_p" in body and body["top_p"] is not None:
        gen_config["topP"] = float(body["top_p"])
    # max_tokens semantics: OpenAI counts OUTPUT tokens only, but Gemini's
    # maxOutputTokens includes thinking tokens (3.7-flash-high spends most of
    # a small budget on reasoning, starving the visible output). Compensate by
    # reserving a thinking budget and adding it to the output cap so the
    # client's requested output length is honored. If the sandbox rejects
    # thinkingBudget, we fall back to a 2x multiplier below.
    if "max_tokens" in body and body["max_tokens"] is not None:
        _client_max = int(body["max_tokens"])
        _think_budget = min(max(_client_max, 256), 8192)
        gen_config["maxOutputTokens"] = _client_max + _think_budget
        gen_config.setdefault("thinkingConfig", {})
        gen_config["thinkingConfig"].setdefault("thinkingBudget", _think_budget)
        gen_config["thinkingConfig"].setdefault("includeThoughts", False)
    elif "max_completion_tokens" in body and body["max_completion_tokens"] is not None:
        _client_max = int(body["max_completion_tokens"])
        _think_budget = min(max(_client_max, 256), 8192)
        gen_config["maxOutputTokens"] = _client_max + _think_budget
        gen_config.setdefault("thinkingConfig", {})
        gen_config["thinkingConfig"].setdefault("thinkingBudget", _think_budget)
        gen_config["thinkingConfig"].setdefault("includeThoughts", False)
    # NOTE: frequency_penalty / presence_penalty are intentionally dropped.
    # Gemini's sandbox backend rejects them with 400 "Penalty is not enabled
    # for this model"; OpenAI clients send them by default on some paths.
    if "stop" in body and body["stop"]:
        stops = body["stop"]
        if isinstance(stops, str):
            stops = [stops]
        gen_config["stopSequences"] = stops
    if "seed" in body and body["seed"] is not None:
        gen_config["seed"] = int(body["seed"])
    # Enable thinking for the thinking variant of claude-opus.
    if backend_model == "claude-opus-4-6-thinking":
        thinking_cfg = gen_config.get("thinkingConfig", {})
        thinking_cfg.setdefault("includeThoughts", False)
        gen_config["thinkingConfig"] = thinking_cfg
    # OpenAI response_format -> Gemini responseMimeType / responseSchema.
    rf = body.get("response_format") or {}
    if isinstance(rf, dict):
        rft = rf.get("type")
        if rft == "json_object":
            gen_config["responseMimeType"] = "application/json"
        elif rft == "json_schema":
            sch = rf.get("json_schema") or {}
            # OpenAI wraps as {name, strict, schema:{...}}; Gemini wants the
            # schema object directly as responseSchema.
            if isinstance(sch, dict) and isinstance(sch.get("schema"), dict):
                sch = sch["schema"]
            if isinstance(sch, dict):
                gen_config["responseMimeType"] = "application/json"
                gen_config["responseSchema"] = _sanitize_gemini_schema(sch)
    # Inject thinking level for Gemini 3 Pro models.
    if thinking_level:
        thinking_cfg = gen_config.get("thinkingConfig", {})
        thinking_cfg.setdefault("thinkingLevel", thinking_level)
        thinking_cfg.setdefault("includeThoughts", False)
        gen_config["thinkingConfig"] = thinking_cfg
    if gen_config:
        inner["generationConfig"] = gen_config

    # Safety settings: relax standard categories so coding tasks aren't blocked.
    # NOTE: HARM_CATEGORY_CIVIC_INTEGRITY is rejected by this API variant even
    # though it appears in the error's valid-list — only the 4 below are accepted.
    inner["safetySettings"] = [
        {"category": cat, "threshold": "BLOCK_NONE"}
        for cat in (
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        )
    ]

    # Build the wrapped envelope.
    project_id = get_project_id()
    envelope = {
        "project": project_id,
        "model": backend_model,
        "request": inner,
        "requestType": "agent",
        "userAgent": "antigravity",
        "requestId": f"agent-{uuid.uuid4().hex}",
    }
    return envelope
