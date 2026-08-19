#!/usr/bin/env python3
"""Antigravity OpenAI-compatible proxy - entry point.

Split package layout: agp/* (auth, schema, content, request,
response, upstream, errors, constants). This file keeps the
HTTP handler, routes, and main().
"""

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

from agp.auth import _read_token_from_disk, get_project_id
from agp.constants import DEFAULT_MODEL, TOKEN_FILE, _GEMINI_PASSTHROUGH_RE, _GEMINI_STREAM_PASSTHROUGH_RE
from agp.models import get_merged_model_map
from agp.errors import _openai_error, _stream_error_payload
from agp.log import _log
from agp.request import _build_gemini_request, _build_passthrough_envelope, _responses_input_to_messages, _responses_tools_to_chat_tools
from agp.response import _ResponsesStreamState, _extract_delta_from_parts, _gemini_to_responses_output, _gemini_usage_to_responses_usage, gemini_response_to_openai
from agp.upstream import UpstreamError, call_generate_content, stream_generate_content

from agp.auth import _deep_find, _load_code_assist, _parse_expiry, _read_token_from_disk, _refresh_access_token, _write_token_to_disk, get_access_token, get_project_id
from agp.content import _content_to_parts, _content_to_text, _normalize_responses_content
from agp.errors import _openai_error, _stream_error_payload
from agp.log import _log
from agp.request import _build_gemini_request, _build_passthrough_envelope, _openai_tools_to_gemini, _responses_input_to_messages, _responses_tools_to_chat_tools, _tool_choice_to_gemini
from agp.response import _ResponsesStreamState, _extract_delta_from_parts, _extract_parts, _extract_usage, _finish_reason_from_gemini, _gemini_to_responses_output, _gemini_usage_to_responses_usage, gemini_response_to_openai
from agp.schema import _sanitize_gemini_schema, _schema_constraint_suffix
from agp.upstream import UpstreamError, _is_location_error, _make_upstream_request, _parse_remaining, _parse_sse_block, _stream_once, _try_parse_json_element_in_array, _try_parse_one_event, _try_parse_one_json_object, call_generate_content, stream_generate_content


def main():
    parser = argparse.ArgumentParser(
        description="Antigravity OpenAI-compatible proxy (Cloud Code Assist API)"
    )
    parser.add_argument("--port", type=int, default=8877, help="Port to listen on (default 8877)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default 127.0.0.1)")
    args = parser.parse_args()

    if not _preflight_check():
        sys.exit(1)

    # Pre-discover the project ID so the first request is fast.
    try:
        get_project_id()
    except Exception as e:
        _log(f"WARNING: Could not pre-discover project ID: {e}")
        _log("It will be discovered on the first request.")

    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    server.daemon_threads = True
    _log(f"Antigravity proxy listening on http://{args.host}:{args.port}/v1")
    try:
        merged, added, err = get_merged_model_map()
        _log("  Models: {} ({} discovered from upstream){}".format(len(merged), added, "" if not err else ", discovery failed: "+str(err)))
        _log("  Model list: " + ", ".join(sorted(merged.keys())))
    except Exception as e:
        _log("  Models: {} (static; discovery error: {})".format(len(MODEL_MAP), e))
    _log(f"  Token:  {TOKEN_FILE}")
    _log(f"  Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("\nShutting down...")
        server.shutdown()

def _preflight_check():
    """Validate the token file exists and is readable before starting."""
    if not os.path.isfile(TOKEN_FILE):
        _log(f"ERROR: Token file not found at {TOKEN_FILE}")
        _log("Run the Antigravity CLI (`agy`) to authenticate first.")
        return False
    data = _read_token_from_disk()
    if not data:
        _log("ERROR: Token file exists but could not be parsed.")
        return False
    token_obj = data.get("token") or {}
    if not token_obj.get("refresh_token"):
        _log("ERROR: No refresh_token in token file. Re-run `agy` to authenticate.")
        return False
    _log(f"Token file OK (auth_method={data.get('auth_method', 'unknown')})")
    return True

class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AntigravityProxy/2.0"

    def _check_auth(self) -> bool:
        expected = os.environ.get("AG_TOKEN", "").strip()
        if not expected:
            return True
        hdr = self.headers.get("Authorization", "")
        xgk = self.headers.get("X-Goog-Api-Key", "")
        if hdr == f"Bearer {expected}" or xgk == expected:
            return True
        self._send_json(401, _openai_error("Invalid API key", err_type="authentication_error", code=401))
        return False

    def log_message(self, fmt, *args):
        # We do our own logging.
        pass

    def _send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error_json(self, code, message, err_type="api_error", http_code=None):
        self._send_json(http_code or code, _openai_error(message, err_type, code=code))

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None  # signal parse failure

    # ── GET routes ────────────────────────────────────────────────────────────

    def do_GET(self):
        if not self._check_auth():
            return
        t0 = time.time()
        path = urlparse(self.path).path
        try:
            if path == "/v1/models" or path == "/models":
                self._handle_models()
                self._log_req("GET", path, 200, time.time() - t0)
            elif path in ("/health", "/"):
                self._send_json(200, {"status": "ok", "service": "antigravity-proxy"})
                self._log_req("GET", path, 200, time.time() - t0)
            else:
                self._send_error_json("not_found", "Not found", err_type="invalid_request_error", http_code=404)
                self._log_req("GET", path, 404, time.time() - t0)
        except Exception as e:
            _log(f"GET {path} error: {e}\n{traceback.format_exc()}")
            try:
                self._send_error_json("internal_error", str(e), http_code=500)
            except Exception:
                pass
            self._log_req("GET", path, 500, time.time() - t0)

    def _handle_models(self):
        now = int(time.time())
        merged, _, _ = get_merged_model_map()
        models = []
        for name in merged:
            models.append({
                "id": name,
                "object": "model",
                "created": now,
                "owned_by": "google",
            })
        self._send_json(200, {"object": "list", "data": models})

    # ── POST routes ───────────────────────────────────────────────────────────

    def do_POST(self):
        if not self._check_auth():
            return
        t0 = time.time()
        path = urlparse(self.path).path
        try:
            if path in ("/v1/chat/completions", "/chat/completions"):
                self._handle_chat()
                self._log_req("POST", path, 200, time.time() - t0)
            elif path in ("/v1/responses", "/responses"):
                self._handle_responses()
                self._log_req("POST", path, 200, time.time() - t0)
            elif _GEMINI_PASSTHROUGH_RE.match(path):
                self._handle_gemini_passthrough(path)
                self._log_req("POST", path, 200, time.time() - t0)
            elif _GEMINI_STREAM_PASSTHROUGH_RE.match(path):
                self._handle_gemini_stream_passthrough(path)
                self._log_req("POST", path, 200, time.time() - t0)
            else:
                self._send_error_json("not_found", "Not found", err_type="invalid_request_error", http_code=404)
                self._log_req("POST", path, 404, time.time() - t0)
        except Exception as e:
            _log(f"POST {path} error: {e}\n{traceback.format_exc()}")
            try:
                self._send_error_json("internal_error", str(e), http_code=500)
            except Exception:
                pass
            self._log_req("POST", path, 500, time.time() - t0)

    def _log_req(self, method, path, status, duration):
        _log(f"{method} {path} {status} {duration:.3f}s")

    # ── Chat completions ──────────────────────────────────────────────────────

    def _handle_gemini_passthrough(self, path: str):
        """Handle a native Gemini generateContent request, wrapping it in the
        Antigravity envelope and returning the raw Gemini response (unwrapped).
        This is used by invoice_reader.py to send inline_data + responseSchema
        requests through the OAuth-funded proxy instead of a paid API key."""
        body = self._read_body()
        if body is None or not isinstance(body, dict):
            self._send_error_json("invalid_request", "Invalid JSON body", err_type="invalid_request_error", http_code=400)
            return

        model = _GEMINI_PASSTHROUGH_RE.match(path).group(1)

        try:
            envelope = _build_passthrough_envelope(model, body)
        except UpstreamError as e:
            self._send_error_json("upstream_error", str(e), err_type="api_error", http_code=e.status)
            return
        except RuntimeError as e:
            self._send_error_json("auth_error", str(e), err_type="authentication_error", http_code=401)
            return
        except Exception as e:
            self._send_error_json("request_error", f"Failed to build request: {e}", err_type="invalid_request_error", http_code=400)
            return

        try:
            gemini_resp = call_generate_content(envelope)
        except UpstreamError as e:
            self._send_error_json("upstream_error", str(e), err_type="api_error", http_code=e.status)
            return
        except Exception as e:
            self._send_error_json("upstream_error", f"Upstream call failed: {e}", err_type="api_error", http_code=502)
            return

        self._send_json(200, gemini_resp)

    def _handle_gemini_stream_passthrough(self, path: str):
        """Handle a native Gemini streamGenerateContent request.

        Same passthrough envelope as the non-streaming variant; the Antigravity
        SSE stream is re-serialized back to the caller as SSE events. This lets
        Gemini-native SDKs/clients (e.g. omp's google-generative-ai protocol)
        stream through the OAuth-funded proxy.
        """
        body = self._read_body()
        if body is None or not isinstance(body, dict):
            self._send_error_json("invalid_request", "Invalid JSON body", err_type="invalid_request_error", http_code=400)
            return

        model = _GEMINI_STREAM_PASSTHROUGH_RE.match(path).group(1)

        try:
            envelope = _build_passthrough_envelope(model, body)
        except UpstreamError as e:
            self._send_error_json("upstream_error", str(e), err_type="api_error", http_code=e.status)
            return
        except RuntimeError as e:
            self._send_error_json("auth_error", str(e), err_type="authentication_error", http_code=401)
            return
        except Exception as e:
            self._send_error_json("request_error", f"Failed to build request: {e}", err_type="invalid_request_error", http_code=400)
            return

        # Send SSE headers.
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
        except Exception:
            return

        try:
            for event in stream_generate_content(envelope):
                if event is None:
                    continue
                try:
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        except UpstreamError as e:
            _log(f"gemini stream passthrough upstream error: {e.status} {e.body}")
            try:
                err = {"error": {"code": e.status, "message": e.body or str(e), "status": "UPSTREAM_ERROR"}}
                self.wfile.write(f"data: {json.dumps(err)}\n\n".encode())
                self.wfile.flush()
            except Exception:
                pass
        except Exception as e:
            _log(f"gemini stream passthrough error: {e}\n{traceback.format_exc()}")

    # ── Responses API (/v1/responses) ─────────────────────────────────────────

    def _handle_responses(self):
        body = self._read_body()
        if body is None:
            self._send_error_json("invalid_request", "Invalid JSON body", err_type="invalid_request_error", http_code=400)
            return
        if not isinstance(body, dict):
            self._send_error_json("invalid_request", "Request body must be a JSON object", err_type="invalid_request_error", http_code=400)
            return

        openai_model = body.get("model") or DEFAULT_MODEL
        stream = bool(body.get("stream", False))

        merged, _, _ = get_merged_model_map()

        model_entry = merged.get(openai_model)
        if model_entry:
            backend_model, thinking_level = model_entry
        else:
            backend_model = openai_model
            thinking_level = None
        # NOTE: `reasoning.effort` is intentionally ignored. Mapping it to the
        # "-low/-medium/-high" backend tier suffix produces 404s because those
        # tier models do not exist on the Antigravity backend.

        messages = _responses_input_to_messages(body)
        tools = _responses_tools_to_chat_tools(body.get("tools"))
        chat_body = {
            "model": openai_model,
            "messages": messages,
            "tools": tools,
            "stream": stream,
        }
        if body.get("max_output_tokens") is not None:
            chat_body["max_tokens"] = body["max_output_tokens"]
        if body.get("temperature") is not None:
            chat_body["temperature"] = body["temperature"]
        if body.get("top_p") is not None:
            chat_body["top_p"] = body["top_p"]

        try:
            envelope = _build_gemini_request(chat_body, backend_model, thinking_level)
        except Exception as e:
            self._send_error_json("request_error", f"Failed to build request: {e}", err_type="invalid_request_error", http_code=400)
            return

        if stream:
            self._handle_responses_stream(envelope, openai_model)
        else:
            self._handle_responses_nonstream(envelope, openai_model)

    def _handle_responses_nonstream(self, envelope, openai_model):
        try:
            gemini_resp = call_generate_content(envelope)
        except UpstreamError as e:
            self._send_error_json("upstream_error", str(e), err_type="api_error", http_code=e.status)
            return
        except Exception as e:
            self._send_error_json("upstream_error", f"Upstream call failed: {e}", err_type="api_error", http_code=502)
            return

        try:
            output, status = _gemini_to_responses_output(gemini_resp, openai_model)
        except Exception as e:
            _log(f"Responses transform error: {e}\n{traceback.format_exc()}")
            output, status = [], "incomplete"

        resp = {
            "id": f"resp_{uuid.uuid4().hex[:24]}",
            "object": "response",
            "created_at": int(time.time()),
            "status": status,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
            "model": openai_model,
            "output": output,
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": None,
            "temperature": None,
            "text": {},
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "truncation": None,
            "usage": _gemini_usage_to_responses_usage(gemini_resp),
            "user": None,
            "max_output_tokens": None,
        }
        self._send_json(200, resp)

    def _handle_responses_stream(self, envelope, openai_model):
        resp_id = f"resp_{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
        except Exception:
            return

        def send_event(evt):
            try:
                self.wfile.write(f"data: {json.dumps(evt)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return False
            return True

        st = _ResponsesStreamState(resp_id, created, openai_model)

        try:
            gen = stream_generate_content(envelope)
        except UpstreamError as e:
            # Streaming rejected (e.g. 400 on tool round-trips) — fall back to
            # non-streaming, mirroring the chat path.
            if e.status == 400:
                try:
                    resp = call_generate_content(envelope)
                    output, status = _gemini_to_responses_output(resp, openai_model)
                    usage = _gemini_usage_to_responses_usage(resp)
                    for evt in st.initial_events():
                        if not send_event(evt):
                            return
                    for item in output:
                        itype = item.get("type")
                        if itype == "message":
                            for evt in st.on_text_delta(item["content"][0]["text"]):
                                if not send_event(evt):
                                    return
                        elif itype == "function_call":
                            for evt in st.on_tool_call(0, item.get("call_id", ""), item.get("name", ""), item.get("arguments", "")):
                                if not send_event(evt):
                                    return
                    for evt in st.finish("STOP" if status == "completed" else "MAX_TOKENS", usage):
                        if not send_event(evt):
                            return
                    try:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except Exception:
                        pass
                    return
                except Exception:
                    pass
            send_event({"type": "error", "code": "upstream_error", "message": str(e)})
            return
        except Exception as e:
            send_event({"type": "error", "code": "upstream_error", "message": str(e)})
            return

        seen_tool_names = set()
        usage = None
        finish_reason = "STOP"
        try:
            for event in gen:
                if not isinstance(event, dict):
                    continue
                resp = event.get("response", event)
                um = resp.get("usageMetadata")
                if um:
                    usage = {
                        "input_tokens": um.get("promptTokenCount", 0),
                        "output_tokens": um.get("candidatesTokenCount", 0),
                        "total_tokens": um.get("totalTokenCount", 0),
                    }
                cands = resp.get("candidates") or []
                if not cands:
                    continue
                cand = cands[0]
                parts = ((cand.get("content") or {}).get("parts")) or []

                text_delta, tool_call_deltas, seen_tool_names, thought_delta = _extract_delta_from_parts(
                    parts, seen_tool_names
                )
                fr = cand.get("finishReason")
                if fr:
                    finish_reason = str(fr).upper()

                if thought_delta:
                    for evt in st.on_thought_delta(thought_delta):
                        if not send_event(evt):
                            return
                if text_delta:
                    for evt in st.on_text_delta(text_delta):
                        if not send_event(evt):
                            return
                for tcd in tool_call_deltas:
                    fn = tcd.get("function") or {}
                    for evt in st.on_tool_call(
                        tcd.get("index", 0),
                        tcd.get("id", ""),
                        fn.get("name", ""),
                        fn.get("arguments", ""),
                    ):
                        if not send_event(evt):
                            return

            for evt in st.finish(finish_reason, usage):
                if not send_event(evt):
                    return
        except Exception as e:
            _log(f"responses stream error: {e}\n{traceback.format_exc()}")
            send_event({"type": "error", "code": "upstream_error", "message": str(e)})
            return
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception:
            pass

    def _handle_chat(self):
        body = self._read_body()
        if body is None:
            self._send_error_json("invalid_request", "Invalid JSON body", err_type="invalid_request_error", http_code=400)
            return
        if not isinstance(body, dict):
            self._send_error_json("invalid_request", "Request body must be a JSON object", err_type="invalid_request_error", http_code=400)
            return

        openai_model = body.get("model") or DEFAULT_MODEL
        stream = bool(body.get("stream", False))

        merged, _, _ = get_merged_model_map()

        # Map model. MODEL_MAP values are (backend_name, thinking_level) tuples.
        model_entry = merged.get(openai_model)
        if model_entry:
            backend_model, thinking_level = model_entry
        else:
            # Allow pass-through of unknown models.
            backend_model = openai_model
            thinking_level = None

        # OpenAI reasoning_effort passthrough -> daily backend tier suffix.
        effort = body.get("reasoning_effort")
        if effort:
            _e = str(effort).strip().lower()
            _tier = {"low": "low", "medium": "medium", "high": "high"}.get(_e)
            if _tier:
                _base = backend_model
                for _suf in ("-low", "-medium", "-high", "-tiered"):
                    if _base.endswith(_suf):
                        _base = _base[: -len(_suf)]
                        break
                if _base.endswith("-flash"):
                    backend_model = _base + "-" + _tier
                    thinking_level = _tier
        _log(f"chat model={openai_model} backend={backend_model} effort={body.get('reasoning_effort')} level={thinking_level}")

        # Build the Gemini request envelope.
        try:
            envelope = _build_gemini_request(body, backend_model, thinking_level)
        except UpstreamError as e:
            self._send_error_json("upstream_error", str(e), err_type="api_error", http_code=e.status)
            return
        except RuntimeError as e:
            self._send_error_json("auth_error", str(e), err_type="authentication_error", http_code=401)
            return
        except Exception as e:
            self._send_error_json("request_error", f"Failed to build request: {e}", err_type="invalid_request_error", http_code=400)
            return

        # Validate we have at least one content entry.
        inner = envelope.get("request", {})
        if not inner.get("contents"):
            self._send_error_json("invalid_request", "No messages provided", err_type="invalid_request_error", http_code=400)
            return

        if stream:
            self._handle_stream(envelope, openai_model)
        else:
            self._handle_nonstream(envelope, openai_model)

    def _handle_nonstream(self, envelope, openai_model):
        try:
            gemini_resp = call_generate_content(envelope)
        except UpstreamError as e:
            self._send_error_json("upstream_error", str(e), err_type="api_error", http_code=e.status)
            return
        except Exception as e:
            self._send_error_json("upstream_error", f"Upstream call failed: {e}", err_type="api_error", http_code=502)
            return

        try:
            openai_resp = gemini_response_to_openai(gemini_resp, openai_model)
        except Exception as e:
            _log(f"Response transform error: {e}\n{traceback.format_exc()}")
            # Fallback: return raw text if we can find any.
            try:
                text = (gemini_resp.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", ""))
            except Exception:
                text = ""
            openai_resp = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": openai_model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text or "[error: could not parse response]"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        self._send_json(200, openai_resp)

    def _handle_stream(self, envelope, openai_model):
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        # Send SSE headers.
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
        except Exception:
            return

        def send_sse(obj):
            data = f"data: {json.dumps(obj)}\n\n"
            try:
                self.wfile.write(data.encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return False
            return True

        # Initial role chunk.
        first_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": openai_model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }],
        }
        if not send_sse(first_chunk):
            return

        try:
            gen = stream_generate_content(envelope)
        except UpstreamError as e:
            # Streaming failed — fall back to non-streaming and return as a single chunk.
            # This handles the case where the streamGenerateContent endpoint rejects
            # tool-result round-trips with 400 "invalid argument".
            if e.status == 400:
                try:
                    resp = call_generate_content(envelope)
                    openai_resp = gemini_response_to_openai(resp, openai_model)
                    msg = openai_resp["choices"][0]["message"]
                    delta = {}
                    if msg.get("content"):
                        delta["content"] = msg["content"]
                    if msg.get("tool_calls"):
                        delta["tool_calls"] = msg["tool_calls"]
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": openai_model,
                        "choices": [{
                            "index": 0,
                            "delta": delta,
                            "finish_reason": openai_resp["choices"][0].get("finish_reason", "stop"),
                        }],
                    }
                    if openai_resp.get("usage"):
                        chunk["usage"] = openai_resp["usage"]
                    send_sse(chunk)
                    try:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except Exception:
                        pass
                    return
                except Exception:
                    pass  # fall through to error chunk
            send_sse(_stream_error_payload(str(e), err_type="upstream_error",
                                         code=e.status if hasattr(e, "status") else 500))
            send_sse({"__done__": True})  # sentinel handled below
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass
            return
        except Exception as e:
            _log(f"Streaming error: {e}")
            send_sse(_stream_error_payload(str(e), err_type="upstream_error"))
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass
            return

        finish_reason = "stop"
        seen_tool_names = set()
        total_usage = None
        events_processed = 0

        try:
            for event in gen:
                if not isinstance(event, dict):
                    continue
                if event.get("__done__"):
                    break

                events_processed += 1

                # Unwrap envelope if present.
                resp = event.get("response", event)

                # Check for usage metadata (often on the last chunk).
                usage_meta = resp.get("usageMetadata")
                if usage_meta:
                    total_usage = usage_meta

                candidates = resp.get("candidates") or []
                if not candidates:
                    continue

                cand = candidates[0]
                content = cand.get("content") or {}
                parts = content.get("parts") or []

                text_delta, tool_call_deltas, seen_tool_names, thought_delta = _extract_delta_from_parts(
                    parts, seen_tool_names
                )

                # Determine finish reason.
                fr = cand.get("finishReason")
                if fr:
                    if tool_call_deltas:
                        finish_reason = "tool_calls"
                    else:
                        fr_up = str(fr).upper()
                        mapping = {
                            "STOP": "stop",
                            "MAX_TOKENS": "length",
                            "SAFETY": "content_filter",
                            "RECITATION": "content_filter",
                            "BLOCKLIST": "content_filter",
                            "PROHIBITED_CONTENT": "content_filter",
                            "SPII": "content_filter",
                            "IMAGE_SAFETY": "content_filter",
                            "LANGUAGE": "content_filter",
                            "OTHER": "stop",
                        }
                        finish_reason = mapping.get(fr_up, "stop")

                # Build delta.
                delta = {}
                if text_delta:
                    delta["content"] = text_delta
                if thought_delta:
                    # Thinking chain -> deepseek-style reasoning_content so
                    # clients display it as a thinking block.
                    delta["reasoning_content"] = thought_delta
                if tool_call_deltas:
                    delta["tool_calls"] = tool_call_deltas

                if not delta and not fr:
                    # Empty chunk with nothing — skip.
                    continue

                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": openai_model,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }],
                }
                if not send_sse(chunk):
                    return

            # Final chunk with finish_reason.
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": openai_model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }],
            }
            # Include usage in the final chunk if the client requested
            # stream_options.include_usage (OpenAI convention).
            if total_usage:
                pt = total_usage.get("promptTokenCount", 0) or 0
                ct = total_usage.get("candidatesTokenCount", 0) or total_usage.get("completionTokenCount", 0) or 0
                tt = total_usage.get("totalTokenCount", (pt + ct)) or (pt + ct)
                final_chunk["usage"] = {
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": tt,
                }
            send_sse(final_chunk)

            # Termination sentinel.
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass

        except Exception as e:
            _log(f"Streaming error: {e}")
            # If the stream endpoint returned 400, try non-streaming as fallback.
            # The streamGenerateContent endpoint sometimes rejects tool-result round-trips.
            if "400" in str(e):
                try:
                    resp = call_generate_content(envelope)
                    openai_resp = gemini_response_to_openai(resp, openai_model)
                    msg = openai_resp["choices"][0]["message"]
                    delta = {}
                    if msg.get("content"):
                        delta["content"] = msg["content"]
                    if msg.get("tool_calls"):
                        delta["tool_calls"] = msg["tool_calls"]
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": openai_model,
                        "choices": [{
                            "index": 0,
                            "delta": delta,
                            "finish_reason": openai_resp["choices"][0].get("finish_reason", "stop"),
                        }],
                    }
                    if openai_resp.get("usage"):
                        chunk["usage"] = openai_resp["usage"]
                    send_sse(chunk)
                    try:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except Exception:
                        pass
                    return
                except Exception:
                    pass  # fall through to error chunk
            # Send a structured SSE error event, then close the stream.
            try:
                send_sse(_stream_error_payload(str(e), err_type="upstream_error",
                                               code=e.status if hasattr(e, "status") else 500))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass

if __name__ == "__main__":
    main()
