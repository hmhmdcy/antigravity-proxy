#!/usr/bin/env python3
"""Live integration tests against a running proxy.

Requires the proxy to be reachable and an API key. Skipped automatically
when PROXY_URL / PROXY_KEY env vars are absent, so the unit suite stays
green offline.

Run with proxy up:
  PROXY_URL=http://127.0.0.1:8877 PROXY_KEY=sk-... \
    python3 -m unittest tests.test_integration -v
"""
import json
import os
import unittest
import urllib.request
import urllib.error

PROXY_URL = os.environ.get("PROXY_URL", "http://127.0.0.1:8877")
PROXY_KEY = os.environ.get("PROXY_KEY", "")


def _post(path, body, timeout=40):
    req = urllib.request.Request(
        PROXY_URL + path,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {PROXY_KEY}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


class TestLiveProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PROXY_KEY:
            raise unittest.SkipTest("PROXY_KEY not set; skipping live tests")

    def test_chat_basic(self):
        status, raw = _post("/v1/chat/completions", {
            "model": "gemini-3.7-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        self.assertEqual(status, 200, raw[:300])
        data = json.loads(raw)
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")

    def test_tool_roundtrip(self):
        # Round 1: ask the model to call a tool; get the REAL encoded id
        # (call_<fc_id>|<thought_signature>). Fake signatures are rejected
        # upstream ("Corrupted thought signature").
        tools = [{"type": "function", "function": {
            "name": "glob", "description": "List files matching a glob pattern",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}},
                           "required": ["path"]}}}]
        status, raw = _post("/v1/chat/completions", {
            "model": "gemini-3.7-flash",
            "messages": [{"role": "user",
                          "content": "use the glob tool with path 'x' and say done"}],
            "tools": tools,
            "stream": False,
        })
        self.assertEqual(status, 200, raw[:300])
        data = json.loads(raw)
        msg = data["choices"][0]["message"]
        self.assertTrue(msg.get("tool_calls"), f"expected tool_calls, got: {raw[:200]}")
        tc = msg["tool_calls"][0]
        cid = tc["id"]
        self.assertIn("|", cid)  # encoded id carries thought signature

        # Round 2: send the tool result back with the same id
        status, raw = _post("/v1/chat/completions", {
            "model": "gemini-3.7-flash",
            "messages": [
                {"role": "user", "content": "use the glob tool with path 'x' and say done"},
                {"role": "assistant", "content": None, "tool_calls": msg["tool_calls"]},
                {"role": "tool", "tool_call_id": cid, "content": "found: x"},
            ],
            "tools": tools,
            "stream": False,
        })
        self.assertEqual(status, 200, raw[:400])

    def test_penalty_tolerated(self):
        status, _ = _post("/v1/chat/completions", {
            "model": "gemini-3.7-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "stream": False,
        })
        self.assertEqual(status, 200)

    def test_responses_basic(self):
        status, raw = _post("/v1/responses", {
            "model": "gemini-3.7-flash",
            "input": "hello",
            "stream": False,
        })
        self.assertEqual(status, 200, raw[:300])
        data = json.loads(raw)
        self.assertEqual(data["object"], "response")

    def test_json_schema(self):
        status, raw = _post("/v1/chat/completions", {
            "model": "gemini-3.7-flash",
            "messages": [{"role": "user", "content": "give one name"}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "u", "strict": True,
                "schema": {"type": "object",
                           "properties": {"name": {"type": "string"}},
                           "required": ["name"]}}},
            "stream": False,
        })
        self.assertEqual(status, 200, raw[:300])
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        self.assertIn("name", parsed)


if __name__ == "__main__":
    unittest.main()
