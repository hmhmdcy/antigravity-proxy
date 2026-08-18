#!/usr/bin/env python3
"""Unit tests for the Gemini -> OpenAI response transformation and the
Responses-API bridge. Offline, no network.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import antigravity_proxy as ap


class TestGeminiResponseToOpenAI(unittest.TestCase):
    def test_plain_text(self):
        resp = {
            "candidates": [{
                "content": {"parts": [{"text": "hello"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
        }
        out = ap.gemini_response_to_openai(resp, "gemini-3.7-flash")
        self.assertEqual(out["choices"][0]["message"]["content"], "hello")
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")
        self.assertEqual(out["usage"]["prompt_tokens"], 5)
        self.assertEqual(out["usage"]["completion_tokens"], 3)

    def test_tool_call_encoded_id(self):
        resp = {
            "candidates": [{
                "content": {"parts": [{
                    "functionCall": {"name": "glob", "args": {"path": "x"}, "id": "call_42"},
                    "thoughtSignature": "sig9",
                }]},
                "finishReason": "STOP",
            }],
        }
        out = ap.gemini_response_to_openai(resp, "gemini-3.7-flash")
        tc = out["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(tc["id"], "call_call_42|sig9")
        self.assertEqual(tc["function"]["name"], "glob")
        args = json.loads(tc["function"]["arguments"])
        self.assertEqual(args["path"], "x")
        self.assertEqual(out["choices"][0]["finish_reason"], "tool_calls")

    def test_tool_call_no_id_gets_generated(self):
        resp = {
            "candidates": [{
                "content": {"parts": [{
                    "functionCall": {"name": "glob", "args": {}},
                }]},
                "finishReason": "STOP",
            }],
        }
        out = ap.gemini_response_to_openai(resp, "gemini-3.7-flash")
        tc = out["choices"][0]["message"]["tool_calls"][0]
        self.assertTrue(tc["id"].startswith("call_"))
        self.assertGreater(len(tc["id"]), 5)

    def test_no_candidates_content_filter(self):
        resp = {"promptFeedback": {"blockReason": "SAFETY"}}
        out = ap.gemini_response_to_openai(resp, "gemini-3.7-flash")
        self.assertEqual(out["choices"][0]["finish_reason"], "content_filter")
        self.assertIn("blocked", out["choices"][0]["message"]["content"].lower())


class TestStreamDelta(unittest.TestCase):
    def test_text_delta(self):
        text, tcs, names = ap._extract_delta_from_parts([{"text": "hel"}], set())
        self.assertEqual(text, "hel")
        self.assertEqual(tcs, [])
        self.assertEqual(names, set())

    def test_function_call_delta(self):
        parts = [{"functionCall": {"name": "glob", "args": {"path": "x"}, "id": "call_7"},
                  "thoughtSignature": "sig1"}]
        text, tcs, names = ap._extract_delta_from_parts(parts, set())
        self.assertEqual(tcs[0]["index"], 0)
        self.assertEqual(tcs[0]["id"], "call_call_7|sig1")
        self.assertEqual(tcs[0]["function"]["name"], "glob")
        self.assertEqual(names, {"glob"})

    def test_second_tool_gets_next_index(self):
        parts = [{"functionCall": {"name": "glob", "args": {}, "id": "call_1"}},
                 {"functionCall": {"name": "read", "args": {}, "id": "call_2"}}]
        _, tcs, _ = ap._extract_delta_from_parts(parts, set())
        self.assertEqual([t["index"] for t in tcs], [0, 1])


class TestResponsesBridge(unittest.TestCase):
    def test_responses_input_to_messages_tool_pairing(self):
        msgs = ap._responses_input_to_messages({
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "q"}]},
                {"type": "function_call", "call_id": "call_1", "name": "glob",
                 "arguments": '{"path":"x"}'},
                {"type": "function_call_output", "call_id": "call_1", "output": "res"},
            ],
            "instructions": "be brief",
        })
        # first message: system from instructions
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("be brief", msgs[0]["content"])
        roles = [m["role"] for m in msgs]
        self.assertIn("user", roles)
        # tool message must carry the recovered function name
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["name"], "glob")

    def test_responses_tools_to_chat_tools(self):
        tools = [
            {"type": "function", "name": "glob",
             "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
            {"type": "web_search"},
        ]
        out = ap._responses_tools_to_chat_tools(tools)
        self.assertEqual(len(out), 1)  # web_search dropped
        self.assertEqual(out[0]["function"]["name"], "glob")

    def test_responses_usage(self):
        gemini = {"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4}}
        out = ap._gemini_usage_to_responses_usage(gemini)
        self.assertEqual(out["input_tokens"], 10)
        self.assertEqual(out["output_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
