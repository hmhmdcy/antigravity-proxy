#!/usr/bin/env python3
"""Unit tests for the antigravity-proxy request/response transformations.

Runs offline (no network, no OAuth token): exercises the pure conversion
functions that map OpenAI <-> Gemini formats. This is the regression suite
for the compatibility fixes:
  - role:"tool" -> role:"user" functionResponse (+ name registry)
  - schema sanitizer (type arrays, exclusiveMinimum, additionalProperties,
    constraints folded into description)
  - penalty stripping
  - response_format -> responseMimeType/responseSchema
  - image_url data URLs -> inlineData parts
  - max_tokens thinking-budget compensation

Run:  python3 -m unittest discover -s tests -v
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import antigravity_proxy as ap


class TestSchemaSanitizer(unittest.TestCase):
    def test_type_array_nullable(self):
        out = ap._sanitize_gemini_schema({"type": ["number", "null"]})
        self.assertEqual(out["type"], "number")
        self.assertTrue(out.get("nullable"))

    def test_type_array_all_null(self):
        out = ap._sanitize_gemini_schema({"type": ["null"]})
        self.assertEqual(out["type"], "string")
        self.assertTrue(out.get("nullable"))

    def test_exclusive_minimum_folded_to_description(self):
        out = ap._sanitize_gemini_schema({"type": "integer", "exclusiveMinimum": 0})
        self.assertNotIn("exclusiveMinimum", out)  # dropped from schema
        self.assertIn("exclusiveMinimum: 0", out["description"])  # folded in

    def test_multiple_of_folded(self):
        out = ap._sanitize_gemini_schema({"type": "integer", "multipleOf": 5})
        self.assertIn("multipleOf: 5", out["description"])

    def test_additional_properties_false_kept(self):
        out = ap._sanitize_gemini_schema({"type": "object", "additionalProperties": False})
        self.assertIs(out.get("additionalProperties"), False)

    def test_reference_folded(self):
        out = ap._sanitize_gemini_schema({"type": "object", "$ref": "#/definitions/X"})
        self.assertNotIn("$ref", out)
        self.assertIn("$ref: #/definitions/X", out["description"])

    def test_recursive_properties(self):
        out = ap._sanitize_gemini_schema({
            "type": "object",
            "properties": {"a": {"type": ["string", "null"], "exclusiveMinimum": 1}},
        })
        self.assertEqual(out["properties"]["a"]["type"], "string")
        self.assertTrue(out["properties"]["a"]["nullable"])
        self.assertIn("exclusiveMinimum: 1", out["properties"]["a"]["description"])

    def test_enum_null(self):
        out = ap._sanitize_gemini_schema({"type": "string", "enum": ["a", None]})
        self.assertEqual(out["enum"], ["a"])
        self.assertTrue(out.get("nullable"))

    def test_constraint_not_duplicated(self):
        s = {"type": "integer", "exclusiveMinimum": 0}
        out1 = ap._sanitize_gemini_schema(s)
        out2 = ap._sanitize_gemini_schema(out1)
        self.assertEqual(out1["description"].count("exclusiveMinimum"), 1)
        self.assertEqual(out1["description"], out2["description"])

    def test_non_dict_passthrough(self):
        self.assertEqual(ap._sanitize_gemini_schema("nope"), "nope")
        self.assertEqual(ap._sanitize_gemini_schema(None), None)


class TestBuildGeminiRequest(unittest.TestCase):
    def _tool(self, name="glob", params=None):
        return [{
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": params or {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }]

    def _build(self, messages, **extra):
        body = {"model": "gemini-3.7-flash", "messages": messages}
        body.update(extra)
        return ap._build_gemini_request(body, "gemini-3.7-flash")

    def test_tool_message_role_is_user(self):
        cid = "call_call_2959504|ErgECrUEARFNMg9txNLdovEWM1gPNj8ZX0nvPKIdtC"
        env = self._build([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": cid, "type": "function",
                 "function": {"name": "glob", "arguments": '{"path":"x"}'}}]},
            {"role": "tool", "tool_call_id": cid, "content": "result"},
        ], tools=self._tool())
        contents = env["request"]["contents"]
        # role user + functionResponse part, not role function
        self.assertEqual(contents[2]["role"], "user")
        part = contents[2]["parts"][0]
        self.assertIn("functionResponse", part)
        # name must match the original functionCall name, not the encoded id
        self.assertEqual(part["functionResponse"]["name"], "glob")

    def test_tool_message_without_registry_falls_back(self):
        env = self._build([
            {"role": "user", "content": "q"},
            {"role": "tool", "tool_call_id": "call_abc", "content": "result"},
        ])
        contents = env["request"]["contents"]
        self.assertEqual(contents[-1]["role"], "user")
        self.assertEqual(contents[-1]["parts"][0]["functionResponse"]["name"], "call_abc")

    def test_assistant_function_call_encoded_id_decoded(self):
        cid = "call_call_42|sig123"
        env = self._build([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": cid, "type": "function",
                 "function": {"name": "glob", "arguments": '{"path":"x"}'}}]},
        ])
        contents = env["request"]["contents"]
        self.assertEqual(contents[1]["role"], "model")
        fc = contents[1]["parts"][0]["functionCall"]
        self.assertEqual(fc["name"], "glob")
        self.assertEqual(fc["id"], "call_42")
        self.assertEqual(contents[1]["parts"][0]["thoughtSignature"], "sig123")

    def test_parallel_tool_calls(self):
        id1 = "call_call_1|sig1"
        id2 = "call_call_2|"  # trailing pipe (no signature) - observed in the wild
        env = self._build([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": id1, "type": "function",
                 "function": {"name": "glob", "arguments": '{}'}},
                {"id": id2, "type": "function",
                 "function": {"name": "read", "arguments": '{}'}}]},
            {"role": "tool", "tool_call_id": id1, "content": "a"},
            {"role": "tool", "tool_call_id": id2, "content": "b"},
        ], tools=self._tool("glob") + self._tool("read"))
        contents = env["request"]["contents"]
        self.assertEqual(contents[1]["role"], "model")
        self.assertEqual(len(contents[1]["parts"]), 2)
        # both functionResponse parts resolve to their real names
        fr_names = [c["parts"][0]["functionResponse"]["name"]
                    for c in contents[2:]]
        self.assertIn("glob", fr_names)
        self.assertIn("read", fr_names)

    def test_penalty_stripped(self):
        env = self._build(
            [{"role": "user", "content": "hi"}],
            frequency_penalty=0.5, presence_penalty=0.5,
        )
        gc = env["request"].get("generationConfig", {})
        self.assertNotIn("frequencyPenalty", gc)
        self.assertNotIn("presencePenalty", gc)

    def test_response_format_json_object(self):
        env = self._build(
            [{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )
        gc = env["request"].get("generationConfig", {})
        self.assertEqual(gc.get("responseMimeType"), "application/json")

    def test_response_format_json_schema(self):
        env = self._build(
            [{"role": "user", "content": "hi"}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "p", "strict": True,
                "schema": {"type": "object",
                           "properties": {"age": {"type": "integer", "exclusiveMinimum": 0}},
                           "required": ["age"]}}},
        )
        gc = env["request"].get("generationConfig", {})
        self.assertEqual(gc.get("responseMimeType"), "application/json")
        schema = gc.get("responseSchema", {})
        self.assertNotIn("exclusiveMinimum", schema["properties"]["age"])
        self.assertIn("exclusiveMinimum: 0", schema["properties"]["age"]["description"])

    def test_max_tokens_thinking_budget(self):
        env = self._build([{"role": "user", "content": "hi"}], max_tokens=60)
        gc = env["request"].get("generationConfig", {})
        # output cap = client max_tokens + thinking budget; budget clamped 256..8192
        self.assertEqual(gc["maxOutputTokens"], 60 + 256)
        self.assertEqual(gc["thinkingConfig"]["thinkingBudget"], 256)

    def test_no_max_tokens_no_budget(self):
        env = self._build([{"role": "user", "content": "hi"}])
        gc = env["request"].get("generationConfig", {})
        self.assertNotIn("thinkingConfig", gc)

    def test_tool_choice_none(self):
        env = self._build([{"role": "user", "content": "hi"}],
                           tools=self._tool(), tool_choice="none")
        tc = env["request"].get("toolConfig", {})
        self.assertEqual(tc["functionCallingConfig"]["mode"], "NONE")

    def test_tool_choice_specific(self):
        env = self._build([{"role": "user", "content": "hi"}],
                           tools=self._tool(), tool_choice={"type": "function", "function": {"name": "glob"}})
        tc = env["request"].get("toolConfig", {})
        self.assertEqual(tc["functionCallingConfig"]["mode"], "ANY")
        self.assertIn("glob", tc["functionCallingConfig"]["allowedFunctionNames"])


class TestContentParts(unittest.TestCase):
    def test_text_string(self):
        self.assertEqual(ap._content_to_parts("hello"), [{"text": "hello"}])

    def test_text_blocks(self):
        parts = ap._content_to_parts([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
        self.assertEqual(parts, [{"text": "a"}, {"text": "b"}])

    def test_image_data_url_inline_data(self):
        parts = ap._content_to_parts([{
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        }])
        self.assertEqual(parts, [{
            "inlineData": {"mimeType": "image/png", "data": "AAAA"},
        }])

    def test_image_remote_url_marker(self):
        parts = ap._content_to_parts([{
            "type": "image_url",
            "image_url": {"url": "https://example.com/x.png"},
        }])
        self.assertEqual(parts, [{"text": "[image: https://example.com/x.png]"}])

    def test_none(self):
        self.assertEqual(ap._content_to_parts(None), [])


class TestToolChoice(unittest.TestCase):
    def test_auto(self):
        self.assertEqual(ap._tool_choice_to_gemini("auto"), "AUTO")

    def test_none(self):
        self.assertEqual(ap._tool_choice_to_gemini("none"), "NONE")

    def test_required(self):
        self.assertEqual(ap._tool_choice_to_gemini("required"), "ANY")

    def test_specific(self):
        self.assertEqual(ap._tool_choice_to_gemini({"type": "function", "function": {"name": "x"}}), "ANY")

    def test_unknown(self):
        self.assertIsNone(ap._tool_choice_to_gemini("weird"))


if __name__ == "__main__":
    unittest.main()
