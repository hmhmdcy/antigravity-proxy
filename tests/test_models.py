#!/usr/bin/env python3
"""Unit tests for agp.models dynamic discovery / merge logic."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agp.models as M


class TestModelMerge(unittest.TestCase):
    def test_friendly_name_strips_tier(self):
        self.assertEqual(M._friendly_name("gemini-3.7-flash-high"), "gemini-3.7-flash")
        self.assertEqual(M._friendly_name("gemini-3.6-flash-low"), "gemini-3.6-flash")
        self.assertEqual(M._friendly_name("gemini-3-flash"), "gemini-3-flash")
        self.assertEqual(M._friendly_name("claude-sonnet-4-6"), "claude-sonnet-4-6")

    def test_level_for(self):
        self.assertEqual(M._level_for("gemini-3.7-flash-high"), "high")
        self.assertEqual(M._level_for("gemini-3.6-flash-low"), "low")
        self.assertEqual(M._level_for("gemini-3.5-flash-extra-low"), "low")
        self.assertEqual(M._level_for("gemini-3-flash"), None)

    def test_merge_keeps_manual_and_adds_new(self):
        upstream = {
            "gemini-3.7-flash-high": {"maxTokens": 1048576},
            "gemini-3.8-flash-low": {"maxTokens": 1048576},
            "gemini-3-flash": {"maxTokens": 1048576},
            "chat_99999": {"maxTokens": 16384},
        }
        with mock.patch.object(M, "_fetch_upstream_models", return_value=upstream):
            merged, added, err = M.build_merged_model_map()
        self.assertIsNone(err)
        # manual 3.7-flash entry kept (high tier), discovered low variant skipped
        self.assertEqual(merged["gemini-3.7-flash"], ("gemini-3.7-flash-high", None))
        # brand-new 3.8 discovered under stripped name with level
        self.assertEqual(merged["gemini-3.8-flash"], ("gemini-3.8-flash-low", "low"))
        # no-tier model maps to itself
        self.assertEqual(merged["gemini-3-flash"], ("gemini-3-flash", None))
        # internal chat_* added
        self.assertEqual(merged["chat_99999"], ("chat_99999", None))
        self.assertEqual(added, 2)

    def test_merge_failure_degrades_to_static(self):
        with mock.patch.object(M, "_fetch_upstream_models",
                               side_effect=RuntimeError("boom")):
            merged, added, err = M.build_merged_model_map()
        self.assertIsNotNone(err)
        self.assertEqual(added, 0)
        # static map intact
        self.assertIn("gemini-3.7-flash", merged)


if __name__ == "__main__":
    unittest.main()
