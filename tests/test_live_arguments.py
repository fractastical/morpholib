"""Hermetic tests for the live-arguments map (predictions/live_arguments.json).

The map exists to aim the repo at contested questions scientists are
actually arguing about. These tests keep it honest: every entry must have
>=2 camps, datasets that exist on disk, a settling measurement, and a
candidate prediction that has NOT been quietly promoted to registered.

Run: python3 -m unittest tests.test_live_arguments -v
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_live_arguments", ROOT / "predictions" / "check_live_arguments.py")
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


class TestLiveArguments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.data = json.loads((ROOT / "predictions" / "live_arguments.json").read_text())

    def test_map_passes_all_checks(self):
        self.assertEqual(checker.check(self.data), [])

    def test_seven_arguments_two_plus_camps_each(self):
        args = self.data["arguments"]
        self.assertEqual(len(args), 7)
        for a in args:
            self.assertGreaterEqual(len(a["camps"]), 2, a["id"])

    def test_partly_settled_states_settled_half(self):
        for a in self.data["arguments"]:
            if a["argument_status"] == "partly_settled":
                self.assertTrue(a["settled_part"], a["id"])

    def test_no_candidate_is_registered(self):
        for a in self.data["arguments"]:
            cp = a["candidate_prediction"]
            self.assertEqual(cp["status"], "candidate_not_registered", a["id"])
            self.assertTrue(cp["falsifier"], a["id"])


if __name__ == "__main__":
    unittest.main()
