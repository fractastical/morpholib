"""Focused tests for morphoeval/check_lean_formula_alignment.py."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = ROOT / "morphoeval" / "check_lean_formula_alignment.py"
_spec = importlib.util.spec_from_file_location("check_lean_formula_alignment", CHECKER_PATH)
checker = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(checker)

POSITIVE = ROOT / "morphoeval" / "math_kg_loops" / "forterre-venus-flytrap-snap-buckling.yaml"
NEGATIVE = ROOT / "morphoeval" / "math_kg_loops" / "zhou-ginkgo-yimaensis-flux-canalization.yaml"


class TestLeanFormulaAlignment(unittest.TestCase):
    def test_positive_control_yaml_passes(self):
        self.assertEqual(checker.check_loop_yaml(POSITIVE), [])

    def test_negative_control_yaml_fails(self):
        errs = checker.check_loop_yaml(NEGATIVE)
        self.assertTrue(errs, "zhou stub YAML should report alignment mismatches")
        self.assertTrue(any("overnight stub" in e for e in errs))

    def test_overnight_template_stub_fails(self):
        errs = checker.check_equation_alignment(
            "overnight-law1",
            checker.OVERNIGHT_STUB_FORMULA,
            checker.OVERNIGHT_STUB_LEAN,
        )
        self.assertTrue(errs)

    def test_cli_self_test_exits_zero(self):
        proc = subprocess.run(
            [sys.executable, str(CHECKER_PATH), "--self-test"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
