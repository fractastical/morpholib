"""KG must not register placeholder Lean templates as proofs of named laws."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from morphoeval.math_kg import is_template_lean_equation  # noqa: E402
from morphoeval.audit_lean_proofs import classify_lean_text  # noqa: E402

TEMPLATE = """
def law1_invariant (a b : Int) : Int := a * b
theorem law1_positive_scaling (a b : Int) (ha : 0 < a) (hb : 0 < b) : 0 < law1_invariant a b := by
  dsimp [law1_invariant]; exact Int.mul_pos ha hb
def law2_balance (p q : Int) : Int := p + q - q
def law3_threshold_margin (lhs rhs : Int) : Int := lhs - rhs
"""


class TestGate(unittest.TestCase):
    def test_template_by_decl_name(self):
        self.assertTrue(is_template_lean_equation({"lean_decl": "MathOfBiology.Loop_0042.law1_invariant"}))
        self.assertTrue(is_template_lean_equation({"lean_theorem": "MathOfBiology.Loop_0042.law2_balance"}))

    def test_template_by_code_and_status(self):
        self.assertTrue(is_template_lean_equation({"lean_code": "def law2_balance (p q : Int) : Int := p + q - q"}))
        self.assertTrue(is_template_lean_equation({"formalization_status": "template_placeholder_not_a_proof"}))

    def test_substantive_equation_admitted(self):
        self.assertFalse(is_template_lean_equation({
            "lean_decl": "MathOfBiology.Turing.dispersion_det_pos",
            "lean_code": "theorem dispersion_det_pos (Du Dv : Real) : 0 < Du * Dv := by positivity",
        }))

    def test_classify(self):
        self.assertEqual(classify_lean_text(TEMPLATE)[0], "template_trivial")
        self.assertEqual(classify_lean_text("theorem murray_cube (r : Real) : r^3 = r^3 := rfl")[0], "substantive")


if __name__ == "__main__":
    unittest.main()
