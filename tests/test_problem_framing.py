"""Framing audit: promotion must use each problem's native falsifier, never the generic one."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "morphoeval"))

from morphoeval import audit_problem_framing as apf  # noqa: E402
from morphoeval.promote_findings import (  # noqa: E402
    falsifier_for, load_config, mechanism_from_slug, render_brief, score_finding,
)


class TestAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = apf.run_audit(write=False)

    def test_no_problem_graph_node_fits_generic_template(self):
        self.assertEqual(self.res["n_generic_framing_fits"], 0)
        self.assertGreaterEqual(self.res["n_nodes"], 30)

    def test_every_promotable_node_has_native_falsifier(self):
        for r in self.res["rows"]:
            if r["framing_verdict"] != "do_not_promote":
                self.assertTrue(r.get("native_falsifier"), r["node_id"])

    def test_registered_predictions_are_blind_locked(self):
        preds = [r for r in self.res["rows"] if r["problem_type"] == "registered_prediction"]
        self.assertEqual(len(preds), 14)
        for r in preds:
            self.assertEqual(r["framing_verdict"], "blind_locked_use_native_verbatim")
            self.assertTrue(r["native_falsifier_locked_on"])

    def test_solved_op_is_not_promoted(self):
        op06 = next(r for r in self.res["rows"] if r["node_id"] == "OP-06")
        self.assertEqual(op06["framing_verdict"], "do_not_promote")

    def test_prize_problems_route_to_math_audience(self):
        for nid in ("OP-01", "OP-02", "OP-03"):
            r = next(x for x in self.res["rows"] if x["node_id"] == nid)
            self.assertEqual(r["problem_type"], "formal_proof")
            self.assertIn("mathematicians", r["audience"])
            self.assertIn("barrier", r["promotion_track_hint"])  # posterior <15%

    def test_live_argument_with_settled_part_promotes_only_live_part(self):
        la01 = next(r for r in self.res["rows"] if r["node_id"] == "LA-01")
        self.assertEqual(la01["framing_verdict"], "adjudicate_camps")
        self.assertIn("remaining live part", la01["promotion_track_hint"])


class TestPromotionUsesNativeFraming(unittest.TestCase):
    def test_mechanism_from_slug(self):
        self.assertEqual(mechanism_from_slug("drosophila-guttifera-flux-canalization"), "flux_canalization")
        self.assertEqual(mechanism_from_slug("danio-rerio-turing-wavelength-s0"), "turing_wavelength")

    def test_dataset_match_uses_locked_falsifier_verbatim(self):
        row = {"finding_id": "x", "organism": "Danio rerio", "script": "models/2000-a-danio-rerio-turing-wavelength.py",
               "dataset_dir": "datasets/danio_rerio_turing_wavelength", "equations": [{"law_name": "Turing"}]}
        f = falsifier_for(row)
        self.assertIn(row["problem_ref"]["problem_type"], ("registered_prediction", "live_argument"))
        self.assertNotIn("right generative law", f)
        if row["problem_ref"]["problem_type"] == "registered_prediction":
            reg = next(r for r in apf.run_audit(write=False)["rows"] if r["node_id"] == row["problem_ref"]["node_id"])
            self.assertEqual(f, reg["native_falsifier"])  # verbatim, never paraphrased

    def test_unmapped_finding_falls_back_to_generic(self):
        row = {"finding_id": "y", "organism": "Neurospora crassa",
               "script": "models/2026-delphi10000-neurospora-crassa-biomineral-accretion-s0.py",
               "equations": [{"law_name": "Raup"}]}
        f = falsifier_for(row)
        self.assertIn("right generative law", f)
        self.assertEqual(row["problem_ref"]["framing_verdict"], "generic_fits")

    def test_brief_reflects_problem_type(self):
        row = {"finding_id": "x", "organism": "Danio rerio", "script": "models/2000-a-danio-rerio-turing-wavelength.py",
               "dataset_dir": "datasets/danio_rerio_turing_wavelength", "equations": [{"law_name": "Turing"}],
               "overall_score": 70.0, "image_fitness": 80.0, "pearson_r": 0.99, "rmse": 0.1, "parameters": {},
               "script_exists": True, "comparison_figure": "docs/media/x.png", "audit_file": "a.yaml",
               "phenomenon_size_m": 1e-3, "validation": {"consensus": None, "eligible_experts": 2, "field_citations": 10}}
        row["falsifier"] = falsifier_for(row)
        brief = render_brief(row, score_finding(row, load_config()), cfg=load_config())
        self.assertIn("Problem-graph node", brief)
        self.assertNotIn("does Turing actually generate this form", brief)
        self.assertTrue("locked" in brief or "done-criterion" in brief)


if __name__ == "__main__":
    unittest.main()
