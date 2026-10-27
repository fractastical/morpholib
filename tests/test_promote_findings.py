"""Scientist promotion loop: care scoring, two-track gates, brief, funnel."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "morphoeval"))

from morphoeval.promote_findings import (  # noqa: E402
    load_config, render_brief, report, score_finding, track_event,
)


def row(**kw):
    base = {"finding_id": "f1", "organism": "Drosophila melanogaster", "script": "models/x.py",
            "overall_score": 62.0, "scores": {}, "phenomenon_size_m": 1e-4, "audit_file": "a.yaml",
            "image_fitness": 88.0, "pearson_r": 0.999, "rmse": 0.05, "parameters": {"a": 1},
            "equations": [{"law_name": "Test Law"}], "math_family": "Test", "script_exists": True,
            "comparison_figure": "docs/media/x_comparison.png", "animation": "x.gif",
            "dataset_dir": "datasets/x", "falsifier": "Is 'Test Law' right?",
            "validation": {"consensus": None, "eligible_experts": 3, "field_citations": 900}}
    return {**base, **kw}


class TestGates(unittest.TestCase):
    def test_unvalidated_never_promoted_as_result(self):
        s = score_finding(row(), load_config())
        self.assertEqual(s["track"], "promote_for_scrutiny")

    def test_validated_and_strong_becomes_result(self):
        s = score_finding(row(overall_score=82.0,
                              validation={"consensus": "validated", "eligible_experts": 2,
                                          "field_citations": 100}), load_config())
        self.assertEqual(s["track"], "promote_as_result")

    def test_validated_but_weak_stays_scrutiny(self):
        s = score_finding(row(overall_score=55.0,
                              validation={"consensus": "validated", "eligible_experts": 2,
                                          "field_citations": 100}), load_config())
        self.assertNotEqual(s["track"], "promote_as_result")
        self.assertIn("overall >= gate", s["missing_for_result"])

    def test_no_experts_means_hold_even_if_interesting(self):
        s = score_finding(row(validation=None), load_config())
        self.assertEqual(s["track"], "hold_do_not_promote")
        self.assertIn("eligible experts to ask", s["missing_for_scrutiny"])

    def test_no_falsifier_means_hold(self):
        s = score_finding(row(falsifier=None, equations=[]), load_config())
        self.assertEqual(s["track"], "hold_do_not_promote")


class TestBriefIsActionable(unittest.TestCase):
    def test_brief_contract(self):
        r = row()
        brief = render_brief(r, score_finding(r, load_config()), cfg=load_config())
        for needle in ("2-minute triage", "`care`", "`kill_it`", "15-minute verdict",
                       "One-sitting replication", "python3 models/x.py --self-test",
                       "What would kill it", "Caveats, upfront", "no human expert has validated"):
            self.assertIn(needle, brief)

    def test_personalised_why_you(self):
        r = row()
        expert = {"name": "Dr Wing", "institutions": ["Oxford"],
                  "example_works": [{"title": "Wing patterns in flies"}]}
        brief = render_brief(r, score_finding(r, load_config()), expert=expert, cfg=load_config())
        self.assertIn("Why you, Dr Wing", brief)
        self.assertIn("wrong_expert", brief)


class TestFunnelTiming(unittest.TestCase):
    def test_care_rate_and_time_to_first_care(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            track_event("f1", "brief_ready", d, at="2026-10-01T00:00:00Z")
            track_event("f1", "sent", d, expert_id="oa-1", channel="direct_expert_invitation",
                        at="2026-10-01T01:00:00Z")
            track_event("f1", "sent", d, expert_id="oa-2", at="2026-10-01T01:00:00Z")
            track_event("f1", "care_response", d, expert_id="oa-1", care="care", at="2026-10-01T07:00:00Z")
            doc = track_event("f1", "care_response", d, expert_id="oa-2", care="kill_it",
                              at="2026-10-02T01:00:00Z")
            self.assertEqual(doc["funnel"]["time_to_first_care_hours"], 6.0)
            self.assertEqual(doc["funnel"]["care_rate"], 0.5)
            self.assertEqual(doc["funnel"]["kill_it_count"], 1)
            rep = report(d)
            self.assertEqual(rep["n_promotions"], 1)
            self.assertEqual(rep["overall_care_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
