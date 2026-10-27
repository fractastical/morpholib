"""Tier C / Step 13 human-expert validation loop: discovery, COI, timing."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from morphoeval.human_validation import (  # noqa: E402
    aggregate, load, load_config, new_request, path_for, rank_candidates,
    record_event, refresh, route_expertise, save,
)

POOL = [
    {"expert_id": "oa-src", "name": "Ada Source", "works_in_area": 9, "cited_by_count": 900,
     "last_active_year": 2026, "institutions": ["MIT"]},
    {"expert_id": "oa-a", "name": "Grace Expert", "works_in_area": 5, "cited_by_count": 200,
     "last_active_year": 2025, "institutions": ["Oxford"]},
    {"expert_id": "oa-b", "name": "Alan Expert", "works_in_area": 4, "cited_by_count": 100,
     "last_active_year": 2024, "institutions": ["Cambridge"]},
    {"expert_id": "oa-thin", "name": "Long Inactive", "works_in_area": 1, "cited_by_count": 3,
     "last_active_year": 1990, "institutions": []},
]


def queued(finding_id="test-finding", **kw):
    req = new_request({"finding_id": finding_id, "organism": "Neurospora crassa",
                       "claim": "claim", "source_authors": ["Ada Source"], **kw}, load_config())
    req["sla"]["queued_at"] = req["created_at"] = "2026-10-01T00:00:00Z"
    return refresh(req)


class TestExpertDiscovery(unittest.TestCase):
    def test_source_author_is_coi_excluded_and_ranked_last(self):
        ranked = rank_candidates(POOL, queued()["finding"], load_config(), current_year=2026)
        self.assertEqual(ranked[0]["name"], "Grace Expert")
        src = next(e for e in ranked if e["expert_id"] == "oa-src")
        self.assertTrue(src["coi"]["is_source_author"] and src["coi"]["ineligible"])
        self.assertFalse(src["eligibility"]["eligible"])

    def test_long_inactive_record_is_ineligible(self):
        ranked = rank_candidates(POOL, queued()["finding"], load_config(), current_year=2026)
        thin = next(e for e in ranked if e["expert_id"] == "oa-thin")
        self.assertFalse(thin["eligibility"]["eligible"])
        self.assertIn("inactive", thin["eligibility"]["reasons"])

    def test_routing(self):
        cfg = load_config()
        self.assertIn("zebrafish", route_expertise(cfg, "Danio rerio")["expertise"])
        self.assertIn("fungal", route_expertise(cfg, "Neurospora crassa")["expertise"])
        self.assertIn("query", route_expertise(cfg, "Unknown thing"))


class TestTimingValidateOrNot(unittest.TestCase):
    def test_time_to_verdict_after_quorum(self):
        req = queued()
        req["experts"] = rank_candidates(POOL, req["finding"], load_config(), current_year=2026)
        record_event(req, "invited", "oa-a", at="2026-10-02T00:00:00Z")
        record_event(req, "invited", "oa-b", at="2026-10-02T00:00:00Z")
        record_event(req, "accepted", "oa-a", at="2026-10-04T00:00:00Z")
        record_event(req, "accepted", "oa-b", at="2026-10-05T00:00:00Z")
        record_event(req, "verdict", "oa-a", at="2026-10-11T00:00:00Z", verdict="validated")
        self.assertIsNone(req["timing"]["time_to_validate_or_not_hours"])  # quorum not met
        record_event(req, "verdict", "oa-b", at="2026-10-13T00:00:00Z", verdict="validated")
        self.assertEqual(req["status"], "verdict:validated")
        self.assertEqual(req["timing"]["time_to_verdict_hours"], 12 * 24)
        self.assertEqual(req["timing"]["time_to_validate_or_not_hours"], 12 * 24)
        self.assertEqual(req["timing"]["time_to_first_response_hours"], 48.0)
        self.assertFalse(req["timing"]["is_overdue"])

    def test_refutation_is_also_a_timed_verdict(self):
        req = queued()
        req["experts"] = rank_candidates(POOL, req["finding"], load_config(), current_year=2026)
        for ex in ("oa-a", "oa-b"):
            record_event(req, "invited", ex, at="2026-10-02T00:00:00Z")
            record_event(req, "verdict", ex, at="2026-10-03T00:00:00Z", verdict="refuted")
        self.assertEqual(req["timing"]["outcome"], "refuted")
        self.assertEqual(req["timing"]["time_to_validate_or_not_hours"], 48.0)

    def test_non_validation_still_produces_a_duration(self):
        req = queued("never-validated")
        req["events"].append({"event": "closed_without_validation",
                              "at": "2026-10-22T00:00:00Z", "outcome": "expired"})
        refresh(req, at="2026-10-23T00:00:00Z")
        self.assertEqual(req["timing"]["outcome"], "expired")
        self.assertEqual(req["timing"]["time_to_non_validation_hours"], 21 * 24)
        self.assertEqual(req["timing"]["time_to_validate_or_not_hours"], 21 * 24)

    def test_open_and_overdue_is_flagged_not_silent(self):
        req = queued()
        refresh(req, at="2026-11-15T00:00:00Z")  # past 21-day SLA, no events
        self.assertEqual(req["timing"]["outcome"], "pending")
        self.assertTrue(req["timing"]["is_overdue"])
        self.assertIsNotNone(req["timing"]["elapsed_open_hours"])

    def test_coi_expert_cannot_record_events(self):
        req = queued()
        with self.assertRaises(ValueError):
            record_event(req, "verdict", "oa-a", verdict="maybe")  # bad verdict

    def test_roundtrip_and_aggregate_report(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            req = queued()
            req["experts"] = rank_candidates(POOL, req["finding"], load_config(), current_year=2026)
            for ex in ("oa-a", "oa-b"):
                record_event(req, "invited", ex, at="2026-10-02T00:00:00Z")
                record_event(req, "verdict", ex, at="2026-10-06T00:00:00Z", verdict="validated")
            save(req, path_for("test-finding", d))
            save(queued("open-one"), path_for("open-one", d))
            rep = aggregate(d)
            self.assertEqual(rep["n_requests"], 2)
            self.assertEqual(rep["by_outcome"]["validated"], 1)
            self.assertEqual(rep["by_outcome"]["pending"], 1)
            self.assertEqual(rep["time_to_validate_or_not"]["median_hours"], 5 * 24)
            self.assertEqual(json.loads(path_for("test-finding", d).read_text())["status"],
                             "verdict:validated")


if __name__ == "__main__":
    unittest.main()
