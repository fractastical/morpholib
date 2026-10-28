"""Tests for the Thompson (1917) claim-validation pipeline.

Hermetic by design: no network, no publishing, no signing. Everything here
runs against the committed artifacts, so a failure means the registry, the
literature layer, the dataset audit, the loop queue, or the nanopubs have
drifted apart -- which is exactly what this suite exists to catch.

Run (either works):
    venv/bin/python -m unittest discover -s tests -v
    venv/bin/python -m pytest tests/ -q        # if pytest is installed

Discipline under test (do not weaken these casually):
  * every Thompson claim is dated 1917 (1st edition);
  * every claim has exactly one literature verdict, dated on its own side;
  * NO_DIRECT_TEST is an explicit finding, never a missing field;
  * repo-2026 results are recorded separately and never as literature;
  * an UNPROBED dataset probe is never reported as "no dataset exists".
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NP_DIR = ROOT / "nanopubs" / "thompson_predictions"

CLAIMS = ROOT / "thompson" / "thompson_1917_claims.json"
COUNTER = ROOT / "thompson" / "thompson_1917_counterclaims.json"
AUDIT = ROOT / "thompson" / "thompson_1917_dataset_audit.json"
QUEUE = ROOT / "thompson" / "thompson_1917_loop_queue.yaml"

RELATIONS = {"REFUTED", "PARTIALLY_REFUTED", "SUPPORTED", "MIXED",
             "SUPERSEDED", "QUALIFIED", "NO_DIRECT_TEST", "NOT_TESTABLE"}
EXISTENCE = {"IN_REPO", "PUBLIC_SCRIPTABLE", "PUBLIC_CANDIDATE", "NO_DATA_NEEDED",
             "NO_PUBLIC_FOUND", "UNPROBED", "NOT_TESTABLE"}
AUTOMATION = {"FULL_LOOP", "LOOP_WITH_SEED", "HUMAN_REQUIRED", "DONE", "NA"}
PASS2 = {"DONE_REPO", "PARTIAL_REPO", "DONE_ELSEWHERE", "UNTESTED", "NOT_TESTABLE_AS_STATED"}
SCRIPTS = [
    "thompson/build_1917_claim_inventory.py",
    "thompson/build_1917_counterclaims.py",
    "thompson/scout_1917_datasets.py",
    "thompson/score_1917_resolvability.py",
    "thompson/merge_back_1917_verdicts.py",
    "thompson/run_overnight_1917.py",
    "nanopubs/thompson_1917_process_to_nanopub.py",
    "nanopubs/thompson_1917_counterclaims_to_nanopubs.py",
]


def load(path):
    return json.loads(path.read_text())


class TestClaimRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = load(CLAIMS)
        cls.claims = cls.reg["claims"]

    def test_59_claims_sequential_ids(self):
        self.assertEqual(len(self.claims), 59)
        self.assertEqual([c["id"] for c in self.claims],
                         [f"T1917-{i:03d}" for i in range(1, 60)])

    def test_every_claim_dated_1917_first_edition(self):
        self.assertEqual(self.reg["source"]["date_made"], "1917")
        self.assertIn("1st edition", self.reg["source"]["edition"])
        for c in self.claims:
            self.assertEqual(c["date_made"], "1917", c["id"])
            for field in ("chapter", "pages", "figures", "claim"):
                self.assertTrue(str(c[field]).strip(), f"{c['id']}.{field}")

    def test_source_full_text_pinned(self):
        src = self.reg["source"]["primary_text"]
        self.assertEqual(src["internet_archive_id"], "ongrowthform1917thom")
        self.assertRegex(src["full_text_sha256"], r"^[0-9a-f]{64}$")

    def test_pass2_blocks_valid(self):
        for c in self.claims:
            p = c["pass2"]
            self.assertIn(p["status"], PASS2, c["id"])
            lo, hi = p["effort_days"]
            self.assertTrue(0 <= lo <= hi, f"{c['id']} effort {lo}-{hi}")

    def test_resolutions_record_model_and_pr(self):
        # Every merged-back verdict must say who resolved the 1917 claim,
        # with what model, and where the evidence lives.
        resolved = [c for c in self.claims if "resolution" in c]
        self.assertEqual(len(resolved), 18)
        for c in resolved:
            r = c["resolution"]
            self.assertEqual(r["resolved_on"], "2026-10-28", c["id"])
            self.assertIn(r["model"], ("composer-2.5", "claude-fable-5-1"), c["id"])
            self.assertTrue(r["pr_url"].startswith("https://github.com/fractastical/morpholib/pull/"), c["id"])
            self.assertIn(r["verdict"], ("supported", "refuted", "mixed", "untestable"), c["id"])
            self.assertTrue((ROOT / r["result_json"]).exists(), c["id"])


class TestCounterclaims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = load(CLAIMS)
        cls.counter = load(COUNTER)
        cls.by_id = cls.counter["claims"]

    def test_full_coverage_no_extras(self):
        self.assertEqual(set(self.by_id), {c["id"] for c in self.reg["claims"]})

    def test_counts_block_matches_rows(self):
        from collections import Counter
        computed = dict(Counter(v["relation"] for v in self.by_id.values()))
        self.assertEqual(computed, self.counter["counts"])

    def test_every_row_dated_on_both_sides(self):
        for cid, v in self.by_id.items():
            self.assertIn("(1917)", v["citation_thompson_1917"], cid)
            self.assertIn(v["relation"], RELATIONS, cid)
            if v["relation"] not in ("NO_DIRECT_TEST", "NOT_TESTABLE"):
                self.assertNotEqual(v["who_and_when"].strip(), "-", cid)
                self.assertRegex(v["who_and_when"], r"(1[6-9]|20)\d\d", cid)
            self.assertTrue(v["counterclaim"].strip(), cid)

    def test_headline_margins_pinned(self):
        # The "beaten by X%" numbers: if these change, someone edited the
        # literature layer and should have to say so in a failing test.
        self.assertIn("0.35", self.by_id["T1917-021"]["quantitative_margin"])   # Fejes Toth 1964
        self.assertIn("0.3", self.by_id["T1917-022"]["quantitative_margin"])    # Weaire & Phelan 1994
        self.assertIn("1964", self.by_id["T1917-021"]["who_and_when"])

    def test_repo_2026_never_dressed_as_literature(self):
        # T1917-046: the quantitative refutation is repo 2026; the literature
        # verdict must stay NO_DIRECT_TEST and say so in words.
        v = self.by_id["T1917-046"]
        self.assertEqual(v["relation"], "NO_DIRECT_TEST")
        self.assertIn("REPO 2026", v["counterclaim"])
        for cid, row in self.by_id.items():
            self.assertIn("repo_2026_status_separate", row, cid)


class TestDatasetAuditAndQueue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = load(AUDIT)
        cls.rows = cls.audit["rows"]

    def test_rows_align_with_registry(self):
        reg_ids = {c["id"] for c in load(CLAIMS)["claims"]}
        self.assertEqual({r["id"] for r in self.rows}, reg_ids)
        for r in self.rows:
            self.assertEqual(r["date_made"], "1917", r["id"])
            self.assertIn(r["existence"], EXISTENCE, r["id"])
            self.assertIn(r["automation"], AUTOMATION, r["id"])

    def test_unprobed_never_reported_as_no_dataset(self):
        # NO_PUBLIC_FOUND is allowed only when a probe actually ran, or when
        # an explicit human blocker states why a dataset probe cannot settle
        # the claim (e.g. it needs an experiment, not a shape dataset).
        for r in self.rows:
            if r["existence"] == "NO_PUBLIC_FOUND":
                self.assertTrue(r["dryad_hits"] is not None or r["human_blocker"],
                                f"{r['id']}: NO_PUBLIC_FOUND with neither probe nor blocker")
            if r["automation"] == "HUMAN_REQUIRED":
                self.assertTrue(r["human_blocker"], r["id"])

    def test_loop_queue_matches_automation_tiers(self):
        queued = set(re.findall(r"claim_id:\s*(T1917-\d+)", QUEUE.read_text()))
        expected = {r["id"] for r in self.rows
                    if r["automation"] in ("FULL_LOOP", "LOOP_WITH_SEED")}
        self.assertEqual(queued, expected)
        self.assertEqual(len(queued), 47)


class TestResolvability(unittest.TestCase):
    """P(resolvable) must be logged for every claim and consistent everywhere."""

    @classmethod
    def setUpClass(cls):
        cls.ledger = load(ROOT / "thompson" / "thompson_1917_resolvability.json")
        cls.by_id = {x["claim_id"]: x for x in cls.ledger["claims"]}

    def test_ledger_covers_all_59_dated_1917(self):
        self.assertEqual([x["claim_id"] for x in self.ledger["claims"]],
                         [f"T1917-{i:03d}" for i in range(1, 60)])
        for x in self.ledger["claims"]:
            self.assertEqual(x["date_made"], "1917")
            self.assertTrue(0 <= x["p_percent"] <= 100, x["claim_id"])

    def test_audit_and_queue_carry_same_probability(self):
        audit = load(AUDIT)["rows"]
        for r in audit:
            self.assertEqual(r["p_resolvable_percent"], self.by_id[r["id"]]["p_percent"], r["id"])
        queue = QUEUE.read_text()
        for cid, x in self.by_id.items():
            if x["automation"] in ("FULL_LOOP", "LOOP_WITH_SEED"):
                self.assertIn(cid, queue)

    def test_rubric_invariants(self):
        sys.path.insert(0, str(ROOT / "thompson"))
        from score_1917_resolvability import score_claim
        self.assertGreater(score_claim("IN_REPO", "FULL_LOOP", "UNTESTED", [1, 2])["p_percent"],
                           score_claim("NO_PUBLIC_FOUND", "FULL_LOOP", "UNTESTED", [1, 2])["p_percent"])
        self.assertEqual(score_claim("IN_REPO", "DONE", "DONE_REPO", [1, 2])["p_percent"], 100.0)
        self.assertLessEqual(score_claim("PUBLIC_CANDIDATE", "HUMAN_REQUIRED", "UNTESTED", [1, 2])["p_percent"], 25.0)
        # An unprobed claim must never score as if absence were proven
        self.assertGreater(score_claim("UNPROBED", "FULL_LOOP", "UNTESTED", [1, 2])["p_percent"],
                           score_claim("NO_PUBLIC_FOUND", "FULL_LOOP", "UNTESTED", [1, 2])["p_percent"])


class TestNanopubs(unittest.TestCase):
    def test_claim_nanopub_files_and_manifest(self):
        files = sorted(NP_DIR.glob("np_thompson_claim1917-*.trig"))
        self.assertEqual(len(files), 59)
        for f in files:
            self.assertTrue((NP_DIR / f"signed.{f.name}").exists(), f.name)
        manifest = (NP_DIR / "published_manifest_prod.txt").read_text().splitlines()
        prod = [line.split("|") for line in manifest if "claim1917" in line]
        self.assertEqual(len(prod), 59)
        uris = [p[1] for p in prod]
        self.assertEqual(len(set(uris)), 59)
        for _, uri, _, registry in prod:
            self.assertRegex(uri, r"^https://w3id\.org/morpholib/thompson/np/claim1917-t1917-\d{3}/RA[0-9A-Za-z_-]+$")
            self.assertEqual(registry, "prod")

    def test_process_nanopubs_on_test_manifest(self):
        manifest = (NP_DIR / "published_manifest_test.txt").read_text().splitlines()
        files = {line.split("|")[0] for line in manifest if "process-" in line}
        self.assertEqual(len(files), 60)

    @unittest.skipUnless(Path(ROOT / "venv").exists(), "repo venv not present")
    def test_trig_parses_and_carries_dates(self):
        try:
            from rdflib import Dataset, Literal, Namespace
        except ImportError:
            self.skipTest("rdflib not installed")
        mlt = Namespace("https://w3id.org/morpholib/thompson/")
        counter = load(COUNTER)["claims"]
        for f in sorted(NP_DIR.glob("np_thompson_claim1917-*.trig")):
            ds = Dataset(default_union=True)
            ds.parse(f, format="trig")
            cid = f.stem.replace("np_thompson_claim1917-", "").upper()
            subj = list(ds.subjects(mlt.claimId, Literal(cid)))
            self.assertEqual(len(subj), 1, f.name)
            dates = [str(o) for o in ds.objects(subj[0], mlt.dateMade)]
            self.assertEqual(dates, ["1917"], f.name)
            relations = list(ds.objects(subj[0], mlt.literatureRelation)) or \
                [o for v in ds.objects(subj[0], mlt.hasLiteratureVerdict)
                 for o in ds.objects(v, mlt.literatureRelation)]
            self.assertTrue(relations, f.name)
            self.assertEqual(str(relations[0]), counter[cid]["relation"], f.name)


class TestBuildersAndDocs(unittest.TestCase):
    def run_builder(self, *argv):
        return subprocess.run([sys.executable, *argv], cwd=ROOT,
                              capture_output=True, text=True, timeout=300)

    def test_claim_inventory_check(self):
        r = self.run_builder("thompson/build_1917_claim_inventory.py", "--check", "--no-text")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PASS", r.stdout)

    def test_counterclaims_check(self):
        r = self.run_builder("thompson/build_1917_counterclaims.py", "--check")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PASS", r.stdout)

    def test_scripts_compile(self):
        import py_compile
        for rel in SCRIPTS:
            py_compile.compile(ROOT / rel, doraise=True)

    def test_docs_cover_every_claim(self):
        for doc in ("docs/THOMPSON_1917_CLAIM_INVENTORY.md",
                    "docs/THOMPSON_1917_COUNTERCLAIMS.md",
                    "docs/THOMPSON_1917_DATASET_EXISTENCE.md"):
            text = (ROOT / doc).read_text()
            for i in range(1, 60):
                self.assertIn(f"T1917-{i:03d}", text, f"{doc} missing T1917-{i:03d}")


if __name__ == "__main__":
    unittest.main()
