"""10,000-animal basic-look campaign: catalog coverage + prompt contract."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from morphoeval.catalog_10000_animals import (  # noqa: E402
    END_SLOT, START_SLOT, TOTAL_NEW_SLOTS, build_all_specs, build_slot_spec, campaign_summary,
)
from morphoeval.cloud_campaign_10000 import build_prompt  # noqa: E402


class TestCatalog10000(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = build_all_specs()

    def test_slot_range_is_6900_new_loops_to_10000(self):
        self.assertEqual((START_SLOT, END_SLOT, TOTAL_NEW_SLOTS), (3101, 10000, 6900))
        self.assertEqual(len(self.specs), 6900)
        self.assertEqual([s["slot"] for s in self.specs], list(range(3101, 10001)))

    def test_all_isolation_paths_unique(self):
        for key in ("slug", "script", "dataset_dir", "lean_module", "sim_id"):
            vals = [s[key] for s in self.specs]
            self.assertEqual(len(vals), len(set(vals)), key)

    def test_all_eight_scale_tiers_and_ten_cohorts(self):
        self.assertEqual({s["scale_tier_num"] for s in self.specs}, set(range(1, 9)))
        self.assertEqual(len({s["cohort_idx"] for s in self.specs}), 10)
        sizes = [s["phenomenon_size_m"] for s in self.specs]
        self.assertLess(min(sizes), 1e-8)
        self.assertGreater(max(sizes), 1.0)

    def test_evolutionary_history_present(self):
        fossil = [s for s in self.specs if s["geological_age_ma"] > 0]
        self.assertGreater(len(fossil), 500)
        self.assertGreater(max(s["geological_age_ma"] for s in self.specs), 500)
        for s in self.specs:
            self.assertIn("chronoseries", s["evolutionary_axis"])

    def test_every_animal_gets_multiple_scales_and_cohorts(self):
        by_animal: dict[tuple, list] = {}
        for s in self.specs:
            by_animal.setdefault((s["genus"], s["species_epithet"]), []).append(s)
        self.assertGreater(len(by_animal), 250)
        for animal, runs in by_animal.items():
            self.assertGreaterEqual(len(runs), 10, animal)
            self.assertGreaterEqual(len({r["scale_tier_num"] for r in runs}), 3, animal)

    def test_summary_counts(self):
        summ = campaign_summary(self.specs)
        self.assertEqual(summ["new_loops"], 6900)
        self.assertEqual(summ["scale_tiers_covered"], list(range(1, 9)))


class TestPromptContract(unittest.TestCase):
    def test_prompt_has_2_to_3_remediation_loops_and_image_fitness(self):
        prompt = build_prompt(build_slot_spec(3101))
        for needle in ("REMEDIATION LOOP 1", "REMEDIATION LOOP 2", "PASS 3",
                       "IMAGE-FITNESS TEST", "image_fitness >= 70.0", "cv_sim_similarity >= 58.0",
                       "value != reference", "git diff --name-only --diff-filter=MD origin/main",
                       "ZERO-CONFLICT"):
            self.assertIn(needle, prompt)

    def test_prompt_names_assigned_scale_and_age(self):
        spec = build_slot_spec(10000)
        prompt = build_prompt(spec)
        self.assertIn(spec["organism"], prompt)
        self.assertIn(spec["size_human"], prompt)
        self.assertIn(spec["script"], prompt)


if __name__ == "__main__":
    unittest.main()
