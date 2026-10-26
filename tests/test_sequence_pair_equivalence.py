"""Logged test: BMP activator-antagonist pair-equivalence claim.

Claim logged 2026-10-26 (sequences/README.md): Xenopus bmp4/nog are
"orthologs of the same pair" as zebrafish bmp2b/chd, interchangeable as a
Turing activator-inhibitor pair under the repo size prior D ~ L^(-1/3).

Hermetic half (this file, no network): the diffusion-ordering half is
REFUTED on the SHA-256-pinned sequences themselves -- zebrafish gives
D_chd/D_bmp2b ~= 0.759 < 1 (inhibitor predicted slower), Xenopus gives
D_nog/D_bmp4 ~= 1.283 > 1. The pairs cannot both be the same Turing pair
under this prior. The orthology half needs Ensembl Compara and lives in
sequences/check_pair_equivalence.py --online; its run log is committed at
sequences/pair_equivalence_log.json.

Run:
    venv/bin/python -m pytest tests/test_sequence_pair_equivalence.py -q
    python3 sequences/check_pair_equivalence.py --online   # full verdict
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_pair_equivalence", ROOT / "sequences" / "check_pair_equivalence.py")
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)


class TestPairEquivalenceLoggedClaim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolved = checker.load_resolved()
        cls.log = checker.run(online=False)

    def test_all_17_pinned_proteins_verify_against_cache(self):
        rows = checker.verify_hashes(self.resolved)
        self.assertEqual(len(rows), 17)
        for row in rows:
            self.assertTrue(row["sha256_verified"], row)

    def test_both_catalog_pairs_present_with_roles(self):
        for species, activator, inhibitor in checker.PAIRS:
            self.assertIn((species, activator), self.resolved)
            self.assertIn((species, inhibitor), self.resolved)

    def test_diffusion_ratios_pinned(self):
        pairs = {(r["species"]): r for r in
                 self.log["subclaim_B_diffusion_equivalence"]["pairs"]}
        # (411/940)^(1/3) and (469/222)^(1/3) on the pinned precursor lengths.
        self.assertAlmostEqual(pairs["danio_rerio"]["D_inhibitor_over_D_activator"],
                               0.759, places=3)
        self.assertAlmostEqual(pairs["xenopus_tropicalis"]["D_inhibitor_over_D_activator"],
                               1.283, places=3)

    def test_catalog_declares_corrected_relationship(self):
        # After the 2026-10-26 refutation, catalog.json must carry the
        # correction in machine-readable form, and its declared size-prior
        # flags must match the ratios computed from the pinned sequences.
        cc = self.log["subclaim_C_catalog_consistency"]
        self.assertTrue(cc["catalog_consistent"], cc)
        for row in cc["entries"]:
            if "declared_size_prior_supports_faster_inhibitor" in row:
                self.assertEqual(row["relationship"],
                                 "functionally_analogous_not_orthologous", row)
                self.assertTrue(row["flag_matches_computed"], row)

    def test_pair_equivalence_claim_is_refuted_offline(self):
        # The logged claim requires BOTH pairs to pass; zebrafish does not.
        equiv = self.log["subclaim_B_diffusion_equivalence"]
        self.assertFalse(equiv["all_pairs_pass"])
        self.assertEqual(self.log["verdict"], "REFUTED")
        failing = [r["species"] for r in equiv["pairs"]
                   if not r["supports_faster_inhibitor"]]
        self.assertEqual(failing, ["danio_rerio"])


if __name__ == "__main__":
    unittest.main()
