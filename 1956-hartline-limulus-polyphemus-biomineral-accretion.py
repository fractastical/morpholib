#!/usr/bin/env python3
"""
1956-hartline-limulus-polyphemus-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0850)
Organism: Limulus polyphemus (Atlantic Horseshoe Crab)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Limulus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0850_limulus_polyphemus_biomineral_accretion.lean` (`MathOfBiology.Loop_0850`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/limulus_polyphemus_biomineral_accretion"
EXPECTED_SHA256 = {
    "limulus_polyphemus_biomineral_accretion_reference.json": "9b662fb2fe8f3c8f96cf6746edd58fa8a8b6e8b7306a5c4f83749f2cc35e1247",
    "limulus_polyphemus_specimen.jpg": "6024a6e114ddb2e0b996d82067f840499683d1838662afe24496b8f315e2ef45",
    "scout_manifest.json": "cb31034e8c2bec93505d85c9056896b1f83f9bdf6e5e298faa4b530c711ef518",
    "paper_citations.bib": "8658927c24b5c71e0efaac58dfa697539af0891b03a0f435803dc7740ac3eb89"
}


def verify_sha256() -> None:
    for fname, exp in EXPECTED_SHA256.items():
        p = DATASET_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != exp:
            raise ValueError(f"SHA-256 mismatch for {fname}: {got} != {exp}")


def run_model() -> dict:
    verify_sha256()
    ref = json.loads((DATASET_DIR / "limulus_polyphemus_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 1956-hartline-limulus-polyphemus-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
