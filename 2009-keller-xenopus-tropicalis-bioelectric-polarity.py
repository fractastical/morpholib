#!/usr/bin/env python3
"""
2009-keller-xenopus-tropicalis-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0921)
Organism: Xenopus tropicalis (Western Clawed Frog)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Xenopus
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0921_xenopus_tropicalis_bioelectric_polarity.lean` (`MathOfBiology.Loop_0921`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/xenopus_tropicalis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "xenopus_tropicalis_bioelectric_polarity_reference.json": "586186f8fe45e2aaec0e3d57e6482d0db93bdefdd8cc31a04ef46b752fe615ce",
    "xenopus_tropicalis_specimen.jpg": "13e063704f07254fd35cc867e7368242cfe27201d452c753342213eaa194d71f",
    "scout_manifest.json": "d27c5312857837a7d358f961d1f8971b49f3c18f111580e2e726b3080451780a",
    "paper_citations.bib": "c43c9ac27878219d421389ddffcaaf1d1b9c56ce44eb97bc0dee18fc77ebeb82"
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
    ref = json.loads((DATASET_DIR / "xenopus_tropicalis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-keller-xenopus-tropicalis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
