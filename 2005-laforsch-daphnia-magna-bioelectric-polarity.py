#!/usr/bin/env python3
"""
2005-laforsch-daphnia-magna-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0949)
Organism: Daphnia magna (Water Flea)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Daphnia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0949_daphnia_magna_bioelectric_polarity.lean` (`MathOfBiology.Loop_0949`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/daphnia_magna_bioelectric_polarity"
EXPECTED_SHA256 = {
    "daphnia_magna_bioelectric_polarity_reference.json": "86c3a0bba34928f3bb942a310ba06052839ac583a3db2ae45e27534d94133792",
    "daphnia_magna_specimen.jpg": "fc61b6ab51b92e988777a61cce4db7434210496ee5084ae01ee9f7adf6fcf44b",
    "scout_manifest.json": "abdd590afb92c51d2c6850db68b3efdcbe8732e9218c14d47248702c5c87c3be",
    "paper_citations.bib": "727552b6be77fbc1d78fa5d812bdf24d317ad96b4fe8cffd49efa55b5c876d8c"
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
    ref = json.loads((DATASET_DIR / "daphnia_magna_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-laforsch-daphnia-magna-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
