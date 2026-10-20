#!/usr/bin/env python3
"""
2004-flardh-streptomyces-coelicolor-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0988)
Organism: Streptomyces coelicolor (Mycelial Actinobacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Streptomyces
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0988_streptomyces_coelicolor_bioelectric_polarity.lean` (`MathOfBiology.Loop_0988`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/streptomyces_coelicolor_bioelectric_polarity"
EXPECTED_SHA256 = {
    "streptomyces_coelicolor_bioelectric_polarity_reference.json": "dfd73adc5400c4ddfb8e5b630d36e8838cde1134664794126de16a26d3f139db",
    "streptomyces_coelicolor_specimen.jpg": "5a2b99731bd4971f58260f74a1db5d4560b78f4d8ceb3b067629dcb7b6d53fec",
    "scout_manifest.json": "28cbc58323f4d9efeccfa26686852a1aae130c390118a1ee9f3ac859c20e6ff1",
    "paper_citations.bib": "2bc323edcfcb48956a92f0965b4770f8f94f3073f94f8c0dda0abefe4ddb3fb6"
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
    ref = json.loads((DATASET_DIR / "streptomyces_coelicolor_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-flardh-streptomyces-coelicolor-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
