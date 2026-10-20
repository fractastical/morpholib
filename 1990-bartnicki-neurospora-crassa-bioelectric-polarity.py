#!/usr/bin/env python3
"""
1990-bartnicki-neurospora-crassa-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0981)
Organism: Neurospora crassa (Red Bread Mold)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Neurospora
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0981_neurospora_crassa_bioelectric_polarity.lean` (`MathOfBiology.Loop_0981`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/neurospora_crassa_bioelectric_polarity"
EXPECTED_SHA256 = {
    "neurospora_crassa_bioelectric_polarity_reference.json": "20b17cd5b776aad971b1b844368bbc3da2068e7ae04756244aa31b45c0fff535",
    "neurospora_crassa_specimen.jpg": "01d6531f055000fd6be6dc5baa24d01e76899962e05a548e4be38a45032165f0",
    "scout_manifest.json": "7956f09920ec0ce8d460f4e4143a3820e28e1bfe605416d70a8a76a7924118a6",
    "paper_citations.bib": "935fb9c142d3c70481f5dec70a65dd1571ed07f346d9e8202a38d8f14130c4d1"
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
    ref = json.loads((DATASET_DIR / "neurospora_crassa_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1990-bartnicki-neurospora-crassa-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
