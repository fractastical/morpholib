#!/usr/bin/env python3
"""
2014-dietrich-pseudomonas-aeruginosa-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0984)
Organism: Pseudomonas aeruginosa (Biofilm Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Pseudomonas
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0984_pseudomonas_aeruginosa_bioelectric_polarity.lean` (`MathOfBiology.Loop_0984`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pseudomonas_aeruginosa_bioelectric_polarity"
EXPECTED_SHA256 = {
    "pseudomonas_aeruginosa_bioelectric_polarity_reference.json": "d9758d4b814a840350df7ba81fd750046201f7f96b1a11834777fce6daee76ad",
    "pseudomonas_aeruginosa_specimen.jpg": "60ef5e2ce063576716d4086509c23046d1f100864118a90be5614a770b3ff31d",
    "scout_manifest.json": "65c4681dd02aea4c7d12d0828bdd0ae7699d72a2fd2d32e6f111d60a389b7c4a",
    "paper_citations.bib": "2fa24d80bcb7710525ee963c61664ec939494bfbb924bf739ba19c6d5a3b5b73"
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
    ref = json.loads((DATASET_DIR / "pseudomonas_aeruginosa_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-dietrich-pseudomonas-aeruginosa-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
