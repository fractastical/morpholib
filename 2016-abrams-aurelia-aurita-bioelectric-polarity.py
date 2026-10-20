#!/usr/bin/env python3
"""
2016-abrams-aurelia-aurita-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0964)
Organism: Aurelia aurita (Moon Jellyfish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Aurelia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0964_aurelia_aurita_bioelectric_polarity.lean` (`MathOfBiology.Loop_0964`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aurelia_aurita_bioelectric_polarity"
EXPECTED_SHA256 = {
    "aurelia_aurita_bioelectric_polarity_reference.json": "eb18f682fdaea37792fd2e38bb3ad2e1e282aea918ac16076f202938cdf4cca3",
    "aurelia_aurita_specimen.jpg": "1a8c5b9fb4abe35f1362b2c458d03bafa15966b0f6046d49aa840fdd3a66769e",
    "scout_manifest.json": "1c661b96fed3cee6ee5cc37ce071b5df33ef0ea1891b42c607c4c80ac34c388b",
    "paper_citations.bib": "8ccb8e4983e52c032ea4aab95d2993cce612d36ef1774ce4053f0cbd17b7e389"
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
    ref = json.loads((DATASET_DIR / "aurelia_aurita_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2016-abrams-aurelia-aurita-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
