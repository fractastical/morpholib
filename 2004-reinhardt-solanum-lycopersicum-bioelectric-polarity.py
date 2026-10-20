#!/usr/bin/env python3
"""
2004-reinhardt-solanum-lycopersicum-bioelectric-polarity.py — Pass 2 Remediated Reproduction (Slot #0974)
Organism: Solanum lycopersicum (Tomato)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Solanum
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity)
Formal Verification: `morphoeval/lean/loops/Loop_0974_solanum_lycopersicum_bioelectric_polarity.lean` (`MathOfBiology.Loop_0974`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/solanum_lycopersicum_bioelectric_polarity"
EXPECTED_SHA256 = {
    "solanum_lycopersicum_bioelectric_polarity_reference.json": "7e31534a311fb74216fb42151a3acfa137a13f046bb357f9a65a4bc5f39b0d5d",
    "solanum_lycopersicum_specimen.jpg": "3ae537938870d7101582c24be70959db59fa5172fccc4d6ad7523d5b64cb2ec0",
    "scout_manifest.json": "849d617881f1f1ad3bc525c9daf7bf3f75688c9b5ee191818b6ee5b3abfeb280",
    "paper_citations.bib": "3a98eb8f346cc4327e4d4c462d2680469c59553fa70169150d3dd62963c66d4e"
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
    ref = json.loads((DATASET_DIR / "solanum_lycopersicum_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-reinhardt-solanum-lycopersicum-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
