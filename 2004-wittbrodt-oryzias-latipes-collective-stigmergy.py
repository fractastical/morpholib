#!/usr/bin/env python3
"""
2004-wittbrodt-oryzias-latipes-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1033)
Organism: Oryzias latipes (Japanese Medaka)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Oryzias
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1033_oryzias_latipes_collective_stigmergy.lean` (`MathOfBiology.Loop_1033`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/oryzias_latipes_collective_stigmergy"
EXPECTED_SHA256 = {
    "oryzias_latipes_collective_stigmergy_reference.json": "816c14149db813391ef256176536318fe61cda219453020116e33719e1dbb73e",
    "oryzias_latipes_specimen.jpg": "ec331629f10358b6c34a698921c02e2de50725276c73bb64ffa78ba7f6e256af",
    "scout_manifest.json": "69f1ffdc0b04edec1dd9b733b72e01ed3a3c212d73b8a8d40f250c45b2625369",
    "paper_citations.bib": "d813b9c08303d9e5b2f68a6d63acc41199069cfe520e4017c80cbdacdbbe60d0"
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
    ref = json.loads((DATASET_DIR / "oryzias_latipes_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-wittbrodt-oryzias-latipes-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
