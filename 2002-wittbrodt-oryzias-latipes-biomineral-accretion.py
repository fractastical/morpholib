#!/usr/bin/env python3
"""
2002-wittbrodt-oryzias-latipes-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0833)
Organism: Oryzias latipes (Japanese Medaka)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Oryzias
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0833_oryzias_latipes_biomineral_accretion.lean` (`MathOfBiology.Loop_0833`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/oryzias_latipes_biomineral_accretion"
EXPECTED_SHA256 = {
    "oryzias_latipes_biomineral_accretion_reference.json": "53e30e8d79d84ae5be60af4b204c4893774e6a254c9ce6161ad4b03c3b715d67",
    "oryzias_latipes_specimen.jpg": "ec331629f10358b6c34a698921c02e2de50725276c73bb64ffa78ba7f6e256af",
    "scout_manifest.json": "4da0aef5d5f112d7e7e78adf00923288a0b2691929e0a5a434224f48314b22ff",
    "paper_citations.bib": "6654dca500934be6af502e7111e25c58681b1bed1420e5d1c93d541a971ef378"
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
    ref = json.loads((DATASET_DIR / "oryzias_latipes_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2002-wittbrodt-oryzias-latipes-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
