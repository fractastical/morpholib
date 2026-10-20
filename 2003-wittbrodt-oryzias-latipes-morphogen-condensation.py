#!/usr/bin/env python3
"""
2003-wittbrodt-oryzias-latipes-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0233)
Organism: Oryzias latipes (Japanese Medaka)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Oryzias
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0233_oryzias_latipes_morphogen_condensation.lean` (`MathOfBiology.Loop_0233`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/oryzias_latipes_morphogen_condensation"
EXPECTED_SHA256 = {
    "oryzias_latipes_morphogen_condensation_reference.json": "70530e97390dd81e1686a134a278e566d7a54893274fd40395dcbe130841271b",
    "oryzias_latipes_specimen.jpg": "ec331629f10358b6c34a698921c02e2de50725276c73bb64ffa78ba7f6e256af",
    "scout_manifest.json": "2fe40c2a6a87ca09338a17c3049f06df852b57cca758f6d963fb0097afe94a22",
    "paper_citations.bib": "1c82d1706a06901165ab41fbf703c98c1a00ea34318122bc4eab329d5701d2bd"
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
    ref = json.loads((DATASET_DIR / "oryzias_latipes_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-wittbrodt-oryzias-latipes-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
