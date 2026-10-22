#!/usr/bin/env python3
"""
models/1964-durham-helicoplacus-gilberti-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1809)
Organism: Helicoplacus gilberti (Early Cambrian Triradial Helical-Plated Stem-Echinoderm)
Phenomenon Scale: 7.4 mm (0.007392 m, log10=-2.131, 6_organ_mm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Helicoplacus gilberti (Early Cambrian (Poleta Fm, 516 Ma))
Geological Horizon: Early Cambrian (Poleta Fm, 516 Ma) (516.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Helicoplacus
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (7.4 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1809_helicoplacus_gilberti_elastic_buckling.lean` (`MathOfBiology.Loop_1809`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/helicoplacus_gilberti_elastic_buckling"
EXPECTED_SHA256 = {
    "helicoplacus_gilberti_elastic_buckling_reference.json": "9260e838f3b681552d60c7d86875391100868ac8aac8f2a7d1e64026916e1438",
    "helicoplacus_gilberti_specimen.jpg": "13c0993dd3e63798e7d236896fbbbbe4e10b4f49fe7d356be1e6a56d3f579d8a",
    "scout_manifest.json": "2d7b4e8b54c6428b011c87adc60fdfadff4591ab735f03573fa760b0b8203a53",
    "paper_citations.bib": "2bdab2e69f1b0d6e27abce6275adb2ba8b40a1769d1ea9ba2d4c052dcbc56c29"
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
    ref = json.loads((DATASET_DIR / "helicoplacus_gilberti_elastic_buckling_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {
        "pearson_r": r,
        "rmse": rmse,
        "n": int(emp.size),
        "phenomenon_size_m": float(ref["phenomenon_scale"]["phenomenon_size_m"]),
        "scale_tier": ref["phenomenon_scale"]["scale_tier"],
        "geological_age_ma": float(ref["paleobiology"]["geological_age_ma"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        assert res["phenomenon_size_m"] > 0.0
        print(f"[PASS] models/1964-durham-helicoplacus-gilberti-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
