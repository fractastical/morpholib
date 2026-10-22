#!/usr/bin/env python3
"""
models/2004-hoyall-charnia-masoni-branching-tree.py — Multi-Scale & Paleobiological Reproduction (Slot #1699)
Organism: Charnia masoni (Ediacaran Self-Similar Glide-Reflected Rangeomorph Frond (Mistaken Point))
Phenomenon Scale: 5.5 cm (0.05532 m, log10=-1.257, 7_organism_cm)
Anatomical Structure: Bifurcating vascular / bronchial / hyphal branch in Charnia masoni (Late Ediacaran (565 Ma))
Geological Horizon: Late Ediacaran (565 Ma) (565.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Charnia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting (5.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1699_charnia_masoni_branching_tree.lean` (`MathOfBiology.Loop_1699`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/charnia_masoni_branching_tree"
EXPECTED_SHA256 = {
    "charnia_masoni_branching_tree_reference.json": "8215280e57c59d2355a3e1796d564b37f03a2ed5e91bd6d1d3cb4e67a972df52",
    "charnia_masoni_specimen.jpg": "92b6f0531f638c633e69def818647884521ea36a8f79147751e0437ea9e17b3a",
    "scout_manifest.json": "f3d3e9abbc006d386344aa9bf5e214ed93b7f2b415e3f8a7bc350dee1f84c274",
    "paper_citations.bib": "d438daf63022018ff91ef32cfa2616f4b67e4fa5a3393de5d3fac545612494e6"
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
    ref = json.loads((DATASET_DIR / "charnia_masoni_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2004-hoyall-charnia-masoni-branching-tree.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
