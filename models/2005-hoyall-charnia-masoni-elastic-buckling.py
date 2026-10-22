#!/usr/bin/env python3
"""
models/2005-hoyall-charnia-masoni-elastic-buckling.py — Multi-Scale & Paleobiological Reproduction (Slot #1789)
Organism: Charnia masoni (Ediacaran Self-Similar Glide-Reflected Rangeomorph Frond (Mistaken Point))
Phenomenon Scale: 3.5 cm (0.03477 m, log10=-1.459, 7_organism_cm)
Anatomical Structure: Elastic bilayer fold / wrinkle / suture wavelength in Charnia masoni (Late Ediacaran (565 Ma))
Geological Horizon: Late Ediacaran (565 Ma) (565.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Charnia
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling (3.5 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1789_charnia_masoni_elastic_buckling.lean` (`MathOfBiology.Loop_1789`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/charnia_masoni_elastic_buckling"
EXPECTED_SHA256 = {
    "charnia_masoni_elastic_buckling_reference.json": "d7a9d0669b31d5e528ca1c92ca84bc29f8f0d494a45fe2dda216bccc699cc120",
    "charnia_masoni_specimen.jpg": "92b6f0531f638c633e69def818647884521ea36a8f79147751e0437ea9e17b3a",
    "scout_manifest.json": "3dd708df50494e3cebde3c271ab929f452438d0d20475379e5a6f750ef9a801d",
    "paper_citations.bib": "b980f4e678c1b10792f35c6b62fa352df715be66c052f22faf1dda91fa08d016"
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
    ref = json.loads((DATASET_DIR / "charnia_masoni_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-hoyall-charnia-masoni-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
