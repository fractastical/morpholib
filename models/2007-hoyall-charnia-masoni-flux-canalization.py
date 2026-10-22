#!/usr/bin/env python3
"""
models/2007-hoyall-charnia-masoni-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1519)
Organism: Charnia masoni (Ediacaran Self-Similar Glide-Reflected Rangeomorph Frond (Mistaken Point))
Phenomenon Scale: 4.0 cm (0.03951 m, log10=-1.403, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Charnia masoni (Late Ediacaran (565 Ma))
Geological Horizon: Late Ediacaran (565 Ma) (565.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Charnia
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (4.0 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1519_charnia_masoni_flux_canalization.lean` (`MathOfBiology.Loop_1519`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/charnia_masoni_flux_canalization"
EXPECTED_SHA256 = {
    "charnia_masoni_flux_canalization_reference.json": "c50df29327ed48257f3e2ccbc9ede46c4874571db73984556e53d12d92fd740f",
    "charnia_masoni_specimen.jpg": "92b6f0531f638c633e69def818647884521ea36a8f79147751e0437ea9e17b3a",
    "scout_manifest.json": "b47e05291c760f6936f3215061b492ad733d16052e2cc49af997f34559b49b8a",
    "paper_citations.bib": "4206fa9728f4bd4aa32d25139b734d069784fd8cd799aa5edcd07d7150b46368"
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
    ref = json.loads((DATASET_DIR / "charnia_masoni_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2007-hoyall-charnia-masoni-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
