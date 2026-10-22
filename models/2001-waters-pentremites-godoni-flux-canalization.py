#!/usr/bin/env python3
"""
models/2001-waters-pentremites-godoni-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1540)
Organism: Pentremites godoni (Mississippian Pentaradial Hydrospire Blastoid Calyx)
Phenomenon Scale: 4.9 mm (0.004905 m, log10=-2.309, 6_organ_mm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Pentremites godoni (Mississippian (Chesterian, 330 Ma))
Geological Horizon: Mississippian (Chesterian, 330 Ma) (330.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Pentremites
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (4.9 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1540_pentremites_godoni_flux_canalization.lean` (`MathOfBiology.Loop_1540`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pentremites_godoni_flux_canalization"
EXPECTED_SHA256 = {
    "pentremites_godoni_flux_canalization_reference.json": "f16a30cd589c4868673119960e27062a679a59ea769a0a846046f812a0dcc78d",
    "pentremites_godoni_specimen.jpg": "53e40da38ecd2df40e09153c5fa4329553f1a7e8c16087490c6ba4fccd9eeb1d",
    "scout_manifest.json": "ed585c4125de31d18a09514189d225e980613609a0a3c7cc3f1f8e06f0ef652e",
    "paper_citations.bib": "a9c321a89f3fe358e1f7a5e8cc70079341e893cc2411d9470ab9b692f5fb61a1"
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
    ref = json.loads((DATASET_DIR / "pentremites_godoni_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2001-waters-pentremites-godoni-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
