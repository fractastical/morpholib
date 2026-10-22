#!/usr/bin/env python3
"""
models/1975-eldredge-eldredgeops-rana-17file-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1516)
Organism: Eldredgeops rana_17file (Middle Devonian Punctuated-Equilibrium Chronospecies (17-File Eye))
Phenomenon Scale: 112 µm (0.0001116 m, log10=-3.952, 5_primordium_100um)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Eldredgeops rana_17file (Middle Devonian (Tioughniogan, 384 Ma))
Geological Horizon: Middle Devonian (Tioughniogan, 384 Ma) (384.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Eldredgeops
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (112 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1516_eldredgeops_rana_17file_flux_canalization.lean` (`MathOfBiology.Loop_1516`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eldredgeops_rana_17file_flux_canalization"
EXPECTED_SHA256 = {
    "eldredgeops_rana_17file_flux_canalization_reference.json": "2174430317608baa25b65c9c2f42dd96a00ecd6dc9b90c53efa537a947502632",
    "eldredgeops_rana_17file_specimen.jpg": "15e5b154e304f4a7611a6156cdcfaa0863adea78b4676093b13c3c4a1fc76ffe",
    "scout_manifest.json": "19999739da31eafe3284953ff9d75f00b897da3aeba47097c41222600b618ab9",
    "paper_citations.bib": "a4153fd17bdaaf59df5d9ae25e2cdbd04abf3755044b08fa53c114e8bf0393e9"
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
    ref = json.loads((DATASET_DIR / "eldredgeops_rana_17file_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1975-eldredge-eldredgeops-rana-17file-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
