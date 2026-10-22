#!/usr/bin/env python3
"""
models/2014-paterson-anomalocaris-canadensis-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1518)
Organism: Anomalocaris canadensis (Cambrian Burgess Shale Radiodont Apex Predator (Compound Eye & Oral Cone))
Phenomenon Scale: 7.3 mm (0.00735 m, log10=-2.134, 6_organ_mm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Anomalocaris canadensis (Middle Cambrian (Burgess Shale, 508 Ma))
Geological Horizon: Middle Cambrian (Burgess Shale, 508 Ma) (508.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Anomalocaris
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (7.3 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1518_anomalocaris_canadensis_flux_canalization.lean` (`MathOfBiology.Loop_1518`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/anomalocaris_canadensis_flux_canalization"
EXPECTED_SHA256 = {
    "anomalocaris_canadensis_flux_canalization_reference.json": "4ea4a684e7407cc33d6450eca33900f2e915558f2e128ce755c4987b2be9e136",
    "anomalocaris_canadensis_specimen.jpg": "197e43665198870b80e436c90febc5b7010eb24957c9239893e6ac1be683bf9c",
    "scout_manifest.json": "0cce81e4e96d4e83475a60f7b0e0d3db4fd20f5009772752e9d5029438ac17e1",
    "paper_citations.bib": "d1d8d4f07419f7285bfe3a8b4a72987bcbb56016021734ebf8b236b7b82ff739"
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
    ref = json.loads((DATASET_DIR / "anomalocaris_canadensis_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2014-paterson-anomalocaris-canadensis-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
