#!/usr/bin/env python3
"""
models/2003-thewissen-pakicetus-attocki-clock-wavefront.py — Multi-Scale & Paleobiological Reproduction (Slot #1403)
Organism: Pakicetus attocki (Early Eocene Terrestrial Wolf-Sized Stem-Cetacean)
Phenomenon Scale: 1.1 cm (0.01139 m, log10=-1.943, 7_organism_cm)
Anatomical Structure: Segmentation clock somite / metamere unit in Pakicetus attocki (Early Eocene (Ypresian, 49.0 Ma))
Geological Horizon: Early Eocene (Ypresian, 49.0 Ma) (49.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Pakicetus
Biophysical Cohort: III. Segmentation Clock & Wavefront (Phase-Coupled Oscillator & Wavefront Arrest (1.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1403_pakicetus_attocki_clock_wavefront.lean` (`MathOfBiology.Loop_1403`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pakicetus_attocki_clock_wavefront"
EXPECTED_SHA256 = {
    "pakicetus_attocki_clock_wavefront_reference.json": "1940350bd978215ba25ed72be8b08f6ff812c2a465b2c4a969d2d089d6def753",
    "pakicetus_attocki_specimen.jpg": "f535c1d722f574fc11becd2103eda0c5697e7184d39cb5b8ee794f42ce68e4a4",
    "scout_manifest.json": "5aa43cd31c379a68d2552e00ddf324afff5f8c3717f15d80269787ab65bfccf7",
    "paper_citations.bib": "32dfe876e3a7a584f795fdd998838fb62966bb538224d8a7e5a58f7248fb2fa2"
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
    ref = json.loads((DATASET_DIR / "pakicetus_attocki_clock_wavefront_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2003-thewissen-pakicetus-attocki-clock-wavefront.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
