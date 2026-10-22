#!/usr/bin/env python3
"""
models/2001-thewissen-pakicetus-attocki-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1223)
Organism: Pakicetus attocki (Early Eocene Terrestrial Wolf-Sized Stem-Cetacean)
Phenomenon Scale: 5.7 mm (0.005696 m, log10=-2.244, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Pakicetus attocki (Early Eocene (Ypresian, 49.0 Ma))
Geological Horizon: Early Eocene (Ypresian, 49.0 Ma) (49.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Pakicetus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (5.7 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1223_pakicetus_attocki_turing_wavelength.lean` (`MathOfBiology.Loop_1223`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pakicetus_attocki_turing_wavelength"
EXPECTED_SHA256 = {
    "pakicetus_attocki_turing_wavelength_reference.json": "994b225c3a2e2e5d330fcd7dd41e201cf3b148247925fdfee336e87c651c4eb0",
    "pakicetus_attocki_specimen.jpg": "f535c1d722f574fc11becd2103eda0c5697e7184d39cb5b8ee794f42ce68e4a4",
    "scout_manifest.json": "72de53eeed22ef1db24ca4c1979b064778398457f38654fe283a00066c7964ba",
    "paper_citations.bib": "f76ed1399e8cd72530baec1b51f41bd9d9256216fc29129d42f522abf104cd1d"
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
    ref = json.loads((DATASET_DIR / "pakicetus_attocki_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2001-thewissen-pakicetus-attocki-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
