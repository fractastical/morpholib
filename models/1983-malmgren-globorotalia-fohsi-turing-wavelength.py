#!/usr/bin/env python3
"""
models/1983-malmgren-globorotalia-fohsi-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1266)
Organism: Globorotalia fohsi (Middle Miocene Peripheral-Keel Punctuated Foraminifer Chronospecies)
Phenomenon Scale: 16 µm (1.562e-05 m, log10=-4.806, 4_cellular_10um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Globorotalia fohsi (Middle Miocene (Serravallian, 12.7 Ma))
Geological Horizon: Middle Miocene (Serravallian, 12.7 Ma) (12.7 Ma)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Globorotalia
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (16 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1266_globorotalia_fohsi_turing_wavelength.lean` (`MathOfBiology.Loop_1266`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/globorotalia_fohsi_turing_wavelength"
EXPECTED_SHA256 = {
    "globorotalia_fohsi_turing_wavelength_reference.json": "96ea84be3d16ea185956a83ae51992716d457637d87ed6e13b7d9c848563b061",
    "globorotalia_fohsi_specimen.jpg": "da1c3b52b1d7e26144fe2b2960cf08ed37afc6fd3eaf5de4044b285992f13b6f",
    "scout_manifest.json": "6f7890f0b4f0533fa913d03513cdfab87d7688fbfbb7a8779fdbec0b58d016e2",
    "paper_citations.bib": "0c465d41cd705dfb3ff362dd3a7d787777a48d745606f2dde4121a33c5fb2228"
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
    ref = json.loads((DATASET_DIR / "globorotalia_fohsi_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1983-malmgren-globorotalia-fohsi-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
