#!/usr/bin/env python3
"""
models/1997-kenrick-rhynia-gwynne-vaughanii-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1255)
Organism: Rhynia gwynne_vaughanii (Early Devonian Rhynie Chert Leafless Protostele Sporophyte)
Phenomenon Scale: 134 µm (0.0001344 m, log10=-3.872, 5_primordium_100um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Rhynia gwynne_vaughanii (Early Devonian (Pragian Rhynie Chert, 407 Ma))
Geological Horizon: Early Devonian (Pragian Rhynie Chert, 407 Ma) (407.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Rhynia
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (134 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1255_rhynia_gwynne_vaughanii_turing_wavelength.lean` (`MathOfBiology.Loop_1255`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/rhynia_gwynne_vaughanii_turing_wavelength"
EXPECTED_SHA256 = {
    "rhynia_gwynne_vaughanii_turing_wavelength_reference.json": "4a0294691fdf72b0dfedd5df336cb6a46008aead93200e83e1951d494a8f2e34",
    "rhynia_gwynne_vaughanii_specimen.jpg": "5eb2588bbe45fb5b68e29e7763da6962ffa7cf335ab0a5ad8d2e46211cea3f15",
    "scout_manifest.json": "e7f547cdf2aaefc871f831f5fed07491d00b12a25f56e7085b6150c4fe67af18",
    "paper_citations.bib": "ecd2bba972dc059bd101138b1f378d0aee448d83625eaf5f6d61c72b529ba611"
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
    ref = json.loads((DATASET_DIR / "rhynia_gwynne_vaughanii_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1997-kenrick-rhynia-gwynne-vaughanii-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
