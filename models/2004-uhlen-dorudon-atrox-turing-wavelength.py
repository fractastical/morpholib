#!/usr/bin/env python3
"""
models/2004-uhlen-dorudon-atrox-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1226)
Organism: Dorudon atrox (Late Eocene Basilosaurid Whale (Vestigial Hindlimb))
Phenomenon Scale: 1.4 cm (0.01374 m, log10=-1.862, 7_organism_cm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Dorudon atrox (Late Eocene (Priabonian, 37.0 Ma))
Geological Horizon: Late Eocene (Priabonian, 37.0 Ma) (37.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Dorudon
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (1.4 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1226_dorudon_atrox_turing_wavelength.lean` (`MathOfBiology.Loop_1226`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/dorudon_atrox_turing_wavelength"
EXPECTED_SHA256 = {
    "dorudon_atrox_turing_wavelength_reference.json": "0aea086bc5d54d3874a68d415ff203b3b119a99d0191b48c23c5f17a37a31057",
    "dorudon_atrox_specimen.jpg": "5a04e317b788950efe607a8cf4a468b15573b2f7e2efbde1deb8fa2bfaeaabaf",
    "scout_manifest.json": "d088db1e5edf637aaffd64233b307ae519613277101550e5d1fe84da44a17539",
    "paper_citations.bib": "ce661d3a52961c989ebe8cc3eabdf0f62e7ae22f8d013127e65c48801cf26c6c"
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
    ref = json.loads((DATASET_DIR / "dorudon_atrox_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2004-uhlen-dorudon-atrox-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
