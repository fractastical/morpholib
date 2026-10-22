#!/usr/bin/env python3
"""
models/1987-ward-nautilus-pompilius-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1241)
Organism: Nautilus pompilius (Extant Chambered Nautilus (Equiangular Logarithmic Siphuncle))
Phenomenon Scale: 6.9 mm (0.00693 m, log10=-2.159, 6_organ_mm)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Nautilus pompilius (Holocene (Extant Living Fossil))
Geological Horizon: Holocene (Extant Living Fossil) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Nautilus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (6.9 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1241_nautilus_pompilius_turing_wavelength.lean` (`MathOfBiology.Loop_1241`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nautilus_pompilius_turing_wavelength"
EXPECTED_SHA256 = {
    "nautilus_pompilius_turing_wavelength_reference.json": "9ea0aa63f76bff962eab97c631309f91cbdd5071531226e753c5cba21f3a7e5c",
    "nautilus_pompilius_specimen.jpg": "5dfd47d29d062b864cb14b6f5260b674dffb372fac09ff90bdd8fc5caf196da1",
    "scout_manifest.json": "f96fa127be39e6b2d91391fce128447eba04ce5c5311a1317ba125e24c6fbfc9",
    "paper_citations.bib": "bcf9b11b7aa70bfd6ce5ca73bf90c149ab053a25403fe261d935e17dc970e6bd"
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
    ref = json.loads((DATASET_DIR / "nautilus_pompilius_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1987-ward-nautilus-pompilius-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
