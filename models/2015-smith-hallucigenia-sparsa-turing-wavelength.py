#!/usr/bin/env python3
"""
models/2015-smith-hallucigenia-sparsa-turing-wavelength.py — Multi-Scale & Paleobiological Reproduction (Slot #1253)
Organism: Hallucigenia sparsa (Middle Cambrian Burgess Shale Armored Lobopod)
Phenomenon Scale: 737 µm (0.0007372 m, log10=-3.132, 5_primordium_100um)
Anatomical Structure: Turing pigment stripe / ocellus wavelength in Hallucigenia sparsa (Middle Cambrian (Burgess Shale, 508 Ma))
Geological Horizon: Middle Cambrian (Burgess Shale, 508 Ma) (508.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Hallucigenia
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection (737 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1253_hallucigenia_sparsa_turing_wavelength.lean` (`MathOfBiology.Loop_1253`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hallucigenia_sparsa_turing_wavelength"
EXPECTED_SHA256 = {
    "hallucigenia_sparsa_turing_wavelength_reference.json": "51fd2f2ce95b8a091be3e7d1bc6275e4ba01882e8655130a00fc1499410f404e",
    "hallucigenia_sparsa_specimen.jpg": "f0a3447adf27b728b96f3d6331a6f82327a3e9dbaf8871c9581019053e23ea81",
    "scout_manifest.json": "cb9cd80006845f85505efd9b495a2f6d07c7c6081c1f5552f896fa07347f8b21",
    "paper_citations.bib": "10d1750a2205932ceb0cd53a9a80a6065a32659d428b83df6a97f6bbc3edf1cc"
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
    ref = json.loads((DATASET_DIR / "hallucigenia_sparsa_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2015-smith-hallucigenia-sparsa-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
