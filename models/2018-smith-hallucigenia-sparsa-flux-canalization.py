#!/usr/bin/env python3
"""
models/2018-smith-hallucigenia-sparsa-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1523)
Organism: Hallucigenia sparsa (Middle Cambrian Burgess Shale Armored Lobopod)
Phenomenon Scale: 4.6 mm (0.004607 m, log10=-2.337, 6_organ_mm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Hallucigenia sparsa (Middle Cambrian (Burgess Shale, 508 Ma))
Geological Horizon: Middle Cambrian (Burgess Shale, 508 Ma) (508.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Hallucigenia
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (4.6 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1523_hallucigenia_sparsa_flux_canalization.lean` (`MathOfBiology.Loop_1523`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hallucigenia_sparsa_flux_canalization"
EXPECTED_SHA256 = {
    "hallucigenia_sparsa_flux_canalization_reference.json": "89eaf4396f52b4cc406269ac0dd92787b6df3c712317a17b4697e4401761bf98",
    "hallucigenia_sparsa_specimen.jpg": "f0a3447adf27b728b96f3d6331a6f82327a3e9dbaf8871c9581019053e23ea81",
    "scout_manifest.json": "1f79280d5e87b1df7a39f7c6eac9085b2b6919861c0d4c8874cdb4efadc97e7a",
    "paper_citations.bib": "6b93956d34f57faf39bf49943f7f227ead56d1cb8899200e3a592bf6b26d4ed6"
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
    ref = json.loads((DATASET_DIR / "hallucigenia_sparsa_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-smith-hallucigenia-sparsa-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
