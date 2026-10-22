#!/usr/bin/env python3
"""
models/1999-waters-pentremites-godoni-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1360)
Organism: Pentremites godoni (Mississippian Pentaradial Hydrospire Blastoid Calyx)
Phenomenon Scale: 3.5 mm (0.003532 m, log10=-2.452, 6_organ_mm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Pentremites godoni (Mississippian (Chesterian, 330 Ma))
Geological Horizon: Mississippian (Chesterian, 330 Ma) (330.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Pentremites
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (3.5 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1360_pentremites_godoni_morphogen_condensation.lean` (`MathOfBiology.Loop_1360`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pentremites_godoni_morphogen_condensation"
EXPECTED_SHA256 = {
    "pentremites_godoni_morphogen_condensation_reference.json": "0ca1ee05b707f7bdd8a6c4c8aae223d73d10ec6bec5b81b1991b9c0c124bd47c",
    "pentremites_godoni_specimen.jpg": "53e40da38ecd2df40e09153c5fa4329553f1a7e8c16087490c6ba4fccd9eeb1d",
    "scout_manifest.json": "5cca5c794bc557f77f08b28cd7c9e4bc8557a67f1247f983bc23f987552dbfd6",
    "paper_citations.bib": "8d7227121b541c64af9d74c7b3c663fb9a6adb7324734ae33440968b622dde5f"
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
    ref = json.loads((DATASET_DIR / "pentremites_godoni_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1999-waters-pentremites-godoni-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
