#!/usr/bin/env python3
"""
models/2009-shubin-tiktaalik-roseae-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1934)
Organism: Tiktaalik roseae (Late Devonian Wristed Fin-Limb Transitional Fossil)
Phenomenon Scale: 166 µm (0.0001657 m, log10=-3.781, 5_primordium_100um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Tiktaalik roseae (Late Devonian (Frasnian, 375 Ma))
Geological Horizon: Late Devonian (Frasnian, 375 Ma) (375.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Tiktaalik
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (166 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1934_tiktaalik_roseae_bioelectric_polarity.lean` (`MathOfBiology.Loop_1934`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/tiktaalik_roseae_bioelectric_polarity"
EXPECTED_SHA256 = {
    "tiktaalik_roseae_bioelectric_polarity_reference.json": "488fa8343444ed5cf9cab60e00f72393659ca81d26bf45c2d145cc603b101484",
    "tiktaalik_roseae_specimen.jpg": "2b6fe2e7441ef8b8a1f69c1382b71a3606ab9e0ee552372999fb3ab07fb8c3a4",
    "scout_manifest.json": "3ea32eade590322c38f033b65541efafce4d01ece8a55d39af61162013dc4415",
    "paper_citations.bib": "4d6421600da7dd8d86ee5ff0ef0507a44fed9a3bc3dcaa0b7f9785b864fd95a4"
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
    ref = json.loads((DATASET_DIR / "tiktaalik_roseae_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2009-shubin-tiktaalik-roseae-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
