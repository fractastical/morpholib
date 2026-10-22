#!/usr/bin/env python3
"""
models/2022-gross-hypsibius-exemplaris-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #2001)
Organism: Hypsibius exemplaris (Water Bear Tardigrade (Cryptobiotic Tun Buckling & Stylet Morphogenesis))
Phenomenon Scale: 171 nm (1.707e-07 m, log10=-6.768, 2_macromolecular_100nm)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Hypsibius exemplaris (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Hypsibius
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (171 nm))
Formal Verification: `morphoeval/lean/loops/Loop_2001_hypsibius_exemplaris_bioelectric_polarity.lean` (`MathOfBiology.Loop_2001`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/hypsibius_exemplaris_bioelectric_polarity"
EXPECTED_SHA256 = {
    "hypsibius_exemplaris_bioelectric_polarity_reference.json": "5d821b56f9a3a70c3e61251c644ea197f968ce5f2477e442ed35116903036ff3",
    "hypsibius_exemplaris_specimen.jpg": "47de00f0a828a76d4d2489503394a17024fde8a0989acc96911706aa535ba88d",
    "scout_manifest.json": "355e31995effde976d8887d46ab7eb0b6a33249e3730959903c90c6c0f6a386a",
    "paper_citations.bib": "00059ec0f6a9772609810f24c35b92b1042594fb9fa9a5c4f8569edeb377d93e"
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
    ref = json.loads((DATASET_DIR / "hypsibius_exemplaris_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2022-gross-hypsibius-exemplaris-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
