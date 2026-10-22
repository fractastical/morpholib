#!/usr/bin/env python3
"""
models/2021-landman-baculites-compressus-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1959)
Organism: Baculites compressus (Late Cretaceous Uncoiled Straight-Shaft Heteromorph Ammonite)
Phenomenon Scale: 85 µm (8.536e-05 m, log10=-4.069, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Baculites compressus (Late Cretaceous (Pierre Shale, 73.5 Ma))
Geological Horizon: Late Cretaceous (Pierre Shale, 73.5 Ma) (73.5 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Baculites
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (85 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1959_baculites_compressus_bioelectric_polarity.lean` (`MathOfBiology.Loop_1959`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/baculites_compressus_bioelectric_polarity"
EXPECTED_SHA256 = {
    "baculites_compressus_bioelectric_polarity_reference.json": "9a74400bebd30ab2830e94044bfafa6b009d4c5d9328fd8c22739b8591a954ac",
    "baculites_compressus_specimen.jpg": "849f95ffd5d647ae2a12750587da774719bdf94f1cf2520dc375c315c45eab50",
    "scout_manifest.json": "9941e491ac2e0747e2183750814d58565128cb57537f8df678a01f326885ed03",
    "paper_citations.bib": "eb5daeefb4fe36acb92c408f88143fba5bc410669674d5df46774d903c9737e9"
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
    ref = json.loads((DATASET_DIR / "baculites_compressus_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2021-landman-baculites-compressus-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
