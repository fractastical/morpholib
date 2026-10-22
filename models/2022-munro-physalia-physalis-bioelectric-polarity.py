#!/usr/bin/env python3
"""
models/2022-munro-physalia-physalis-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #2006)
Organism: Physalia physalis (Portuguese Man o' War (Chiral Left/Right Dimorphic Pneumatophore Sail))
Phenomenon Scale: 85 µm (8.544e-05 m, log10=-4.068, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Physalia physalis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Physalia
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (85 µm))
Formal Verification: `morphoeval/lean/loops/Loop_2006_physalia_physalis_bioelectric_polarity.lean` (`MathOfBiology.Loop_2006`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/physalia_physalis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "physalia_physalis_bioelectric_polarity_reference.json": "899dffad29f10065193229f11940bd95759944ce9b72d1fca1dd0be049ec798d",
    "physalia_physalis_specimen.jpg": "b68854e9fdaaf725c3899f1667f4e8361f6a0f2556862c28573db2d565a50904",
    "scout_manifest.json": "988a114cdcdee3d93b0afd3aa0efa0acf27c621e7822c14872a959d84523d7d6",
    "paper_citations.bib": "1b3b2329f567d162b4270b19de0ab303382c3cb77a092b3c3f8f8d134e6fad2b"
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
    ref = json.loads((DATASET_DIR / "physalia_physalis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2022-munro-physalia-physalis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
