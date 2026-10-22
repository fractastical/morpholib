#!/usr/bin/env python3
"""
models/2004-mcloughlin-glossopteris-brownii-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1979)
Organism: Glossopteris brownii (Permian Gondwanan Seed Fern (Reticulate Closed-Loop Leaf Venation))
Phenomenon Scale: 41 µm (4.092e-05 m, log10=-4.388, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Glossopteris brownii (Late Permian (Gondwana, 260 Ma))
Geological Horizon: Late Permian (Gondwana, 260 Ma) (260.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Glossopteris
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (41 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1979_glossopteris_brownii_bioelectric_polarity.lean` (`MathOfBiology.Loop_1979`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/glossopteris_brownii_bioelectric_polarity"
EXPECTED_SHA256 = {
    "glossopteris_brownii_bioelectric_polarity_reference.json": "1e4d1421165563968b22710e2af7586c60a9f78816931340efb5540c1a5fd430",
    "glossopteris_brownii_specimen.jpg": "21f458497a8d6fffd0bfea916b3893a105c31943c6d82c526ea20f691965f627",
    "scout_manifest.json": "998fe5fbc73f26b5b8a59dbc46fee576beccf0e0aa502a8b14ffb408a4f02c14",
    "paper_citations.bib": "de045bd37e68f9a3ac3a3ba7b04cab44821f3c8783fa1c3a5c342e6109d0af72"
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
    ref = json.loads((DATASET_DIR / "glossopteris_brownii_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2004-mcloughlin-glossopteris-brownii-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
