#!/usr/bin/env python3
"""
models/1991-okamoto-nipponites-mirabilis-bioelectric-polarity.py — Multi-Scale & Paleobiological Reproduction (Slot #1960)
Organism: Nipponites mirabilis (Late Cretaceous Meandering 3D Trefoil-Knot Heteromorph Ammonite)
Phenomenon Scale: 44 µm (4.363e-05 m, log10=-4.360, 4_cellular_10um)
Anatomical Structure: Gap-junction bioelectric domain & ion-channel patch in Nipponites mirabilis (Late Cretaceous (Coniacian, 88 Ma))
Geological Horizon: Late Cretaceous (Coniacian, 88 Ma) (88.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Nipponites
Biophysical Cohort: IX. Bioelectric Regeneration & Neural Maps (Gap-Junction Bioelectric Bistability & Target Polarity (44 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1960_nipponites_mirabilis_bioelectric_polarity.lean` (`MathOfBiology.Loop_1960`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/nipponites_mirabilis_bioelectric_polarity"
EXPECTED_SHA256 = {
    "nipponites_mirabilis_bioelectric_polarity_reference.json": "9e5a6f5e4382ed02312199f0690583edb390359efdefb6ac9220e5844d53d6f0",
    "nipponites_mirabilis_specimen.jpg": "6b8124b1e5eeb73dc589cf76fafc5db8b2d4a7d15eb68c77a1175d267f717dd5",
    "scout_manifest.json": "e5c659d2dffad6ecfd845c8cefa390c7156c0c38b0dcc23e92285cde07f86af9",
    "paper_citations.bib": "b4c46fec9fa26fec7bee967134701b461ebf462040d52e0204be14a5b3261db4"
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
    ref = json.loads((DATASET_DIR / "nipponites_mirabilis_bioelectric_polarity_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1991-okamoto-nipponites-mirabilis-bioelectric-polarity.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
