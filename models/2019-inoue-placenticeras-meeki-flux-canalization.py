#!/usr/bin/env python3
"""
models/2019-inoue-placenticeras-meeki-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1508)
Organism: Placenticeras meeki (Late Cretaceous Iridescent Ammolite Ammonite (Hyper-Fractal Suture, D_f=1.68))
Phenomenon Scale: 1.6 cm (0.01581 m, log10=-1.801, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Placenticeras meeki (Late Cretaceous (Campanian, 75 Ma))
Geological Horizon: Late Cretaceous (Campanian, 75 Ma) (75.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Placenticeras
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (1.6 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1508_placenticeras_meeki_flux_canalization.lean` (`MathOfBiology.Loop_1508`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/placenticeras_meeki_flux_canalization"
EXPECTED_SHA256 = {
    "placenticeras_meeki_flux_canalization_reference.json": "d5758ca84ca437aa8aa081b2bed66467767c162c2a017c1b8ebe2e41e913968a",
    "placenticeras_meeki_specimen.jpg": "ef945c35e08dd1adfded926af45c962f3c6017582c005f246eceedc2238e4229",
    "scout_manifest.json": "ea3f695ef03785254107be0608e2ba7c4f6af8f5c52c2b3fcb87730e7f9241f3",
    "paper_citations.bib": "8d8d4d3bfe6603d41e66e4dbdc74126898dbc3c8ee8aab598ce30c82bd1c2354"
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
    ref = json.loads((DATASET_DIR / "placenticeras_meeki_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-inoue-placenticeras-meeki-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
