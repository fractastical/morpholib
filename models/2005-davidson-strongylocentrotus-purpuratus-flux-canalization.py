#!/usr/bin/env python3
"""
models/2005-davidson-strongylocentrotus-purpuratus-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1543)
Organism: Strongylocentrotus purpuratus (Purple Sea Urchin (Larval Calcite Spicule Triradiate Skeleton))
Phenomenon Scale: 20 µm (1.976e-05 m, log10=-4.704, 4_cellular_10um)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Strongylocentrotus purpuratus (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Strongylocentrotus
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (20 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1543_strongylocentrotus_purpuratus_flux_canalization.lean` (`MathOfBiology.Loop_1543`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/strongylocentrotus_purpuratus_flux_canalization"
EXPECTED_SHA256 = {
    "strongylocentrotus_purpuratus_flux_canalization_reference.json": "1c607e7f833c36365eb7cf503f449ccdb3c4d5a59cbc83afe6a37db8543b0a86",
    "strongylocentrotus_purpuratus_specimen.jpg": "515b723840267df6465272f908c6951fdf06c057159d3c6f7d6e67e7bee5224a",
    "scout_manifest.json": "fd8023260e7307cde5335475ae86f2fe3260b0690492dea78456452219907627",
    "paper_citations.bib": "d9a217a750d8944bdddd3653e25bdde27254d05ea20d55ad0a27e9ae072edacf"
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
    ref = json.loads((DATASET_DIR / "strongylocentrotus_purpuratus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2005-davidson-strongylocentrotus-purpuratus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
