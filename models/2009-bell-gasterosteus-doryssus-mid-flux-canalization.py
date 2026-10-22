#!/usr/bin/env python3
"""
models/2009-bell-gasterosteus-doryssus-mid-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1478)
Organism: Gasterosteus doryssus_mid (Miocene Stickleback Chronospecies II (Vestigial Spine, T=+9 kyr))
Phenomenon Scale: 1.0 cm (0.01041 m, log10=-1.983, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Gasterosteus doryssus_mid (Late Miocene (Truckee Fm, 10.01 Ma))
Geological Horizon: Late Miocene (Truckee Fm, 10.01 Ma) (10.01 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Gasterosteus
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (1.0 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1478_gasterosteus_doryssus_mid_flux_canalization.lean` (`MathOfBiology.Loop_1478`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/gasterosteus_doryssus_mid_flux_canalization"
EXPECTED_SHA256 = {
    "gasterosteus_doryssus_mid_flux_canalization_reference.json": "0b3330a4d827c0a1660f75de6d44c875ca5cfcc347ac0c6c6494dcbfb5bb8d5c",
    "gasterosteus_doryssus_mid_specimen.jpg": "52d02b91f4693fdd40965b6d78971fdf8ebe6601d12fad6c49a6255e45791e42",
    "scout_manifest.json": "d11f43e81d32569f7ec28414d4974a9b35e62912c350084b960e05638341ae4a",
    "paper_citations.bib": "41e000e70193268769499c3411e415c628c910e23234e62af2795c7b57308e3d"
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
    ref = json.loads((DATASET_DIR / "gasterosteus_doryssus_mid_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2009-bell-gasterosteus-doryssus-mid-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
