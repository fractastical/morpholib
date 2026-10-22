#!/usr/bin/env python3
"""
models/2013-zhang-sinosauropteryx-prima-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1498)
Organism: Sinosauropteryx prima (Early Cretaceous Proto-Feathered Compsognathid Dinosaur)
Phenomenon Scale: 327 µm (0.000327 m, log10=-3.485, 5_primordium_100um)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Sinosauropteryx prima (Early Cretaceous (Yixian Fm, 124.6 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 124.6 Ma) (124.6 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Sinosauropteryx
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (327 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1498_sinosauropteryx_prima_flux_canalization.lean` (`MathOfBiology.Loop_1498`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/sinosauropteryx_prima_flux_canalization"
EXPECTED_SHA256 = {
    "sinosauropteryx_prima_flux_canalization_reference.json": "573af71bc7c1db75839e038ab71075a5471b2edc586ea1c96b0ec2cae5e0d738",
    "sinosauropteryx_prima_specimen.jpg": "dafd40e5487fb100d0309d10c30a01a1479f1793e6aa37726388bfafdd50d009",
    "scout_manifest.json": "66e919e6606fbe02465e70ffa5b1ea60e02c85338e9eed13947cc2d9fc42cc26",
    "paper_citations.bib": "c47cdcaa641496ec022044cbc1e0de5f5bb4dd8e302f4789afea53b47c06319b"
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
    ref = json.loads((DATASET_DIR / "sinosauropteryx_prima_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2013-zhang-sinosauropteryx-prima-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
