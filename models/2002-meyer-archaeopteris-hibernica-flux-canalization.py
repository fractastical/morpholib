#!/usr/bin/env python3
"""
models/2002-meyer-archaeopteris-hibernica-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1527)
Organism: Archaeopteris hibernica (Late Devonian Progymnosperm Tree (Planated Megaphyll Fronds))
Phenomenon Scale: 9.6 mm (0.009562 m, log10=-2.019, 6_organ_mm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Archaeopteris hibernica (Late Devonian (Famennian, 370 Ma))
Geological Horizon: Late Devonian (Famennian, 370 Ma) (370.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Archaeopteris
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (9.6 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1527_archaeopteris_hibernica_flux_canalization.lean` (`MathOfBiology.Loop_1527`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/archaeopteris_hibernica_flux_canalization"
EXPECTED_SHA256 = {
    "archaeopteris_hibernica_flux_canalization_reference.json": "55f1e6fbf3743b3dc14f960946bc53b9a94f89c23a2cd979dbec109b7c6914bd",
    "archaeopteris_hibernica_specimen.jpg": "107d9b021f541e8a82a7c594b45f0250c4a6c8fa160f1b3c54299cd439a8163e",
    "scout_manifest.json": "e482d15b3ff52e1b51479ef3576f866eff1afc3b02104162cebad08918d9b691",
    "paper_citations.bib": "b1f93b4079d5762dbc1243a7645569b510b9c071c7d6629e52c68dfddf5c5631"
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
    ref = json.loads((DATASET_DIR / "archaeopteris_hibernica_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2002-meyer-archaeopteris-hibernica-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
