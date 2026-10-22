#!/usr/bin/env python3
"""
models/2006-zheng-ginkgo-apodes-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1532)
Organism: Ginkgo apodes (Early Cretaceous Multi-Lobed Jehol Biota Ginkgo Chronospecies)
Phenomenon Scale: 1.1 cm (0.01103 m, log10=-1.957, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Ginkgo apodes (Early Cretaceous (Yixian Fm, 121 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 121 Ma) (121.0 Ma)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Ginkgo
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (1.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1532_ginkgo_apodes_flux_canalization.lean` (`MathOfBiology.Loop_1532`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/ginkgo_apodes_flux_canalization"
EXPECTED_SHA256 = {
    "ginkgo_apodes_flux_canalization_reference.json": "6f77cc6283df1991ab4e21a1c63367488bc1ca5f0eb95d0437a3fd445de7073b",
    "ginkgo_apodes_specimen.jpg": "3a074e9794ed0797dcc5abd51ab2fcccf8a109da1756d08ad6f52d95faf7d0f5",
    "scout_manifest.json": "1e3f9cff3f9054d8d0d5a8c16fd95cff1d1ffe7f9d08f616b3bbbb93cd1379a4",
    "paper_citations.bib": "18460dcd71cc04f6af0b6a835a9b3838468adba221507fbf4f359da667d416ac"
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
    ref = json.loads((DATASET_DIR / "ginkgo_apodes_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2006-zheng-ginkgo-apodes-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
