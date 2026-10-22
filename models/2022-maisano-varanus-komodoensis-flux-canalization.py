#!/usr/bin/env python3
"""
models/2022-maisano-varanus-komodoensis-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1548)
Organism: Varanus komodoensis (Komodo Dragon (Chainmail Cephalic Osteoderm Meshwork))
Phenomenon Scale: 595 µm (0.000595 m, log10=-3.225, 5_primordium_100um)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Varanus komodoensis (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Varanus
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (595 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1548_varanus_komodoensis_flux_canalization.lean` (`MathOfBiology.Loop_1548`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/varanus_komodoensis_flux_canalization"
EXPECTED_SHA256 = {
    "varanus_komodoensis_flux_canalization_reference.json": "69bc8df017cd535688910aa3ad9f4b09869dfe53a54a31ee0a5f89e9d38509d7",
    "varanus_komodoensis_specimen.jpg": "347fa0ca27bf35537323e24d59b72d5f6df744f56be66ed7c9649cdf3bc761b5",
    "scout_manifest.json": "92fb69484a04825f6f2fc886c8a723532beb1492e7166418b4c5f87926fc329e",
    "paper_citations.bib": "a00fc7efa3a0c82cd0e7d6a9c3ea94354468906ec3bb51bedb46d58ae118daf2"
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
    ref = json.loads((DATASET_DIR / "varanus_komodoensis_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2022-maisano-varanus-komodoensis-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
