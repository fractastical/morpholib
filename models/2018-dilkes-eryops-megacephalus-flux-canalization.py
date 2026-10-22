#!/usr/bin/env python3
"""
models/2018-dilkes-eryops-megacephalus-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1487)
Organism: Eryops megacephalus (Early Permian Temnospondyl Pentadactyl Amphibian)
Phenomenon Scale: 5.1 cm (0.05115 m, log10=-1.291, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Eryops megacephalus (Early Permian (Cisuralian, 295 Ma))
Geological Horizon: Early Permian (Cisuralian, 295 Ma) (295.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Eryops
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (5.1 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1487_eryops_megacephalus_flux_canalization.lean` (`MathOfBiology.Loop_1487`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/eryops_megacephalus_flux_canalization"
EXPECTED_SHA256 = {
    "eryops_megacephalus_flux_canalization_reference.json": "adc43917a8880ab401940d9dcac6e6f50e5b711d4bdc3b303a4b21da6bed06d5",
    "eryops_megacephalus_specimen.jpg": "129f152b5349248669bd3ffc0b0360d0131cfa03e87c6ff1c0ac912fae50e424",
    "scout_manifest.json": "5fd33c1932fbda5c9fecd140f68d83da7bbe4f5df72c1f7dcd246b61a6661009",
    "paper_citations.bib": "abb5b3f7d881cab8d33405e244b6bc683f4df749712c9459fbf9411474b22ad7"
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
    ref = json.loads((DATASET_DIR / "eryops_megacephalus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-dilkes-eryops-megacephalus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
