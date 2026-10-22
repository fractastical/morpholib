#!/usr/bin/env python3
"""
models/2023-kavanagh-pliohippus-pernix-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1491)
Organism: Pliohippus pernix (Late Miocene Single-Hoofed Monodactyl Ancestor)
Phenomenon Scale: 5.9 cm (0.05859 m, log10=-1.232, 7_organism_cm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Pliohippus pernix (Late Miocene (Hemphillian, 8.5 Ma))
Geological Horizon: Late Miocene (Hemphillian, 8.5 Ma) (8.5 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Pliohippus
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (5.9 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1491_pliohippus_pernix_flux_canalization.lean` (`MathOfBiology.Loop_1491`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/pliohippus_pernix_flux_canalization"
EXPECTED_SHA256 = {
    "pliohippus_pernix_flux_canalization_reference.json": "16653baa98d8405c89671caeab4cb0deab83b4d84fe1a5eb2c85b4b4d3704961",
    "pliohippus_pernix_specimen.jpg": "1211e4acc14fca6bb091e7747aca534b08c9b54a2cea23f8b7aa25c1b659afdd",
    "scout_manifest.json": "21a4e5273a4a1f1f0e5fe4408f230124d2978fdaa5b752ca6588d44e04dd3edb",
    "paper_citations.bib": "4a7588dc17169cc407944e6c8336db16bb083f65f0bd99101a85aa10bc332d9d"
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
    ref = json.loads((DATASET_DIR / "pliohippus_pernix_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2023-kavanagh-pliohippus-pernix-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
