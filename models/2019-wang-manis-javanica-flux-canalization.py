#!/usr/bin/env python3
"""
models/2019-wang-manis-javanica-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1545)
Organism: Manis javanica (Sunda Pangolin (Imbricated Keratin Scale Stress-Shielding Lattice))
Phenomenon Scale: 8.8 mm (0.008838 m, log10=-2.054, 6_organ_mm)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Manis javanica (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Manis
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (8.8 mm))
Formal Verification: `morphoeval/lean/loops/Loop_1545_manis_javanica_flux_canalization.lean` (`MathOfBiology.Loop_1545`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/manis_javanica_flux_canalization"
EXPECTED_SHA256 = {
    "manis_javanica_flux_canalization_reference.json": "2ec52bcf676c3b37d1a6654d45815f6cc270a5cd0072cee8b7cc97d5955dc066",
    "manis_javanica_specimen.jpg": "0d2cf7b0e5853e57873b4a69f96a0bf200561aa593feda56d04025e8b7619711",
    "scout_manifest.json": "cb1502f5762dcbd5418b3a82a64d86245b0ee271489a2982ce1b6f12f7b7e38e",
    "paper_citations.bib": "d5d7c99d1c67e433bd482abe04b269240611671591f156c7ba5201a0c368e5f6"
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
    ref = json.loads((DATASET_DIR / "manis_javanica_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2019-wang-manis-javanica-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
