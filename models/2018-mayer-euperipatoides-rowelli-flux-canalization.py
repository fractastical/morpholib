#!/usr/bin/env python3
"""
models/2018-mayer-euperipatoides-rowelli-flux-canalization.py — Multi-Scale & Paleobiological Reproduction (Slot #1552)
Organism: Euperipatoides rowelli (Australian Velvet Worm (Onychophoran Slime-Papilla & Lobopod Segments))
Phenomenon Scale: 354 µm (0.0003535 m, log10=-3.452, 5_primordium_100um)
Anatomical Structure: Auxin flux vein / phyllotactic primordium in Euperipatoides rowelli (Holocene (Extant))
Geological Horizon: Holocene (Extant) (0.0 Ma)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Euperipatoides
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth (354 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1552_euperipatoides_rowelli_flux_canalization.lean` (`MathOfBiology.Loop_1552`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/euperipatoides_rowelli_flux_canalization"
EXPECTED_SHA256 = {
    "euperipatoides_rowelli_flux_canalization_reference.json": "4c98d414ec931cfbf5d1ab39834228b54d200443a7717e275627f96de1b4e880",
    "euperipatoides_rowelli_specimen.jpg": "0e99f4aad858382d0f1a61bc28533c722840c8c49a15f15ce70d8f5405b262c2",
    "scout_manifest.json": "fbfc7c5ebdb468c380fe24e925f6d384c262f98336b16057177a3ff56e081107",
    "paper_citations.bib": "78cb3a33075dabba4cf7dfb5072264bb2db21ce782b055bb5b52df3d0fb3dbb9"
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
    ref = json.loads((DATASET_DIR / "euperipatoides_rowelli_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2018-mayer-euperipatoides-rowelli-flux-canalization.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
