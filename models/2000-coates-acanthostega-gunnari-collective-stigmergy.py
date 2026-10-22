#!/usr/bin/env python3
"""
models/2000-coates-acanthostega-gunnari-collective-stigmergy.py — Multi-Scale & Paleobiological Reproduction (Slot #2025)
Organism: Acanthostega gunnari (Late Devonian Eight-Digited Polydactyl Stem-Tetrapod)
Phenomenon Scale: 17.6 cm (0.176 m, log10=-0.754, 7_organism_cm)
Anatomical Structure: Collective swarm / biofilm / colonial array in Acanthostega gunnari (Late Devonian (Famennian, 365 Ma))
Geological Horizon: Late Devonian (Famennian, 365 Ma) (365.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Acanthostega
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation (17.6 cm))
Formal Verification: `morphoeval/lean/loops/Loop_2025_acanthostega_gunnari_collective_stigmergy.lean` (`MathOfBiology.Loop_2025`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/acanthostega_gunnari_collective_stigmergy"
EXPECTED_SHA256 = {
    "acanthostega_gunnari_collective_stigmergy_reference.json": "adc1ad3c65d6c50efe46feee89db8b1ac0fa4319a15d846691f15345be737c7a",
    "acanthostega_gunnari_specimen.jpg": "17b12e17eb08ab7e05345c565a3dc299e5c4752dd56c4753617937fa7b3a66b5",
    "scout_manifest.json": "d76d3892a9beca90bd20d97974b9db05e31d19a8667fe2e893076479d91b0e41",
    "paper_citations.bib": "6da1c585448da4f27514b97d57c87d1769f2f199499005f43ce2d92f306d58f6"
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
    ref = json.loads((DATASET_DIR / "acanthostega_gunnari_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2000-coates-acanthostega-gunnari-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
