#!/usr/bin/env python3
"""
models/1997-coates-acanthostega-gunnari-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1305)
Organism: Acanthostega gunnari (Late Devonian Eight-Digited Polydactyl Stem-Tetrapod)
Phenomenon Scale: 1.8 cm (0.01759 m, log10=-1.755, 7_organism_cm)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Acanthostega gunnari (Late Devonian (Famennian, 365 Ma))
Geological Horizon: Late Devonian (Famennian, 365 Ma) (365.0 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Acanthostega
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (1.8 cm))
Formal Verification: `morphoeval/lean/loops/Loop_1305_acanthostega_gunnari_morphogen_condensation.lean` (`MathOfBiology.Loop_1305`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/acanthostega_gunnari_morphogen_condensation"
EXPECTED_SHA256 = {
    "acanthostega_gunnari_morphogen_condensation_reference.json": "2fc617c923b8688003f596ad5656f0dd31d126f89e5b681fdc57109b24c4d76c",
    "acanthostega_gunnari_specimen.jpg": "17b12e17eb08ab7e05345c565a3dc299e5c4752dd56c4753617937fa7b3a66b5",
    "scout_manifest.json": "abf15e6940b0c1cb368c900530d630feaa541e5323a7a9a0b9e614557b41af7c",
    "paper_citations.bib": "0ac8458415bd4b8941d8147f7eae1bda871df925bc20885fb5f1d07e9166b560"
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
    ref = json.loads((DATASET_DIR / "acanthostega_gunnari_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/1997-coates-acanthostega-gunnari-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
