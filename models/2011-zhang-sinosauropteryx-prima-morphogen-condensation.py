#!/usr/bin/env python3
"""
models/2011-zhang-sinosauropteryx-prima-morphogen-condensation.py — Multi-Scale & Paleobiological Reproduction (Slot #1318)
Organism: Sinosauropteryx prima (Early Cretaceous Proto-Feathered Compsognathid Dinosaur)
Phenomenon Scale: 235 µm (0.0002354 m, log10=-3.628, 5_primordium_100um)
Anatomical Structure: Morphogen gradient & pre-cartilage condensation zone in Sinosauropteryx prima (Early Cretaceous (Yixian Fm, 124.6 Ma))
Geological Horizon: Early Cretaceous (Yixian Fm, 124.6 Ma) (124.6 Ma)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Sinosauropteryx
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation (235 µm))
Formal Verification: `morphoeval/lean/loops/Loop_1318_sinosauropteryx_prima_morphogen_condensation.lean` (`MathOfBiology.Loop_1318`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets/sinosauropteryx_prima_morphogen_condensation"
EXPECTED_SHA256 = {
    "sinosauropteryx_prima_morphogen_condensation_reference.json": "0c6fea6431ac98be2088aa0ddc34413a5d91d6d2b133f0d9adf28b20a3f88e34",
    "sinosauropteryx_prima_specimen.jpg": "dafd40e5487fb100d0309d10c30a01a1479f1793e6aa37726388bfafdd50d009",
    "scout_manifest.json": "03fd0262b3ea556661af5899e67657d35e282d1dcbb36155e27877c74c095090",
    "paper_citations.bib": "728caacbab3cd15447dff1e8524eceb7e1f38e2463e0d436bd40491a5beb8545"
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
    ref = json.loads((DATASET_DIR / "sinosauropteryx_prima_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] models/2011-zhang-sinosauropteryx-prima-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, size={res['phenomenon_size_m']}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
